"""Identify moles."""

import collections
import functools
import heapq

import cv2
import numpy
import scipy.linalg
import scipy.stats

import mel.lib.moleimaging


# Individual probabilities less than this are out of consideration.
_MAGIC_P_THRESHOLD = 0.00001

_MAGIC_CLOSE_DISTANCE = 0.1


def p_to_cost(p):
    # return int(10 / p) - 9
    return int(1 / p)


class PriorityQueue():

    def __init__(self):
        self.heap = []
        self.next_tie_breaker = 0

    def push(self, priority, value):

        heapq.heappush(
            self.heap,
            (priority, self.next_tie_breaker, value))

        self.next_tie_breaker += 1

    def pop(self):
        priority, _, value = heapq.heappop(self.heap)
        return priority, value

    def __len__(self):
        return len(self.heap)

    def __str__(self):
        return ("<PriorityQueue:: len:{}>".format(len(self.heap)))


class StatePriorityQueue():

    def __init__(self):
        self.queue = PriorityQueue()
        self.already_tried = set()

    def push(self, estimate, value, state):

        # normalised = tuple(sorted(state.items()))
        # if normalised in self.already_tried:
        #     return
        # self.already_tried.add(normalised)

        self.queue.push((estimate, value), state)

    def pop(self):
        (estimate, value), state = self.queue.pop()
        return estimate, value, state

    def __len__(self):
        return len(self.queue)

    def __str__(self):
        return ("<StatePriorityQueue:: len:{}>".format(len(self.queue)))


class Guesser():

    def __init__(
            self,
            uuid_to_pos,
            cold_classifier,
            warm_classifier,
            canonical_uuid_set):

        self.uuid_to_pos = uuid_to_pos
        self.cold_classifier = cold_classifier
        self.warm_classifier = warm_classifier
        self.canonical_uuid_set = canonical_uuid_set
        self.init_a_to_bc()

        # Note that we want one cache per instance of Guesser. This means that
        # the values will be correct for the classifiers and positions
        # provided. Therefore we must create these dynamically.

        @functools.lru_cache(maxsize=1024)
        def warm(uuid_for_history, uuid_for_position, uuid_to_guess):
            ref_pos = self.uuid_to_pos[uuid_for_position]
            pos = self.uuid_to_pos[uuid_to_guess]
            return tuple(
                (b, p_to_cost(p * q))
                for b, p, q in self.warm_classifier(
                    uuid_for_history, ref_pos, pos)
                if _MAGIC_P_THRESHOLD < p * q  #and not numpy.isnan(p * q)
                # if not numpy.isclose(0, p * q) and not numpy.isnan(p * q)
            )
        self.warm = warm

        @functools.lru_cache(maxsize=128)
        def closest_uuids(uuid_for_position):
            ref_pos = self.uuid_to_pos[uuid_for_position]
            sqdist_uuid_list = sorted(
                (mel.lib.math.distance_sq_2d(pos, ref_pos), uuid_)
                for uuid_, pos in self.uuid_to_pos.items()
                if uuid_ != uuid_for_position
            )
            return tuple(uuid_ for _, uuid_ in sqdist_uuid_list)
        self.closest_uuids = closest_uuids

    def init_a_to_bc(self):
        uuid_to_numclose = uuidtopos_to_numclose(
            self.uuid_to_pos)

        self.a_to_bc = collections.defaultdict(dict)
        for a, pos in self.uuid_to_pos.items():
            numclose = uuid_to_numclose[a]
            for b, p, q in self.cold_classifier(pos[1], numclose):
                r = p * q
                if r > 1:
                    raise ValueError(
                        f"'r' must be equal to or less than 1, "
                        "got: a={a}, b={b}, r={r}")
                if not numpy.isclose(0, r):
                    cost = p_to_cost(r)
                    self.a_to_bc[a][b] = cost

    def initial_state(self):
        return {
            a: a if a in self.canonical_uuid_set else None
            for a in self.a_to_bc
        }

    def yield_next_states(self, _est_cost, total_cost, state):
        filled = {a: b for a, b in state.items() if b is not None}

        if not filled:
            yield from self.yield_next_states_cold(
                _est_cost, total_cost, state)
            return

        already_taken = set(filled.values())
        total_est, a_to_est = self.estimates(state, already_taken)

        for a, b in state.items():
            if b is not None:
                continue

            uuid_for_pos = next(
                uuid_
                for uuid_ in self.closest_uuids(a)
                if state[uuid_] is not None)
            uuid_for_history = state[uuid_for_pos]

            est_without_a = total_est // a_to_est[a]

            num_added = 0
            for b, cost in self.warm(uuid_for_history, uuid_for_pos, a):
                if b not in already_taken:
                    new_cost = total_cost * cost
                    new_state = dict(state)
                    new_state[a] = b

                    new_est = est_without_a * cost * total_cost

                    yield new_est, new_cost, new_state
                    num_added += 1

            # if not num_added:
            #     # If we ran out of moles to match against, this must be a new
            #     # mole. This isn't the only way we can detect a new mole.
            #     # raise NotImplementedError("New mole!")
            #     new_state = dict(state)
            #     new_state[a] = 'NewMole'
            #     yield total_cost, total_cost, new_state

    def estimates(self, state, already_taken):

        total_est = 1
        a_to_est = {}

        for a, b in state.items():
            if b is not None:
                continue

            uuid_for_pos = next(
                uuid_
                for uuid_ in self.closest_uuids(a)
                if state[uuid_] is not None)
            uuid_for_history = state[uuid_for_pos]

            best_cost = None
            for b, cost in self.warm(uuid_for_history, uuid_for_pos, a):
                if b not in already_taken:
                    if best_cost is None or cost < best_cost:
                        best_cost = cost

            if best_cost is None:
                a_to_est[a] = 1
            else:
                a_to_est[a] = best_cost
                total_est *= best_cost

            # XXX: Temp hack to test without estimates
            a_to_est[a] = 1
            total_est = 1

        return total_est, a_to_est

    def yield_next_states_cold(self, _, total_cost, state):
        best_estimates, new_est_cost = self.estimates_cold(total_cost, state)

        for a, b in state.items():
            if b is not None:
                raise Exception('b must be None')

            num_added = 0
            base_est_cost = new_est_cost // best_estimates[a]
            for b, cost in self.a_to_bc[a].items():
                new_cost = total_cost * cost
                est_cost = base_est_cost * cost
                new_state = dict(state)
                new_state[a] = b
                yield est_cost, new_cost, new_state
                num_added += 1

            if not num_added:
                # If we ran out of moles to match against, this must be a new
                # mole. This isn't the only way we can detect a new mole.
                # raise NotImplementedError("New mole!")
                new_state = dict(state)
                new_state[a] = 'NewMole'
                yield cost, cost, new_state

    def estimates_cold(self, total_cost, state):
        best_estimates = {}
        new_est_cost = total_cost
        for a, b in state.items():
            if b is not None:
                raise Exception('b must be None')
            best_cost = None
            for _, cost in self.a_to_bc[a].items():
                if best_cost is None or cost > best_cost:
                    best_cost = cost
            if best_cost is not None:
                best_estimates[a] = best_cost
                new_est_cost *= best_cost
            else:
                best_estimates[a] = 1

        return best_estimates, new_est_cost

    def yield_next_states_cold(self, _est_cost, total_cost, state):
        for a, b in state.items():
            if b is not None:
                raise Exception('b must be None')

            num_added = 0
            for b, cost in self.a_to_bc[a].items():
                new_cost = total_cost * cost
                new_state = dict(state)
                new_state[a] = b
                yield new_cost, new_cost, new_state
                num_added += 1

            if not num_added:
                # If we ran out of moles to match against, this must be a new
                # mole. This isn't the only way we can detect a new mole.
                # raise NotImplementedError("New mole!")
                new_state = dict(state)
                new_state[a] = 'NewMole'
                yield cost, cost, new_state

    # Automatically guess all the first ones correctly, just for testing.
    # def yield_next_states_cold(self, _est_cost, total_cost, state):
    #     for a, b in state.items():
    #         if b is not None:
    #             raise Exception('b must be None')
    #         new_state = dict(state)
    #         new_state[a] = a
    #         yield 1, 1, new_state


def best_match_combination(guesser, *, max_iterations=10**5):

    state_q = StatePriorityQueue()
    state_q.push(1, 1, guesser.initial_state())

    deepest = 0
    most_correct = 0
    count = 0
    while count != max_iterations:

        if not state_q:
            # If we ran out of moles to match against, this must be a new mole.
            # This isn't the only way we can detect a new mole.
            raise NotImplementedError("Ran out of options!")

        # Advance best state.
        est_cost, total_cost, state = state_q.pop()

        count += 1
        should_report = 0 == count % 10000
        depth = sum(1 for a, b in state.items() if b is not None)
        correct = sum(1 for a, b in state.items() if a == b)
        if depth > deepest:
            deepest = depth
            should_report = True
        if correct > most_correct:
            most_correct = correct
            should_report = True
        if should_report:
            print(
                count,
                (est_cost, total_cost),
                depth,
                correct
            )

        # See if we're done.
        if all(state.values()):
            print(
                count,
                (est_cost, total_cost),
                depth,
                correct
            )
            return total_cost, state

        # Nope, advance states.
        for new_est_cost, new_cost, new_state in guesser.yield_next_states(
                est_cost, total_cost, state):
            state_q.push(new_est_cost, new_cost, new_state)

        if not state_q:
            # This is the last and apparently best option.
            print('Final option')
            print(
                count,
                (est_cost, total_cost),
                depth,
                correct
            )
            # TODO: mark new moles or update contract to say some can be None
            return total_cost, state

    raise LookupError(
        f'Could not find a best match in under {max_iterations:,} iterations.')


class ColdGuessMoleClassifier():

    def __init__(self, uuid_to_frameposlist, ypos_radius, neighbour_radius):
        self.uuid_to_frameposlist = uuid_to_frameposlist
        uuid_to_poslist = {
            uuid_: [pos for frame, pos in frameposlist]
            for uuid_, frameposlist in uuid_to_frameposlist.items()
        }
        uuid_to_yposlist = {
            uuid_: numpy.array(tuple(y for x, y in poslist))
            for uuid_, poslist in uuid_to_poslist.items()
        }

        self.uuids = tuple(uuid_to_poslist.keys())

        self.frames = collections.defaultdict(dict)
        for uuid_, frameposlist in uuid_to_frameposlist.items():
            for frame, pos in frameposlist:
                self.frames[frame][uuid_] = pos

        uuid_to_neighbourlist = collections.defaultdict(list)
        for uuid_to_pos in self.frames.values():
            for uuid_, num_close in uuidtopos_to_numclose(uuid_to_pos).items():
                uuid_to_neighbourlist[uuid_].append(num_close)

        self.uuid_to_neighbourlist = uuid_to_neighbourlist

        self.kernels = tuple(
            Kde(
                numpy.vstack([
                    numpy.array(uuid_to_yposlist[uuid_]),
                    numpy.array(uuid_to_neighbourlist[uuid_])
                ])
            )
            for uuid_ in self.uuids
        )

        self.lower = numpy.array((-ypos_radius, -neighbour_radius))
        self.upper = numpy.array((ypos_radius, neighbour_radius))

    def __call__(self, ypos, numclose):

        if not numpy.isscalar(ypos):
            raise ValueError(f"'ypos' must be a scalar, got '{ypos}'.")

        if not numpy.isscalar(numclose):
            raise ValueError(f"'numclose' must be a scalar, got '{numclose}'.")

        a = numpy.array((ypos, numclose))

        densities = numpy.array(
            tuple(
                k(a + self.lower, a + self.upper)
                for k in self.kernels
            )
        )

        total_density = numpy.sum(densities)

        if numpy.isclose(total_density, 0):
            total_density = -1

        matches = []
        for i, m_uuid in enumerate(self.uuids):
            p = densities[i]
            q = p / total_density
            matches.append((m_uuid, p, q))

        return matches


class MoleRelativeClassifier():

    def __init__(self, uuid_to_frameposlist, box_radius):

        uuid_to_poslist = {
            uuid_: [pos for frame, pos in frameposlist]
            for uuid_, frameposlist in uuid_to_frameposlist.items()
        }

        self.uuids, _ = zip(*uuid_to_poslist.items())

        self.frames = collections.defaultdict(dict)
        for uuid_, frameposlist in uuid_to_frameposlist.items():
            for frame, pos in frameposlist:
                self.frames[frame][uuid_] = pos

        self.lower = numpy.array((box_radius, box_radius))
        self.upper = -self.lower

        self.molepos_kernels = {}

    def calc_uuid_offset_kernels(self, known_uuid):
        uuid_to_xoffsetlist = collections.defaultdict(list)
        uuid_to_yoffsetlist = collections.defaultdict(list)
        for uuid_to_pos in self.frames.values():
            if known_uuid not in uuid_to_pos:
                continue

            center = uuid_to_pos[known_uuid]
            offsets = {
                uuid_: kpos - center
                for uuid_, kpos in uuid_to_pos.items()
                if uuid_ != known_uuid
            }
            for uuid_, offset in offsets.items():
                uuid_to_xoffsetlist[uuid_].append(offset[0])
                uuid_to_yoffsetlist[uuid_].append(offset[1])

        return tuple(
            Kde(
                numpy.vstack([
                    numpy.array(uuid_to_xoffsetlist[uuid_]),
                    numpy.array(uuid_to_yoffsetlist[uuid_])
                ])
            )
            for uuid_ in self.uuids
        )

    def __call__(self, known_uuid, known_pos, pos):

        # TODO: check that known_uuid looks like a uuid
        if known_pos.shape != (2,):
            raise ValueError(
                f'known_pos must be 2d, not {known_pos.shape}')
        if pos.shape != (2,):
            raise ValueError(f'pos must be 2d, not {pos.shape}')

        if not known_uuid in self.molepos_kernels:
            self.molepos_kernels[known_uuid] = self.calc_uuid_offset_kernels(
                known_uuid)

        kernels = self.molepos_kernels[known_uuid]

        rpos = pos - known_pos

        densities = numpy.array(
            tuple(
                k(rpos + self.lower, rpos + self.upper)
                for k in kernels
            )
        )

        total_density = numpy.sum(densities)
        if numpy.isclose(total_density, 0):
            total_density = -1

        matches = []
        for i, m_uuid in enumerate(self.uuids):
            p = densities[i]
            q = p / total_density
            matches.append((m_uuid, p, q))

        return matches


def frame_to_uuid_to_pos(frame):
    mask = frame.load_mask()
    contour = biggest_contour(mask)
    ellipse = cv2.fitEllipse(contour)
    elspace = EllipseSpace(ellipse)

    uuid_to_pos = {
        uuid_: elspace.to_space(pos)
        for uuid_, pos in frame.moledata.uuid_points.items()
    }

    return uuid_to_pos


class Kde():

    def __init__(self, training_data):
        self.len = training_data.shape[-1]

        if self.len < 3:
            self.attenuation = 0.0
            self.kde = lambda x: numpy.array((0.0,))
            return
        else:
            self.attenuation = 1

        try:
            self.kde = scipy.stats.gaussian_kde(training_data)
        except scipy.linalg.LinAlgError as e:
            print(e)
            print(training_data)
            raise

    def __call__(self, lower, upper):
        if self.attenuation:
            return self.kde.integrate_box(lower, upper)
        else:
            return 0


def uuidtopos_to_numclose(uuid_to_pos):
    uuid_to_numclose = {}
    for uuid_, pos in uuid_to_pos.items():
        distances = [
            numpy.linalg.norm(pos - n_pos)
            for n_uuid, n_pos in uuid_to_pos.items()
            if n_uuid != uuid_
        ]
        num_close = sum(
            1 / ((m / _MAGIC_CLOSE_DISTANCE) + 1)
            for m in distances
        )
        uuid_to_numclose[uuid_] = num_close
    return uuid_to_numclose


def biggest_contour(mask):
    _, contours, _ = cv2.findContours(
        mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_index = None
    for i, c in enumerate(contours):
        if c is not None and len(c) > 5:
            area = cv2.contourArea(c)
            if max_index is None or area > max_area:
                max_area = area
                max_index = i

    # TODO: actually get the biggest
    return contours[max_index]


class EllipseSpace():

    def __init__(self, ellipse):
        self.ellipse = ellipse

#         center, up, right, umag, rmag = ellipse_center_up_right(ellipse)

#         self.center, self.up, self.right = (
#             numpy.array(x) for x in (center, up, right))

#         self.mag = numpy.array((rmag, umag))
#         self.inv_mag = 1 / self.mag

    def to_space(self, pos):
        return to_ellipse_space(self.ellipse, pos)
        # pos = numpy.array(pos)
        # pos -= self.center
        # pos = numpy.array(
        #     numpy.dot(pos, self.right),
        #     numpy.dot(pos, self.up),
        # )
        # return pos * self.inv_mag

    def from_space(self, pos):
        return from_ellipse_space(self.ellipse, pos)
        # pos = numpy.array(pos)
        # return (self.right * pos[0] * self.mag[0]
        #     + self.up * pos[1] * self.mag[1]
        #     + self.center)


def from_ellipse_space(ellipse, pos):
    center, up, right, umag, rmag = ellipse_center_up_right(ellipse)

    p =  (
        int(right[0] * pos[0] * rmag + up[0] * pos[1] * umag + center[0]),
        int(right[1] * pos[0] * rmag + up[1] * pos[1] * umag + center[1]),
    )

    return numpy.array(p)


def to_ellipse_space(ellipse, pos):
    center, up, right, umag, rmag = ellipse_center_up_right(ellipse)

    pos = (
        pos[0] - center[0],
        pos[1] - center[1],
    )

    pos = (
        pos[0] * right[0] + pos[1] * right[1],
        pos[0] * up[0] + pos[1] * up[1],
    )

    return numpy.array((
        pos[0] / rmag,
        pos[1] / umag,
    ))


def ellipse_center_up_right(ellipse):
    center = ellipse[0]
    center = mel.lib.moleimaging.point_to_int_point(center)
    angle_degs = ellipse[2]

    # TODO: do this properly
    if angle_degs > 90:
        angle_degs -= 180

    up = (0, -1)
    up = mel.lib.moleimaging.rotate_point_around_pivot(
        up, (0, 0), angle_degs)

    right = (1, 0)
    right = mel.lib.moleimaging.rotate_point_around_pivot(
        right, (0, 0), angle_degs)

    umag = ellipse[1][1] / 2
    rmag = ellipse[1][0] / 2

    return center, up, right, umag, rmag


