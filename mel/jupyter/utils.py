"""Handy stuff for working in Jupyter notebooks."""

import collections
import heapq
import math

import cv2
import numpy
import scipy.linalg
import scipy.stats

import mel.lib.moleimaging


_MAGIC_CLOSE_DISTANCE = 0.1


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


def frames_to_uuid_frameposlist(frame_iterable):
    uuid_to_frameposlist = collections.defaultdict(list)

    for frame in frame_iterable:
        mask = frame.load_mask()
        contour = biggest_contour(mask)
        ellipse = cv2.fitEllipse(contour)
        elspace = EllipseSpace(ellipse)
        for uuid, pos in frame.moledata.uuid_points.items():
            uuid_to_frameposlist[uuid].append(
                (str(frame), elspace.to_space(pos)))

    return uuid_to_frameposlist


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


class AttentuatedKde():

    def __init__(self, kde_factory, training_data):
        self.len = training_data.shape[-1]
        if self.len < 3:
            self.attenuation = 0.0
            self.kde = lambda x: numpy.array((0.0,))
            return
        else:
            self.attenuation = 1 - (1 / (1 + (self.len + 4) / 5))

        try:
            self.kde = kde_factory(training_data)
        except scipy.linalg.LinAlgError as e:
            print(e)
            print(training_data)
            raise
            self.attenuation = 0.0
            self.kde = lambda x: numpy.array((0.0,))

    def __call__(self, x):
        return self.kde(x) * self.attenuation


class StatePriorityQueue():

    def __init__(self):
        self.heap = []
        self.next_tie_breaker = 0
        self.already_tried = set()

    def push(self, estimate, value, state):

        normalised = tuple(sorted(state.items()))
        if normalised in self.already_tried:
            return
        self.already_tried.add(normalised)

        heapq.heappush(
            self.heap,
            (estimate, value, self.next_tie_breaker, state))

        self.next_tie_breaker += 1

    def pop(self):
        estimate, value, _, state = heapq.heappop(self.heap)
        return estimate, value, state

    def __len__(self):
        return len(self.heap)

    def __str__(self):
        return ("<StatePriorityQueue:: len:{}>".format(len(self.heap)))


class Guesser():

    def __init__(self, uuid_to_pos, classifier):
        self.uuid_to_pos = uuid_to_pos
        self.classifier = classifier
        self.init_abc()

    def init_abc(self):
        neighbour_guesses = self.classifier.guesses_from_neighbours(
            self.uuid_to_pos)

        y_guesses = []
        for uuid_, pos in self.uuid_to_pos.items():
            y_guesses.extend(
                (uuid_, *args)
                for args in self.classifier.guesses_from_ypos(pos[1])
            )

        match_to_n = {
            (x[0], x[1]): (x[2] * x[3])
            for x in neighbour_guesses
        }

        match_to_y = {
            (x[0], x[1]): (x[2] * x[3])
            for x in y_guesses
        }

        matches = tuple(
            (*key, match_to_y[key] * value, match_to_y[key], value)
            for key, value in match_to_n.items()
        )

        a_b_p_list = [(a, b, p) for a, b, p, _, _ in matches]

        self.a_to_bc = collections.defaultdict(dict)
        for a, b, p in a_b_p_list:
            if p > 1:
                raise ValueError(
                    f"'p' must be equal to or less than 1, "
                    "got: a={a}, b={b}, p={p}")
            if not numpy.isclose(0, p):
                cost = int(1 / p)
                self.a_to_bc[a][b] = cost

    def print_correct_stats(self):
        correct = 1
        for a, b_to_c in self.a_to_bc.items():
            found = False
            if a in b_to_c:
                correct *= b_to_c[a]
            else:
                correct = 0

        print('cost of correct solution', correct)

    def print_space_stats(self):
        size_est = 1
        for a, bc in self.a_to_bc.items():
            size_est *= len(bc)

        print(f"best_match_combination:: est_size:{size_est:,}")

    def initial_state(self):
        return {a: None for a in self.a_to_bc}

    def yield_next_states(self, est_cost, total_cost, state):
        already_taken = {b for a, b in state.items() if b is not None}

        best_estimates, new_est_cost = self.calc_estimates(
            total_cost, state, already_taken)

        for a, b in state.items():
            if b is not None:
                continue

            num_added = 0
            base_est_cost = new_est_cost / best_estimates[a]
            for b, cost in self.a_to_bc[a].items():
                if b not in already_taken:
                    new_cost = total_cost * cost
                    new_est_cost = base_est_cost * cost
                    new_state = dict(state)
                    new_state[a] = b
                    yield new_est_cost, new_cost, new_state
                    num_added += 1

            if not num_added:
                # If we ran out of moles to match against, this must be a new
                # mole. This isn't the only way we can detect a new mole.
                # raise NotImplementedError("New mole!")
                new_state = dict(state)
                new_state[a] = 'NewMole'
                yield cost, cost, new_state

    def calc_estimates(self, total_cost, state, already_taken):
        best_estimates = {}
        new_est_cost = total_cost
        for a, b in state.items():
            if b is not None:
                continue
            best_cost = None
            for b, cost in self.a_to_bc[a].items():
                if b not in already_taken:
                    if best_cost is None or cost > best_cost:
                        best_cost = cost
            if best_cost is not None:
                best_estimates[a] = best_cost
                new_est_cost *= best_cost
            else:
                best_estimates[a] = 1

        return best_estimates, new_est_cost


def best_match_combination(guesser):

    # guesser.print_space_stats()
    # guesser.print_correct_stats()

    state_q = StatePriorityQueue()
    state_q.push(1, 1, guesser.initial_state())

    deepest = 0
    most_correct = 0
    count = 0
    while True:

        if not state_q:
            # If we ran out of moles to match against, this must be a new mole.
            # This isn't the only way we can detect a new mole.
            raise NotImplementedError("Ran out of options!")

        # Advance best state.
        est_cost, total_cost, state = state_q.pop()

        count += 1
        should_report = 0 == count % 1000
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
                est_cost,
                depth,
                correct
            )

        # See if we're done.
        if all(state.values()):
            return total_cost, state

        # Nope, advance states.
        for new_est_cost, new_cost, new_state in guesser.yield_next_states(
                est_cost, total_cost, state):
            state_q.push(new_est_cost, new_cost, new_state)

        if not state_q:
            # This is the last and apparently best option.
            return total_cost, state


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


class MoleClassifier():

    def __init__(self, uuid_to_frameposlist):
        self.uuid_to_frameposlist = uuid_to_frameposlist
        uuid_to_poslist = {
            uuid_: [pos for frame, pos in frameposlist]
            for uuid_, frameposlist in uuid_to_frameposlist.items()
        }
        self.uuids, self.poslistlist = zip(*uuid_to_poslist.items())
        yposlistlist = tuple(
            numpy.array(tuple(y for x, y in poslist))
            for poslist in self.poslistlist
        )
        self.ykernels = tuple(
            AttentuatedKde(scipy.stats.gaussian_kde, yposlist)
            for yposlist in yposlistlist
        )

        self.frames = collections.defaultdict(dict)
        for uuid_, frameposlist in uuid_to_frameposlist.items():
            for frame, pos in frameposlist:
                self.frames[frame][uuid_] = pos

        uuid_to_neighbourlist = collections.defaultdict(list)
        for uuid_to_pos in self.frames.values():
            for uuid_, num_close in uuidtopos_to_numclose(uuid_to_pos).items():
                uuid_to_neighbourlist[uuid_].append(num_close)

        self.uuid_to_neighbourlist = uuid_to_neighbourlist

        self.numclose_kernels = tuple(
            AttentuatedKde(
                scipy.stats.gaussian_kde,
                numpy.array( uuid_to_neighbourlist[uuid_]))
            for uuid_ in self.uuids
        )

    def guesses_from_ypos(self, ypos):

        if not numpy.isscalar(ypos):
            raise ValueError(f"'ypos' must be a scalar, got '{ypos}'.")

        densities = numpy.array(tuple(k(ypos)[0] for k in self.ykernels))
        total_density = numpy.sum(densities)

        if numpy.isclose(total_density, 0):
            total_density = -1

        matches = []
        for i, m_uuid in enumerate(self.uuids):
            p = densities[i]
            q = p / total_density
            matches.append((m_uuid, p, q))

        return matches

    def guesses_from_neighbours(self, uuid_pos):

        # Calculate neighbour values for all supplied moles
        uuid_to_numclose = uuidtopos_to_numclose(uuid_pos)

        # Calculate probability for all supplied moles vs. all self moles
        matches = []
        for uuid_, numclose in uuid_to_numclose.items():
            densities = numpy.array(tuple(
                k(numclose)[0] for k in self.numclose_kernels))
            total_density = numpy.sum(densities)

            if numpy.isclose(total_density, 0):
                total_density = -1

            # i = numpy.argmax(densities)
            # d = densities[i][0]
            for i, m_uuid in enumerate(self.uuids):
                p = densities[i]
                q = p / total_density
                matches.append((uuid_, m_uuid, p, q))

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


def yield_relative_results(classifier, frame_list):
    for frame in frame_list:
        uuid_to_pos = frame_to_uuid_to_pos(frame)
        for uuid_, pos in uuid_to_pos.items():
            for uuid2, pos2 in uuid_to_pos.items():
                if uuid_ == uuid2:
                    continue
                results = classifier(uuid_, pos, pos2)
                yield (uuid2, results)


def yield_cold_results(classifier, frame_list):
    for frame in frame_list:
        uuid_to_pos = frame_to_uuid_to_pos(frame)
        uuid_to_numclose = uuidtopos_to_numclose(uuid_to_pos)
        for uuid_, pos in uuid_to_pos.items():
            numclose = uuid_to_numclose[uuid_]
            results = classifier(pos[1], numclose)
            yield (uuid_, results)


def yield_ypos_neighbour_results(classifier, frame_list):
    for frame in frame_list:
        uuid_to_pos = frame_to_uuid_to_pos(frame)
        multi_results = classifier.guesses_from_neighbours(uuid_to_pos)
        uuid_to_guesses = collections.defaultdict(list)
        for u1, u2, p, q in multi_results:
            uuid_to_guesses[u1].append((u2, p, q))
        for uuid_, nresults in uuid_to_guesses.items():
            pos = uuid_to_pos[uuid_]
            yresults = classifier.guesses_from_ypos(pos[1])
            uuid_to_yresult = {
                uuid3: (p, q)
                for uuid3, p, q in yresults
            }
            results = [
                (uuid3, np * yp, nq * yq)
                for uuid3, np, nq in nresults
                for yp, yq in (uuid_to_yresult[uuid3],)
            ]
            yield (uuid_, results)


def yield_neighbour_results(classifier, frame_list):
    for frame in frame_list:
        uuid_to_pos = frame_to_uuid_to_pos(frame)
        multi_results = classifier.guesses_from_neighbours(uuid_to_pos)
        uuid_to_guesses = collections.defaultdict(list)
        for u1, u2, p, q in multi_results:
            uuid_to_guesses[u1].append((u2, p, q))
        for uuid_, results in uuid_to_guesses.items():
            yield (uuid_, results)


def yield_ypos_results(classifier, frame_list):
    for frame in frame_list:
        uuid_to_pos = frame_to_uuid_to_pos(frame)
        for uuid_, pos in uuid_to_pos.items():
            results = classifier.guesses_from_ypos(pos[1])
            yield (uuid_, results)


def count_matches(result_generator, pass_threshold):
    quartile_counts = collections.Counter()
    hits = 0
    misses = 0
    passes = 0

    for target_uuid, results in result_generator:
        matched = False
        for i, r in enumerate(reversed(sorted(results, key=lambda x: x[2]))):
            uuid2, p, q = r
            if p * q > pass_threshold:
                if uuid2 == target_uuid:
                    matched = True
                    quartile = int(4 * i / len(results))
                    quartile_counts[quartile] += 1
                    if i == 0:
                        hits += 1
                    else:
                        misses += 1
                    break
            else:
                passes += 1
                matched = True
                break
        if not matched:
            raise Exception(f'No match for {target_uuid}')

    return quartile_counts, hits, misses, passes


def print_matches(quartile_counts, hits, misses, passes):
    print(f'hits: {hits}, misses: {misses}, ', end='')
    print(f'pct: {100*hits/(hits+misses):.2f}%, passes: {passes}')

    total_match_index_counts = sum(quartile_counts.values())
    for index, count in sorted(quartile_counts.items()):
        print(index, count / total_match_index_counts)
