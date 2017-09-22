"""Handy stuff for working in Jupyter notebooks."""

import collections
import heapq

import cv2
import numpy
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


class AttentuatedKde():

    def __init__(self, kde_factory, training_data):
        self.kde = kde_factory(training_data)
        self.len = len(training_data)
        if not self.len:
            self.attenuation = 0.0
        else:
            self.attenuation = 1 - (1 / (1 + (self.len + 4) / 5))

    def __call__(self, x):
        return self.kde(x) * self.attenuation


class StatePriorityQueue():

    def __init__(self):
        self.heap = []
        self.next_tie_breaker = 0
        self.max_filled = 0

    def push(self, estimate, value, state):

        heapq.heappush(
            self.heap,
            (1 - estimate, 1 - value, self.next_tie_breaker, state))

        self.max_filled = max(self._count_filled(state), self.max_filled)
        self.next_tie_breaker += 1

    def pop(self):
        estimate, value, _, state = heapq.heappop(self.heap)
        return 1 - estimate, 1 - value, state

    def _count_filled(self, state):
        return sum(1 for a, b in state.items() if b is not None)

    def max(self):
        filled_counts = collections.Counter()
        max_filled = 0
        max_state = None
        for e, v, _, state in self.heap:
            filled = sum(1 for a, b in state.items() if b is not None)
            max_filled = max(max_filled, filled)
            max_state = state
        return max_state

    def __len__(self):
        return len(self.heap)

    def __str__(self):
        return ("<StatePriorityQueue:: len:{}, max:{}>".format(
            len(self.heap), self.max_filled))

        filled_counts = collections.Counter()
        total_value = 0
        max_value = 0
        max_filled = 0
        for e, v, _, state in self.heap:
            value = 1 - v
            total_value += value
            max_value = max(value, max_value)
            filled = sum(1 for a, b in state.items() if b is not None)
            filled_counts[filled] += 1
            max_filled = max(max_filled, filled)
        mean_value = total_value / len(self.heap)

        return ("<StatePriorityQueue:: "
                "len:{}, max:{}, fill:{}, mean_v:{:.3f}, max_v:{:.3f}>".format(
                len(self.heap),
                max_filled,
                filled_counts.most_common(5),
                mean_value,
                max_value))


def best_match_combination(a_b_p_list):
    # TODO: don't forget new mole cases
    a_to_bp = collections.defaultdict(dict)
    for a, b, p in a_b_p_list:
        if p > 1:
            raise ValueError(
                f"'p' must be equal to or less than 1, "
                "got: a={a}, b={b}, p={p}")
        a_to_bp[a][b] = p

    state_q = StatePriorityQueue()
    state_q.push(1, 1, {a: None for a in a_to_bp})

    loop_count = 0
    while True:
        loop_count += 1
        if loop_count % 100 == 0:
            print("--", loop_count)
            print(state_q)

        if not state_q:
            # If we ran out of moles to match against, this must be a new mole.
            # This isn't the only way we can detect a new mole.
            raise NotImplementedError("Ran out of options!")

        # Advance best state.
        est_p, total_p, state = state_q.pop()

        if loop_count % 100 == 0:
            print('all', all(state.values()), tuple(state.values()))

        # See if we're done.
        if all(state.values()):
            return total_p, state

        if numpy.isclose(0, total_p):
            return state_q.max()
            raise NotImplementedError("Decide what to do when no options.")

        # Nope, advance states.

        already_taken = {b for a, b in state.items() if b is not None}

        best_estimates = {}
        new_est_p = total_p
        for a, b in state.items():
            if b is not None:
                continue
            best_p = None
            for b, p in a_to_bp[a].items():
                if b not in already_taken:
                    if best_p is None or p > best_p:
                        best_p = p
            if best_p is not None:
                best_estimates[a] = best_p
                new_est_p *= best_p

        if numpy.isclose(0, new_est_p):
            # If we ran out of moles to match against, this must be a new mole.
            # This isn't the only way we can detect a new mole.
            # raise NotImplementedError("New mole!")
            pass

        for a, b in state.items():
            if b is not None:
                continue

            num_added = 0
            base_est_p = new_est_p / best_estimates[a]
            for b, p in a_to_bp[a].items():
                if b not in already_taken:
                    new_p = total_p * p
                    if not numpy.isclose(0, new_p):
                        new_state = dict(state)
                        new_state[a] = b
                        state_q.push(base_est_p * p, new_p, new_state)
                        num_added += 1

            if not num_added:
                # If we ran out of moles to match against, this must be a new
                # mole. This isn't the only way we can detect a new mole.
                # raise NotImplementedError("New mole!")
                pass

        if not state_q:
            # This is the last and apparently best option.
            return total_p, state


class MoleClassifier():

    def __init__(self, uuid_to_frameposlist):
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


def uuidtopos_to_numclose(uuid_to_pos):
    uuid_to_numclose = {}
    for uuid_, pos in uuid_to_pos.items():
        distances = [
            numpy.linalg.norm(pos - n_pos)
            for n_uuid, n_pos in uuid_to_pos.items()
            if n_uuid != uuid_
        ]
        if distances:
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
