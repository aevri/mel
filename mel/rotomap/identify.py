"""Identify moles."""

import collections
import functools

import cv2
import numpy

import mel.lib.ellipsespace
import mel.lib.kde
import mel.lib.moleimaging
import mel.lib.priorityq


# Individual probabilities less than this are out of consideration.
_MAGIC_P_THRESHOLD = 0.00001

_MAGIC_CLOSE_DISTANCE = 0.1


def p_to_cost(p):
    # return int(10 / p) - 9
    return int(1 / p)


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

    def yield_next_states(self, total_cost, state):
        filled = {a: b for a, b in state.items() if b is not None}

        if not filled:
            yield from self.yield_next_states_cold(
                total_cost, state)
            return

        already_taken = set(filled.values())

        for a, b in state.items():
            if b is not None:
                continue

            uuid_for_pos = next(
                uuid_
                for uuid_ in self.closest_uuids(a)
                if state[uuid_] is not None)
            uuid_for_history = state[uuid_for_pos]

            num_added = 0
            for b, cost in self.warm(uuid_for_history, uuid_for_pos, a):
                if b not in already_taken:
                    new_cost = total_cost * cost
                    new_state = dict(state)
                    new_state[a] = b

                    yield new_cost, new_state
                    num_added += 1

            # if not num_added:
            #     # If we ran out of moles to match against, this must be a new
            #     # mole. This isn't the only way we can detect a new mole.
            #     # raise NotImplementedError("New mole!")
            #     new_state = dict(state)
            #     new_state[a] = 'NewMole'
            #     yield total_cost, new_state


    def yield_next_states_cold(self, total_cost, state):
        for a, b in state.items():
            if b is not None:
                raise Exception('b must be None')

            num_added = 0
            for b, cost in self.a_to_bc[a].items():
                new_cost = total_cost * cost
                new_state = dict(state)
                new_state[a] = b
                yield new_cost, new_state
                num_added += 1

            if not num_added:
                # If we ran out of moles to match against, this must be a new
                # mole. This isn't the only way we can detect a new mole.
                # raise NotImplementedError("New mole!")
                new_state = dict(state)
                new_state[a] = 'NewMole'
                yield cost, new_state


def count_nonevals(d):
    return sum(1 for x in d.values() if x is None)


def make_cost_state(cost, state):
    return (cost, count_nonevals(state)), state


class DuplicateDetector():

    def __init__(self):
        self.seen = set()

    def has_seen(self, *args):
        return hash(args) in self.seen

    def see(self, *args):
        self.seen.add(hash(args))


def best_match_combination(guesser, *, max_iterations=10**5):

    state_q = mel.lib.priorityq.PriorityQueue()
    state_q.push(*make_cost_state(1, guesser.initial_state()))

    seen = DuplicateDetector()

    deepest = 0
    most_correct = 0
    count = 0
    while count != max_iterations:

        # Advance best state.
        (total_cost, _), state = state_q.pop()

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
                total_cost,
                depth,
                correct
            )

        # See if we're done.
        if all(state.values()):
            print(
                count,
                total_cost,
                depth,
                correct
            )
            return total_cost, state

        # Nope, advance states.
        for new_cost, new_state in guesser.yield_next_states(
                total_cost, state):

            hashable_state = tuple(sorted(new_state.items()))
            if not seen.has_seen(new_cost, hashable_state):
                state_q.push(*make_cost_state(new_cost, new_state))
                seen.see(new_cost, hashable_state)

        if not state_q:
            # This is the last and apparently best option.
            print('Final option')
            print(
                count,
                total_cost,
                depth,
                correct
            )
            # TODO: mark new moles or update contract to say some can be None
            return total_cost, state

    (total_cost, _), state = state_q.pop()
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

        frames = collections.defaultdict(dict)
        for uuid_, frameposlist in uuid_to_frameposlist.items():
            for frame, pos in frameposlist:
                frames[frame][uuid_] = pos

        uuid_to_neighbourlist = collections.defaultdict(list)
        for uuid_to_pos in frames.values():
            for uuid_, num_close in uuidtopos_to_numclose(uuid_to_pos).items():
                uuid_to_neighbourlist[uuid_].append(num_close)

        self.uuid_to_neighbourlist = uuid_to_neighbourlist

        self.kernels = tuple(
            mel.lib.kde.Kde(
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
            mel.lib.kde.Kde(
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
    contour = mel.lib.moleimaging.biggest_contour(mask)
    ellipse = cv2.fitEllipse(contour)
    elspace = mel.lib.ellipsespace.Transform(ellipse)

    uuid_to_pos = {
        uuid_: elspace.to_space(pos)
        for uuid_, pos in frame.moledata.uuid_points.items()
    }

    return uuid_to_pos


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
