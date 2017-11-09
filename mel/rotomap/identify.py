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

MAX_MOLE_COST = 10 ** 5

def p_to_cost(p):
    # return int(10 / p) - 9
    return int(1 / p)


class PosGuesser():

    def __init__(
            self,
            uuid_to_pos,
            pos_classifier,
            canonical_uuid_set,
            possible_uuid_set):

        self.uuid_to_pos = uuid_to_pos
        self.pos_classifier = pos_classifier
        self.canonical_uuid_set = canonical_uuid_set
        self.possible_uuid_set = possible_uuid_set

        # Note that we want one cache per instance of Guesser. This means that
        # the values will be correct for the classifiers and positions
        # provided. Therefore we must create these dynamically.

        @functools.lru_cache(maxsize=1024)
        def pos_guess(uuid_for_history, uuid_for_position, uuid_to_guess):
            ref_pos = self.uuid_to_pos[uuid_for_position]
            pos = self.uuid_to_pos[uuid_to_guess]
            return tuple(
                (b, p_to_cost(p * q))
                for b, p, q in self.pos_classifier(
                    uuid_for_history, ref_pos, pos)
                if _MAGIC_P_THRESHOLD < p * q  #and not numpy.isnan(p * q)
                # if not numpy.isclose(0, p * q) and not numpy.isnan(p * q)
            )
        self.pos_guess = pos_guess

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

    def initial_state(self):
        return 1, {
            a: a if a in self.canonical_uuid_set else None
            for a in self.uuid_to_pos
        }

    def yield_next_states(self, total_cost, state):
        filled = {a: b for a, b in state.items() if b is not None}

        if not filled:
            raise Exception('Must have canonical moles')
            return

        already_taken = set(filled.values())

        # Generate additional states.

        for a, b in state.items():
            if b is not None:
                continue

            uuid_for_pos = next(
                uuid_
                for uuid_ in self.closest_uuids(a))

            for b in self.possible_uuid_set - already_taken:
                new_state = dict(state)
                new_state[a] = b
                lower_bound = self.lower_bound(new_state)
                yield lower_bound, new_state

    def lower_bound(self, state):
        filled = {a: b for a, b in state.items() if b is not None}
        already_taken = set(filled.values())

        lb = 1

        for a, b in state.items():
            if a in self.canonical_uuid_set:
                continue
            if b is not None:
                lb *= self.lower_bound_mole(state, already_taken, a, b)
            else:
                lb *= self.lower_bound_unk_mole(state, already_taken, a)

        return lb

    def lower_bound_mole(self, state, already_taken, a, b):

        uuid_for_position = next(
            uuid_
            for uuid_ in self.closest_uuids(a))

        uuid_for_history = state[uuid_for_position]
        if uuid_for_history is not None:
            return self.cost_for_guess(
                uuid_for_history, uuid_for_position, a, b)
        else:
            return self.lower_bound_guess(
                already_taken, uuid_for_position, a, b)

    def cost_for_guess(self, uuid_for_history, uuid_for_position, a, b):
        guesses = self.pos_guess(uuid_for_history, uuid_for_position, a)
        for g_b, g_cost in guesses:
            if b == g_b:
                return g_cost
        return MAX_MOLE_COST

    def lower_bound_guess(self, already_taken, uuid_for_position, a, b):
        cost_list = []
        for uuid_for_history in self.possible_uuid_set - already_taken:
            guesses = self.pos_guess(uuid_for_history, uuid_for_position, a)
            cost_list.append(min(
                (cost for g, cost in guesses if g == b),
                default=MAX_MOLE_COST
            ))
        return min(cost_list, default=MAX_MOLE_COST)

    def lower_bound_unk_mole(self, state, already_taken, a):
        uuid_for_position = next(
            uuid_
            for uuid_ in self.closest_uuids(a))
        uuid_for_history = state[uuid_for_position]

        cost_list = []
        if uuid_for_history is not None:
            for b in self.possible_uuid_set - already_taken:
                cost_list.append(
                    self.cost_for_guess(
                        uuid_for_history, uuid_for_position, a, b))
        else:
            for b in self.possible_uuid_set - already_taken:
                temp_taken = set(already_taken)
                temp_taken.add(b)
                cost_list.append(
                    self.lower_bound_guess(
                        temp_taken, uuid_for_position, a, b))
        return min(cost_list)


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
        num_to_fill = len(self.a_to_bc) - len(self.canonical_uuid_set)
        return (1, num_to_fill), {
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
        not_taken = set(state.keys()) - already_taken
        num_not_taken = len(not_taken) - 1

        for a in not_taken:

            uuid_for_pos = next(
                uuid_
                for uuid_ in self.closest_uuids(a)
                if state[uuid_] is not None)
            uuid_for_history = state[uuid_for_pos]

            num_added = 0
            for b, cost in self.warm(uuid_for_history, uuid_for_pos, a):
                if b not in already_taken:
                    new_cost = (total_cost[0] * cost, num_not_taken)
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
        num_not_taken = len(state)
        for a, b in state.items():
            if b is not None:
                raise Exception('b must be None')

            num_added = 0
            for b, cost in self.a_to_bc[a].items():
                new_cost = (total_cost[0] * cost, num_not_taken)
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
                yield total_cost, new_state


class DuplicateDetector():

    def __init__(self):
        self.seen = set()

    def has_seen(self, *args):
        return hash(args) in self.seen

    def see(self, *args):
        self.seen.add(hash(args))


class StateFormatter():

    def __init__(self, initial_state):
        unfilled_keys = set(
            key for key, value in initial_state.items() if value is None
        )
        self.key_order = tuple(sorted(unfilled_keys))
        self.value_names = {}
        self.name_count = 0

    def name(self, value):
        if value is None:
            return '__'
        name = self.value_names.get(value, None)
        if name is None:
            name = '{:>2}'.format(self.name_count)
            self.name_count += 1
            self.value_names[value] = name
        return name

    def __call__(self, count, total_cost, state):
        s = ' '.join(self.name(state[key]) for key in self.key_order)
        if isinstance(total_cost, tuple):
            return f'{count:>6} {total_cost} ({s})'
        else:
            return f'{count:>6} {total_cost:>12} ({s})'


def best_match_combination(guesser, *, max_iterations=10**5):

    state_q = mel.lib.priorityq.PriorityQueue()
    initial_cost, initial_state = guesser.initial_state()
    formatter = StateFormatter(initial_state)
    state_q.push(initial_cost, initial_state)

    seen = DuplicateDetector()

    best_cost = initial_cost
    best_state = initial_state
    deepest = 0
    most_correct = 0
    count = 0
    while count != max_iterations:

        # Advance best state.
        total_cost, state = state_q.pop()

        count += 1
        should_report = 0 == count % 10000
        # should_report = 0 == count % 1
        depth = sum(1 for a, b in state.items() if b is not None)
        correct = sum(1 for a, b in state.items() if a == b)
        if depth > deepest:
            deepest = depth
            should_report = True
        if correct > most_correct:
            most_correct = correct
            best_cost = total_cost
            best_state = state
            should_report = True
        if should_report:
            print(formatter(count, total_cost, state))

        # See if we're done.
        if all(state.values()):
            print(formatter(count, total_cost, state))
            return total_cost, state

        # Nope, advance states.
        for new_cost, new_state in guesser.yield_next_states(
                total_cost, state):

            hashable_state = tuple(sorted(new_state.items()))
            if not seen.has_seen(new_cost, hashable_state):
                state_q.push(new_cost, new_state)
                seen.see(new_cost, hashable_state)

        if not state_q:
            # This is the last option, return the best.
            print('Final option')
            print(formatter(count, total_cost, state))
            # TODO: mark new moles or update contract to say some can be None
            return best_cost, best_state

    (total_cost, _), state = state_q.pop()
    return best_cost, best_state
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
