"""Identify moles."""

import collections
import functools

import numpy

import mel.jupyter.utils


# Individual probabilities less than this are out of consideration.
_MAGIC_P_THRESHOLD = 0.00001


def p_to_cost(p):
    # return int(10 / p) - 9
    return int(1 / p)


class Guesser():

    def __init__(self, uuid_to_pos, cold_classifier, warm_classifier):
        self.uuid_to_pos = uuid_to_pos
        self.cold_classifier = cold_classifier
        self.warm_classifier = warm_classifier
        self.init_a_to_bc()

        @functools.lru_cache(maxsize=1024)
        def warm(ref_uuid, ref_a, a):
            ref_pos = self.uuid_to_pos[ref_a]
            pos = self.uuid_to_pos[a]
            return tuple(
                (b, p_to_cost(p * q))
                for b, p, q in self.warm_classifier(ref_uuid, ref_pos, pos)
                if _MAGIC_P_THRESHOLD < p * q  #and not numpy.isnan(p * q)
                # if not numpy.isclose(0, p * q) and not numpy.isnan(p * q)
            )

        self.warm = warm

    def init_a_to_bc(self):
        uuid_to_numclose = mel.jupyter.utils.uuidtopos_to_numclose(
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
        return {a: None for a in self.a_to_bc}

    def yield_next_states(self, _est_cost, total_cost, state):
        filled = {a: b for a, b in state.items() if b is not None}
        already_taken = set(filled.values())

        if not filled:
            yield from self.yield_next_states_cold(
                _est_cost, total_cost, state)
            return

        # TODO: pick a non-arbitrary 'reference a', perhaps the closest to the
        # target?
        ref_a, ref_uuid = next(iter(filled.items()))
        total_est, a_to_est = self.estimates(
            state, already_taken, ref_a, ref_uuid)

        for a, b in state.items():
            if b is not None:
                continue

            pos = self.uuid_to_pos[a]

            est_without_a = total_est // a_to_est[a]

            num_added = 0
            for b, cost in self.warm(ref_uuid, ref_a, a):
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

    def estimates(self, state, already_taken, ref_a, ref_uuid):

        ref_pos = self.uuid_to_pos[ref_a]

        total_est = 1
        a_to_est = {}

        for a, b in state.items():
            if b is not None:
                continue

            best_cost = None
            for b, cost in self.warm(ref_uuid, ref_a, a):
                if b not in already_taken:
                    if best_cost is None or cost < best_cost:
                        best_cost = cost

            if best_cost is None:
                a_to_est[a] = 1
            else:
                a_to_est[a] = best_cost
                total_est *= best_cost

        return total_est, a_to_est

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
    def yield_next_states_cold(self, _est_cost, total_cost, state):
        for a, b in state.items():
            if b is not None:
                raise Exception('b must be None')
            new_state = dict(state)
            new_state[a] = a
            yield 1, 1, new_state
