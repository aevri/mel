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

MAX_MOLE_COST = 10 ** 4


def p_to_cost(p):
    return int(10 / p) - 9
    # return int(1 / p)


# Note that there are two uses for uuids below, for 'location' and for
# 'identity'.
#
# We are trying to contruct a mapping of 'supplied uuid' to
# 'guessed uuid'.
#
# We use the term 'location' to mean a 'supplied uuid', as we will use it
# mainly to map to an (x, y) co-ordinate. At the end we'll use it in the
# supplied mapping of old->new as 'old', or supplied->guessed as 'supplied'.
#
# We use the term 'identity' to mean the uuid of a mole that has previously
# been marked as 'canonical'. As in, "The name of a real mole".
#
# So we can restate our goal as trying to construct a mapping of
# location->identity, where we're guessing at identity. Sometimes the identity
# is already known, which makes the job a lot easier.
#
# We use 'loc' as an abbreviation for 'location'.
# We use 'ident' as an abbreviation for 'identity'.


class UuidToIndexTranslator():

    def __init__(self):
        self._index_to_uuid = []
        self._uuid_to_index = {}

    def add_uuids(self, uuids):
        # We sort the uuids so that we number them in a deterministic fashion,
        # this means that multiple runs of the program should retain the same
        # meaning of the indices.
        for u in sorted(uuids):
            if u not in self._uuid_to_index:
                self._uuid_to_index[u] = len(self._index_to_uuid)
                self._index_to_uuid.append(u)

    def num_uuids(self):
        return len(self._index_to_uuid)

    def uuid_dict_to_index_tuple(self, dict_in, tuple_len):
        if len(dict_in) < tuple_len:
            raise ValueError('Not enough entries in dict_in')
        values_out = [None] * tuple_len
        for uuid_, value in dict_in.items():
            values_out[self._uuid_to_index[uuid_]] = value
        return tuple(values_out)

    def index(self, uuid_):
        return self._uuid_to_index[uuid_]

    def uuid_(self, index):
        return self._index_to_uuid[index]


def make_calc_guesses(positions, uuid_index_translator, pos_classifier):

    def calc_guesses(predictor_loc_ident, guess_location):
        ref_pos = positions[predictor_loc_ident[0]]
        pos = positions[guess_location]
        predictor_ident_uuid = uuid_index_translator.uuid_(
            predictor_loc_ident[1])
        guesses = (
            (guess_ident_uuid, p_to_cost(p * q))
            for guess_ident_uuid, p, q in pos_classifier(
                predictor_ident_uuid, ref_pos, pos)
            if _MAGIC_P_THRESHOLD < p * q

            # These options were unacceptably slow (I seem to remember):
            # if _MAGIC_P_THRESHOLD < p * q  and not numpy.isnan(p * q)
            # if not numpy.isclose(0, p * q) and not numpy.isnan(p * q)
        )
        guesses = (
            (uuid_index_translator.index(guess_ident_uuid), cost)
            for guess_ident_uuid, cost in guesses
            if cost < MAX_MOLE_COST
        )
        return tuple(sorted(guesses, key=lambda x: x[1]))

    return calc_guesses


def predictors(positions, num_canonicals):
    """Return a tuple of (sqdist, predictor_location), ordered by index.

    Ensure that all locations are transitively connected to eachother by a
    prediction link. This seems to result in better overall identification
    performance and correctness by increasing the number of constraints.

    Try to keep the distance of predictive links low, as accuracy does seem to
    dimish with distance.

    If care were not taken to connect all the locations, 'islands' could form.
    These 'islands' would be identified completely independently of the others.
    This would mean in the case where a canonical mole is present, it would
    provide no benefit to the prediction of the islands it's not a part of.

    There is a trade-off to be had with connecting islands that are far apart
    from each-other, as the prediction accuracy does decrease with distance.
    Anecdotally it seems to be most important to connect everything together.

    """

    @functools.lru_cache(maxsize=128)
    def closest_sqdist_locs(location):
        ref_pos = positions[location]
        sqdist_location_list = sorted(
            (mel.lib.math.distance_sq_2d(other_pos, ref_pos), other_loc)
            for other_loc, other_pos in enumerate(positions)
            if other_loc != location
        )
        return sqdist_location_list

    guess_to_predictor = [None] * len(positions)

    # Canonicals can just predict themselves, it's most important to connect
    # things to them.
    for loc in range(num_canonicals):
        guess_to_predictor[loc] = (0, loc)

    if len(positions) == num_canonicals:
        return guess_to_predictor

    remaining_loc_set = set(range(num_canonicals, len(positions)))

    initial_loc = num_canonicals

    remaining_loc_set.remove(initial_loc)
    guess_to_predictor[initial_loc] = take_first(
        closest_sqdist_locs(initial_loc))

    while remaining_loc_set:
        sqdist, loc_a, loc_b = min(
            (sqdist, loc_a, loc_b)
            for loc_a in remaining_loc_set
            for sqdist, loc_b in closest_sqdist_locs(loc_a)
            if guess_to_predictor[loc_b] is not None
        )
        guess_to_predictor[loc_a] = (sqdist, loc_b)
        remaining_loc_set.remove(loc_a)

    return tuple(guess_to_predictor)


class PosGuesser():

    def __init__(
            self,
            num_locations,
            best_sqdist_uuid,
            bounder,
            num_canonicals,
            num_identities):

        self.num_locations = num_locations

        self.best_sqdist_uuid = best_sqdist_uuid
        self.bounder = bounder

        self.num_canonicals = num_canonicals
        self.possible_uuid_set = frozenset(range(num_identities))

    def initial_state(self):
        num_to_guess = self.num_locations - self.num_canonicals
        state = list(range(self.num_canonicals)) + [None] * num_to_guess
        return (1, num_to_guess), tuple(state)

    def yield_next_states(self, total_cost, state):

        decided = {
            loc: ident for loc, ident in enumerate(state) if ident is not None
        }
        already_taken = set(decided.values())
        num_remaining = len(state) - len(already_taken)

        loc = self._next_loc(state, decided)

        for ident in self.possible_uuid_set - already_taken:
            new_state = list(state)
            new_state[loc] = ident
            lower_bound = self.bounder.lower_bound(new_state)
            if lower_bound < total_cost[0]:
                raise Exception('lower_bound lower than previous cost')
            yield (lower_bound, num_remaining - 1), tuple(new_state)

    def _next_loc(self, state, decided):
        remaining_positions = set(range(len(state))) - set(decided.keys())

        sqdist_posuuid_remaineruuid = [
            (*self.best_sqdist_uuid[pos_uuid], pos_uuid)
            for pos_uuid in remaining_positions
        ]
        sqdist_posuuid_remaineruuid.sort()

        canonical_loc = take_first_or_none(
            pos_uuid
            for sqdist, uuid_for_pos, pos_uuid in sqdist_posuuid_remaineruuid
            if uuid_for_pos < self.num_canonicals
        )

        decided_loc = take_first_or_none(
            pos_uuid
            for sqdist, uuid_for_pos, pos_uuid in sqdist_posuuid_remaineruuid
            if uuid_for_pos in decided.keys()
        )

        remaining_loc = take_first(
            pos_uuid
            for sqdist, uuid_for_pos, pos_uuid in sqdist_posuuid_remaineruuid
        )

        if canonical_loc is not None:
            loc = canonical_loc
        elif decided_loc is not None:
            loc = decided_loc
        else:
            loc = remaining_loc

        return loc


class Bounder():

    def __init__(
            self,
            location_to_predictor,
            calc_guesses,
            num_identities,
            num_canonicals):

        @functools.lru_cache(maxsize=8192)
        def calc_guesses_cached(predictor_loc_ident, guess_location):
            return calc_guesses(predictor_loc_ident, guess_location)

        self.pos_guess = calc_guesses_cached
        self.location_to_predictor = location_to_predictor
        self.possible_uuid_set = frozenset(range(num_identities))
        self.num_canonicals = num_canonicals

    def pos_guess_dict(self, predictor_loc_ident, uuid_to_guess):
        return dict(self.pos_guess(predictor_loc_ident, uuid_to_guess))

    def lower_bound(self, state):
        already_taken = frozenset(
            ident for ident in state if ident is not None)

        lb = 1

        for guess_loc, guess_ident in enumerate(state):
            if guess_loc < self.num_canonicals:
                continue
            guess = (guess_loc, guess_ident)

            predictor_loc = self.location_to_predictor[guess_loc]
            predictor_ident = state[predictor_loc]
            predictor = (predictor_loc, predictor_ident)

            if guess_ident is not None:
                if predictor_ident is not None:
                    lb *= self.cost_for_guess(predictor, guess)
                else:
                    lb *= self.lower_bound_unk_predictor(
                        already_taken, predictor_loc, guess)
            else:
                if predictor_ident is not None:
                    lb *= self.lower_bound_unk_guess(
                        already_taken, predictor, guess_loc)
                else:
                    lb *= self.lower_bound_unk_unk(
                        already_taken, predictor_loc, guess_loc)

        return lb

    def cost_for_guess(self, predictor, guess):
        guess_loc, guess_ident = guess
        results = self.pos_guess_dict(predictor, guess_loc)
        return results.get(guess_ident, MAX_MOLE_COST)

    def lower_bound_unk_predictor(self, already_taken, predictor_loc, guess):
        possible_predictor_idents = self.possible_uuid_set - already_taken
        costs = (
            self.cost_for_guess((predictor_loc, predictor_ident), guess)
            for predictor_ident in possible_predictor_idents
        )
        return min(costs, default=MAX_MOLE_COST)

    def lower_bound_unk_guess(self, already_taken, predictor, guess_loc):
        results = self.pos_guess(predictor, guess_loc)
        valid_costs = (
            cost for uuid_, cost in results if uuid_ not in already_taken)
        return take_first_or_default(valid_costs, MAX_MOLE_COST)

    def lower_bound_unk_unk(self, already_taken, predictor_loc, guess_loc):
        possible_predictor_idents = self.possible_uuid_set - already_taken
        costs = (
            self.lower_bound_unk_guess(
                already_taken, (predictor_loc, predictor_ident), guess_loc)
            for predictor_ident in possible_predictor_idents
        )
        return min(costs, default=MAX_MOLE_COST)


def take_first_or_default(iterable, default):
    for item in iterable:
        return item
    return default


def take_first_or_none(iterable):
    return take_first_or_default(iterable, None)


def take_first(iterable):
    for item in iterable:
        return item
    raise IndexError('No first item to take')


class DuplicateDetector():

    def __init__(self):
        self.seen = set()

    def has_seen(self, *args):
        return hash(args) in self.seen

    def see(self, *args):
        self.seen.add(hash(args))


def print_state(count, total_cost, state):

    def ident_to_str(ident):
        if ident is None:
            return '__'
        else:
            return f'{ident:>2}'

    s = ' '.join(ident_to_str(x) for x in state)
    print(f'{count:>6} ({s}) {total_cost}')


def best_match_combination(guesser, *, max_iterations=10**5):

    state_q = mel.lib.priorityq.PriorityQueue()
    initial_cost, initial_state = guesser.initial_state()
    state_q.push(initial_cost, initial_state)

    seen = DuplicateDetector()

    best_cost = initial_cost
    best_state = initial_state
    deepest = 0
    count = 0
    while count != max_iterations:

        # Advance best state.
        total_cost, state = state_q.pop()

        count += 1
        should_report = 0 == count % 1000
        depth = sum(1 for x in state if x is not None)
        if depth > deepest:
            deepest = depth
            best_cost = total_cost
            best_state = state
            should_report = True
        if should_report:
            print_state(count, total_cost, state)

        # See if we're done.
        if all(x is not None for x in state):
            print_state(count, total_cost, state)
            return total_cost, state

        # Nope, advance states.
        for new_cost, new_state in guesser.yield_next_states(
                total_cost, state):

            if not seen.has_seen(new_cost, new_state):
                state_q.push(new_cost, new_state)
                seen.see(new_cost, new_state)

        if not state_q:
            # This is the last option, return the best.
            print('Final option')
            print_state(count, total_cost, state)
            # TODO: mark new moles or update contract to say some can be None
            return best_cost, best_state

    (total_cost, _), state = state_q.pop()
    return best_cost, best_state


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

        # pylint: disable=invalid-unary-operand-type
        self.upper = -self.lower
        # pylint: enable=invalid-unary-operand-type

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

        if known_uuid not in self.molepos_kernels:
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
# -----------------------------------------------------------------------------
# Copyright (C) 2017 Angelos Evripiotis.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------ END-OF-FILE ----------------------------------
