"""Handy stuff for working in Jupyter notebooks."""

import collections
import heapq
import math

import cv2
import numpy
import scipy.linalg
import scipy.stats

import mel.lib.ellipsespace
import mel.lib.moleimaging
import mel.rotomap.identify


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
