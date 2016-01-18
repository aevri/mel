"""Guess the relationships between moles in a rotomap."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import mel.lib.math


def setup_parser(parser):
    parser.add_argument(
        'FROM',
        type=str,
        help="Path to the 'from' rotomap json file.")
    parser.add_argument(
        'TO',
        type=str,
        help="Path to the 'to' rotomap json file.")
    parser.add_argument(
        '--match-cutoff-distance',
        type=int,
        default=None,
        help="The maximum distance to allow between the predicted location of "
        "a mole in the 'to' image, and the location of a candidate "
        "match.")


def process_args(args):
    from_moles = load_json(args.FROM)
    to_moles = load_json(args.TO)

    pairs = relate(from_moles, to_moles, args.match_cutoff_distance)
    for p in pairs:
        if p[0] and p[1]:
            print(p[0], p[1])


def load_json(path):
    with open(path) as f:
        return json.load(f)


def relate(from_moles, to_moles, cutoff):
    if not from_moles:
        raise ValueError('from_moles is empty')
    if not to_moles:
        raise ValueError('to_moles is empty')

    if cutoff is not None:
        cutoff_sq = cutoff * cutoff
    else:
        cutoff_sq = mole_min_sq_distance(to_moles)

    best_theory = None
    best_theory_dist_sq = None
    best_theory_offset_dist_sq = None
    for source in from_moles:
        for dest in to_moles:
            to_x = dest['x'] - source['x']
            to_y = dest['y'] - source['y']
            offset_dist_sq = to_x * to_x + to_y * to_y
            theory, dist_sq = make_offset_theory(
                from_moles, to_moles, (to_x, to_y), cutoff_sq)

            # print(len(theory), dist_sq, (to_x, to_y), cutoff_sq)

            new_best = best_theory is None
            if not new_best and len(theory) < len(best_theory):
                new_best = True
            if not new_best and len(theory) == len(best_theory):
                if dist_sq < best_theory_dist_sq:
                    new_best = True
                if not new_best and dist_sq == best_theory_dist_sq:
                    if offset_dist_sq < best_theory_offset_dist_sq:
                        new_best = True

            if new_best:
                best_theory = theory
                best_theory_dist_sq = dist_sq
                best_theory_offset_dist_sq = offset_dist_sq
                # print('*', len(best_theory), best_theory_dist_sq)

    return best_theory


def mole_min_sq_distance(moles):
    min_dist = None
    for i, a in enumerate(moles):
        for j, b in enumerate(moles):
            if j == i:
                continue
            dist = mole_distance_sq(a, b)
            if min_dist is None or dist < min_dist:
                min_dist = dist
    return min_dist


def make_offset_theory(from_moles, to_moles_in, offset, cutoff_sq):
    to_moles = list(to_moles_in)

    theory = []

    dist_sq_sum = 0

    for a in from_moles:
        point = mole_to_point(a)
        point = (point[0] + offset[0], point[1] + offset[1])
        best_index = None
        best_dist_sq = None
        for j, b in enumerate(to_moles):
            dist_sq = mel.lib.math.distance_sq_2d(
                point,
                mole_to_point(b))
            if dist_sq < cutoff_sq:
                if best_index is None or dist_sq < best_dist_sq:
                    best_index = j
                    best_dist_sq = dist_sq
        if best_index is not None:
            theory.append((a['uuid'], to_moles[best_index]['uuid']))
            del to_moles[best_index]
            dist_sq_sum += best_dist_sq
        else:
            theory.append((a['uuid'], None))

    for b in to_moles:
        theory.append((None, b['uuid']))

    return theory, dist_sq_sum


def mole_distance_sq(from_mole, to_mole):
    return mel.lib.math.distance_sq_2d(
        mole_to_point(from_mole),
        mole_to_point(to_mole))


def mole_to_point(mole):
    return (mole['x'], mole['y'])
