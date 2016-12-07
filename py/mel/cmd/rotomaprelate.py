"""Guess the relationships between moles in a rotomap."""


import json

import mel.lib.math


def setup_parser(parser):
    parser.add_argument(
        'FROM',
        type=str,
        help="Path of the 'from' rotomap json file.")
    parser.add_argument(
        'TO',
        type=str,
        nargs='+',
        help="Paths of the 'to' rotomap json files.")
    parser.add_argument(
        '--match-cutoff-distance',
        type=int,
        default=None,
        help="The maximum distance to allow between the predicted location of "
        "a mole in the 'to' image, and the location of a candidate "
        "match.")
    parser.add_argument(
        '--offset-cutoff-distance',
        type=int,
        default=None,
        help="The maximum distance to consider for translation theories "
        "between maps")


def process_args(args):

    files = [args.FROM]
    files.extend(args.TO)

    for from_path, to_path in pairwise(files):
        process_files(from_path, to_path, args)


def pairwise(iterable):
    return zip(iterable, iterable[1:])


def process_files(from_path, to_path, args):

    from_moles = load_json(from_path)
    to_moles = load_json(to_path)

    pairs = relate(
        from_moles,
        to_moles,
        args.match_cutoff_distance,
        args.offset_cutoff_distance)

    if pairs is None:
        return

    for mole in to_moles:
        for p in pairs:
            if p[0] and p[1]:
                if mole['uuid'] == p[1]:
                    mole['uuid'] = p[0]
                    break

    with open(to_path, 'w') as f:
        json.dump(
            to_moles,
            f,
            indent=4,
            separators=(',', ': '),
            sort_keys=True)

        # There's no newline after dump(), add one here for happier viewing
        print(file=f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def relate(from_moles, to_moles, cutoff, offset_cutoff):
    if not from_moles:
        raise ValueError('from_moles is empty')
    if not to_moles:
        raise ValueError('to_moles is empty')

    offset_cutoff_sq = None
    if offset_cutoff:
        offset_cutoff_sq = offset_cutoff ** 2

    if cutoff is not None:
        cutoff_sq = cutoff * cutoff
    else:
        cutoff_sq = mole_min_sq_distance(to_moles)
        if cutoff_sq is None:
            cutoff_sq = 0

    best_theory = None
    best_theory_dist_sq = None
    best_theory_offset_dist_sq = None
    for source in from_moles:
        for dest in to_moles:
            to_x = dest['x'] - source['x']
            to_y = dest['y'] - source['y']
            offset_dist_sq = to_x * to_x + to_y * to_y

            if offset_cutoff_sq is not None:
                if offset_dist_sq >= offset_cutoff_sq:
                    continue

            theory, dist_sq = make_offset_theory(
                from_moles, to_moles, (to_x, to_y), cutoff_sq)

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

    for i, a in enumerate(from_moles):
        point = mole_to_point(a)
        point = (point[0] + offset[0], point[1] + offset[1])
        best_index, best_dist_sq = nearest_mole_index_to_point(point, to_moles)
        if best_index is not None and best_dist_sq <= cutoff_sq:
            r_point = mole_to_point(to_moles[best_index])
            r_point = (r_point[0] - offset[0], r_point[1] - offset[1])
            r_index, _ = nearest_mole_index_to_point(r_point, from_moles)
            if i == r_index:
                theory.append((a['uuid'], to_moles[best_index]['uuid']))
                del to_moles[best_index]
                dist_sq_sum += best_dist_sq
            else:
                theory.append((a['uuid'], None))
        else:
            theory.append((a['uuid'], None))

    for b in to_moles:
        theory.append((None, b['uuid']))

    return theory, dist_sq_sum


def nearest_mole_index_to_point(point, mole_list):
    best_index = None
    best_dist_sq = None
    for i, mole in enumerate(mole_list):
        dist_sq = mel.lib.math.distance_sq_2d(
            point,
            mole_to_point(mole))
        if best_index is None or dist_sq < best_dist_sq:
            best_index = i
            best_dist_sq = dist_sq
    return best_index, best_dist_sq


def mole_distance_sq(from_mole, to_mole):
    return mel.lib.math.distance_sq_2d(
        mole_to_point(from_mole),
        mole_to_point(to_mole))


def mole_to_point(mole):
    return (mole['x'], mole['y'])
