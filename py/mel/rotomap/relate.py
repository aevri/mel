"""Relate rotomaps to eachother."""

import collections
import math

import cv2
import numpy

import mel.lib.math
import mel.rotomap.moles
import mel.rotomap.tricolour


def draw_canonical_mole(image, x, y, colour):
    radius = 16
    cv2.circle(image, (x, y), radius, colour, -1)


def draw_non_canonical_mole(image, x, y, colour):
    radius = 16
    top_left = (x - radius, y - radius)
    bottom_right = (x + radius, y + radius)
    cv2.rectangle(image, top_left, bottom_right, colour, -1)


def draw_mole(image, mole, colour):
    x = mole['x']
    y = mole['y']
    if mole['is_uuid_canonical']:
        draw_canonical_mole(image, x, y, colour)
    else:
        draw_non_canonical_mole(image, x, y, colour)


def draw_from_to_mole(image, from_mole, to_mole, colour):
    draw_mole(image, from_mole, colour)
    draw_mole(image, to_mole, colour)
    cv2.arrowedLine(
        image,
        tuple(mel.rotomap.moles.molepos_to_nparray(from_mole)),
        tuple(mel.rotomap.moles.molepos_to_nparray(to_mole)),
        (255, 255, 255),
        2,
        cv2.LINE_AA)


def draw_debug(image, to_moles, from_moles):
    # return draw_neighbourhoods2(image, to_moles, from_moles)

    if from_moles is None:
        from_moles = []

    from_dict = mole_list_to_uuid_dict(from_moles)
    to_dict = mole_list_to_uuid_dict(to_moles)
    from_set = set(from_dict.keys())
    to_set = set(to_dict.keys())

    from_only = from_set - to_set
    to_only = to_set - from_set
    in_both = from_set & to_set

    theory = []
    theory.extend((u, None) for u in from_only)
    theory.extend((None, u) for u in to_only)
    theory.extend((u, u) for u in in_both)

    if from_moles and to_moles:
        theory = best_offset_theory(from_moles, to_moles, None, None)

    image = numpy.zeros(image.shape)
    overlay_theory(image, theory, from_dict, to_dict)
    return image


def overlay_theory(image, theory, from_dict, to_dict):
    colour_removed = (0, 0, 255)
    colour_added = (0, 255, 0)
    colour_mapped = (255, 0, 0)
    for from_, to in theory:
        assert (from_ is not None) or (to is not None)
        if to is None:
            draw_mole(image, from_dict[from_], colour_removed)
        elif from_ is None:
            draw_mole(image, to_dict[to], colour_added)
        else:
            draw_from_to_mole(
                image,
                from_dict[from_],
                to_dict[to],
                colour_mapped)

    return image


def mole_list_to_uuid_dict(mole_list):
    return {m['uuid']: m for m in mole_list}


def overlay_unknown(image, from_only, to_only, from_dict, to_dict):
    colour_unknown_from = (255, 0, 255)
    colour_unknown_to = (255, 255, 0)
    for u in from_only:
        draw_mole(image, from_dict[u], colour_unknown_from)
    for u in to_only:
        draw_mole(image, to_dict[u], colour_unknown_to)


def draw_neighbourhoods2(image, to_moles, from_moles):

    if from_moles is None:
        from_moles = []

    from_dict = mole_list_to_uuid_dict(from_moles)
    to_dict = mole_list_to_uuid_dict(to_moles)
    from_set = set(from_dict.keys())
    to_set = set(to_dict.keys())

    from_only = from_set - to_set
    to_only = to_set - from_set
    in_both = from_set & to_set

    theory = []
    theory.extend((u, u) for u in in_both)

    advance_theory_by_neighbours(
        theory, from_only, to_only, from_dict, to_dict)

    image = numpy.zeros(image.shape)
    overlay_theory(image, theory, from_dict, to_dict)
    overlay_unknown(image, from_only, to_only, from_dict, to_dict)
    return image


def advance_theory_by_neighbours(
        theory, from_only, to_only, from_dict, to_dict):

    from_mn, from_nm = calc_neighbours(from_dict.values())
    to_mn, to_nm = calc_neighbours(to_dict.values())

    apply_theory_to_neighbours(theory, from_mn, to_mn)

    import pprint
    pprint.pprint(from_mn)
    pprint.pprint(from_nm)
    pprint.pprint(to_mn)
    pprint.pprint(to_nm)

    assertion = next_best_neighbour_assertion(
        from_mn, from_nm, to_mn, to_nm, from_dict, to_dict)

    if assertion is not None:
        theory.append(assertion)


def calc_neighbours(moles):
    mole_to_neighbours = {}
    neighbour_to_moles = collections.defaultdict(list)

    if len(moles) <= 1:
        return mole_to_neighbours, neighbour_to_moles

    for m in moles:
        x = m['x']
        y = m['y']

        nearest = mel.rotomap.moles.sorted_by_distances(moles, x, y)
        dist2 = mel.rotomap.moles.dist_xy(nearest[1], x, y) * 2
        neighbours = {
            n['uuid'] for n in nearest[1:]
            if mel.rotomap.moles.dist_xy(n, x, y) < dist2
        }
        mole_to_neighbours[m['uuid']] = {'locals': neighbours, 'status': None}
        for n_uuid in neighbours:
            neighbour_to_moles[n_uuid].append(m['uuid'])

    return mole_to_neighbours, neighbour_to_moles


def apply_theory_to_neighbours(theory, from_mn, to_mn):
    for from_, to in theory:
        assert (from_ is not None) or (to is not None)
        if to is None:
            raise NotImplementedError
        elif from_ is None:
            raise NotImplementedError
        else:
            from_mn[from_]['to'] = to
            from_mn[from_]['status'] = 'mapped'
            to_mn[to]['from'] = from_
            to_mn[to]['status'] = 'mapped'


def next_best_neighbour_assertion(
        from_mn, from_nm, to_mn, to_nm, from_dict, to_dict):

    assertions = []

    # Generate likely assertions about neighbours of mapped 'to' moles, look
    # for matches.

    for to_uuid, to_data in to_mn.items():
        if to_data['status'] == 'mapped':
            to_mole = to_dict[to_uuid]
            from_uuid = to_data['from']
            from_data = from_mn[from_uuid]
            from_mole = from_dict[from_uuid]

            to_angle_list = []
            for u in to_data['locals']:
                to_angle_list.append(
                    (angle_between_moles(to_mole, to_dict[u]),
                    u))

            from_angle_list = []
            for v in from_data['locals']:
                from_angle_list.append(
                    (angle_between_moles(from_mole, from_dict[v]),
                    v))

            # Try to match to a neighbour in the corresponding 'from' mole.
            # Check angle.

            print(from_angle_list)
            print(to_angle_list)

    # raise NotImplementedError


def angle_between_moles(start, end):
    start_xy = mel.rotomap.moles.molepos_to_nparray(start)
    end_xy = mel.rotomap.moles.molepos_to_nparray(end)
    vec = end_xy - start_xy
    angle_rads = math.atan2(vec[1], vec[0])
    return angle_rads * 180 / math.pi


def draw_neighbourhoods(image, moles):
    image = numpy.zeros(image.shape)
    for m in moles:
        x = m['x']
        y = m['y']

        # TODO: fix non-zero neighbours assumption

        neighbours = mel.rotomap.moles.sorted_by_distances(moles, x, y)
        dist = mel.rotomap.moles.dist_xy(neighbours[1], x, y)
        neighbours = [
            n for n in neighbours
            if mel.rotomap.moles.dist_xy(n, x, y) < dist * 2
        ]

        clut = mel.rotomap.tricolour._NINE_CLASS_SET1

        colour = (255, 255, 255)
        if len(neighbours) - 1 < len(clut):
            colour = clut[len(neighbours) - 1]

        draw_mole(image, m, colour)

        # Point at neighbours
        for n in neighbours:
            nx = n['x']
            ny = n['y']
            cv2.arrowedLine(
                image, (x, y), (nx, ny), colour, 2, cv2.LINE_AA)

    return image


def best_offset_theory(from_moles, to_moles, cutoff, offset_cutoff):
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
