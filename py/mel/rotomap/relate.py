"""Relate rotomaps to eachother."""

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
        theory = best_offset_theory(from_moles, to_moles)

    image = numpy.zeros(image.shape)
    overlay_theory(image, theory, from_dict, to_dict)
    return image


def overlay_theory(image, theory, from_dict, to_dict):
    colour_removed = (0, 0, 255)
    colour_added = (0, 255, 0)
    colour_mapped = (255, 0, 0)
    colour_known = (255, 255, 0)
    for from_, to in theory:
        assert (from_ is not None) or (to is not None)
        if to is None:
            draw_mole(image, from_dict[from_], colour_removed)
        elif from_ is None:
            draw_mole(image, to_dict[to], colour_added)
        else:
            colour = colour_mapped
            if from_ == to:
                colour = colour_known
            draw_from_to_mole(
                image,
                from_dict[from_],
                to_dict[to],
                colour)

    return image


def mole_list_to_uuid_dict(mole_list):
    return {m['uuid']: m for m in mole_list}


def best_offset_theory(from_moles, to_moles, cutoff=None, offset_cutoff=None):
    if not from_moles:
        raise ValueError('from_moles is empty')
    if not to_moles:
        raise ValueError('to_moles is empty')

    offset_cutoff_sq = None
    if offset_cutoff:
        offset_cutoff_sq = offset_cutoff ** 2

    if cutoff is not None:
        cutoff_sq = cutoff * cutoff

    from_dict = mole_list_to_uuid_dict(from_moles)
    to_dict = mole_list_to_uuid_dict(to_moles)
    from_set = set(from_dict.keys())
    to_set = set(to_dict.keys())

    in_both = from_set & to_set

    if in_both:
        theory = []
        theory.extend((u, u) for u in in_both)
        point_offsets = to_point_offsets(
            [(from_dict[m], to_dict[m]) for m in in_both])
        new_from_moles = [from_dict[m] for m in from_set if m not in in_both]
        new_to_moles = [to_dict[m] for m in to_set if m not in in_both]
        from_uuid_points = mel.rotomap.moles.to_uuid_points(new_from_moles)
        to_uuid_points = mel.rotomap.moles.to_uuid_points(new_to_moles)

        theory += make_offset_field_theory(
            from_uuid_points, to_uuid_points, point_offsets)

        return theory
    else:
        if cutoff is None:
            cutoff_sq = mole_min_sq_distance(to_moles)
            if cutoff_sq is None:
                cutoff_sq = 0
        return best_baseless_offset_theory(
            from_moles, to_moles, cutoff_sq, offset_cutoff_sq)


def to_point_offsets(mole_pairs):
    point_offsets = []
    for from_mole, to_mole in mole_pairs:
        from_pos = mel.rotomap.moles.molepos_to_nparray(from_mole)
        to_pos = mel.rotomap.moles.molepos_to_nparray(to_mole)
        point_offsets.append((from_pos, to_pos - from_pos))
    return point_offsets


def make_offset_field_theory(from_uuid_points, to_uuid_points, point_offsets):
    to_uuid_points = dict(to_uuid_points)
    inv_point_offsets = invert_point_offsets(point_offsets)
    theory = []
    for uuid_, point in from_uuid_points.items():
        if not to_uuid_points:
            theory.append((uuid_, None))
            continue

        offset, error = pick_value_from_field(point, point_offsets)
        to_uuid, distance = nearest_uuid_point(point + offset, to_uuid_points)
        if distance < error * 2:

            # Make sure that the closest match for the 'to' mole is also the
            # 'from' mole.
            to_point = to_uuid_points[to_uuid]
            inv_offset, inv_error = pick_value_from_field(
                to_point, inv_point_offsets)
            from_uuid, from_distance = nearest_uuid_point(
                to_point + inv_offset, from_uuid_points)

            if from_uuid == uuid_:
                theory.append((uuid_, to_uuid))
                del to_uuid_points[to_uuid]
            else:
                theory.append((uuid_, None))
        else:
            theory.append((uuid_, None))

    for uuid_ in to_uuid_points:
        theory.append((None, uuid_))

    return theory


def invert_point_offsets(point_offsets):
    return [
        (point + offset, -offset)
        for point, offset in point_offsets
    ]


def nearest_uuid_point(point, uuid_points):
    nearest_sqdist = None
    nearest_uuid = None
    for uuid_, q in uuid_points.items():
        offset = q - point
        sqdist = numpy.dot(offset, offset)
        if nearest_sqdist is None or sqdist < nearest_sqdist:
            nearest_sqdist = sqdist
            nearest_uuid = uuid_
    return nearest_uuid, numpy.sqrt(nearest_sqdist)


def pick_value_from_field(point, point_values):
    weights = []
    for q, v in point_values:
        offset = q - point
        sq_distance = numpy.dot(offset, offset)
        weights.append(1.0 / (sq_distance + 1))

    weights = numpy.array(weights)
    sum_ = numpy.sum(weights)
    weights /= sum_

    values = numpy.array([x[1] for x in point_values])
    picked_value = numpy.dot(values.T, weights)

    value_errors = numpy.array(
        [numpy.linalg.norm(picked_value - v) for v in values])

    picked_error = numpy.dot(value_errors.T, weights)

    return picked_value, picked_error


def best_baseless_offset_theory(
        from_moles, to_moles, cutoff_sq, offset_cutoff_sq):
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
    offset = numpy.array(offset)

    theory = []

    dist_sq_sum = 0

    for i, a in enumerate(from_moles):
        point = mel.rotomap.moles.molepos_to_nparray(a)
        point += offset
        best_index, best_dist_sq = nearest_mole_index_to_point(point, to_moles)
        if best_index is not None and best_dist_sq <= cutoff_sq:
            r_point = mel.rotomap.moles.molepos_to_nparray(
                to_moles[best_index])
            r_point -= offset
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