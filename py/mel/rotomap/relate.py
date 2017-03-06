"""Relate rotomaps to eachother."""

import copy

import cv2
import numpy

import mel.lib.debugrenderer
import mel.lib.math
import mel.rotomap.moles
import mel.rotomap.tricolour


_DEBUG_RENDERER = mel.lib.debugrenderer.GlobalContext()

# After experimentation with 'mel-debug bench-relate --reset-uuids 1', it turns
# out that pick_value_from_field() is performing much worse than what came
# before. It turns out that this is due to the error bounds it calculates.
#
# Instead of calculating the error bounds to use, simply pick a number that
# works for a reasonable number of images and use that instead. The performance
# is dramatically better.
#
# I'm guessing this magic number is only likely to work for my particular set
# of rotomaps, where the image sizes are the same and the distance from the
# camera is similar.
#
# In later work, we'll want to find a way of automatically determining the best
# error value to use.
#
# Before pick_value_from_field(), the results for --reset-uuids 1 were:
#     155 Flawed
#     177 Flawless
#
# The results for --reset-all-uuids show that improvements can be made:
#     88 Flawed
#     244 Flawless
#
# After pick_value_from_field() the results got much worse:
#     303 Flawed
#     29 Flawless
#
# After _MAGIC_FIELD_ERROR they yielded the best results yet:
#     12 Flawed
#     320 Flawless
#
# Similar results were observed for '--reset-uuids 5', where having more
# unknowns could mean that a large radius is less successful.
#
_MAGIC_FIELD_ERROR = 300


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


def draw_debug(to_image, to_mask, to_moles, from_image, from_mask, from_moles):
    image = numpy.zeros(to_image.shape)

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
        with _DEBUG_RENDERER.image_context(image):
            theory = best_offset_theory(from_moles, to_moles)

    overlay_theory(image, theory, from_dict, to_dict)

    if from_image is not None and from_mask is not None:
        point_offsets = lk_point_offsets(
            from_image, to_image, from_mask, to_mask)
        for point, offset in point_offsets:
            cv2.arrowedLine(
                image,
                tuple(point),
                tuple(point + offset),
                (0, 255, 255),
                2,
                cv2.LINE_AA)

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


def apply_theory(theory, to_moles):
    theory_to_original = {}
    for mole in to_moles:
        for from_, to in theory:
            if from_ is not None and mole['uuid'] == to:
                mole['uuid'] = from_
                if from_ != to:
                    theory_to_original[from_] = to
    return theory_to_original


def reverse_theory(theory, theory_to_original):
    new_theory = []
    for from_, to in theory:
        if to in theory_to_original:
            new_theory.append((from_, theory_to_original[to]))
        else:
            new_theory.append((from_, to))
    return new_theory


def best_theory(
        from_image,
        to_image,
        from_mask,
        to_mask,
        from_moles,
        to_moles,
        iterate):

    if not iterate:
        return best_offset_theory(from_moles, to_moles)

    to_moles = copy.deepcopy(to_moles)

    point_offsets = lk_point_offsets(from_image, to_image, from_mask, to_mask)

    theory = None
    done = False
    theory_to_original = {}
    while not done:
        new_theory = reverse_theory(
            best_offset_theory(from_moles, to_moles, point_offsets),
            theory_to_original)
        done = new_theory == theory
        theory = new_theory
        if not done:
            theory_to_original.update(apply_theory(theory, to_moles))

    return theory


def lk_point_offsets(from_image, to_image, from_mask, to_mask):
    from_gray = cv2.cvtColor(from_image, cv2.COLOR_BGR2GRAY)
    to_gray = cv2.cvtColor(to_image, cv2.COLOR_BGR2GRAY)

    from_points = cv2.goodFeaturesToTrack(
        from_gray,
        maxCorners=0,
        qualityLevel=0.5,
        minDistance=10,
        mask=from_mask)

    to_points, status, _ = cv2.calcOpticalFlowPyrLK(
        from_gray, to_gray, from_points, None)

    from_points = from_points.astype(int)
    to_points = to_points.astype(int)
    stat_point_offsets = zip(status, from_points, to_points - from_points)
    point_offsets = [
        (numpy.squeeze(p), numpy.squeeze(o)) for s, p, o in stat_point_offsets
        if s[0]
    ]

    return point_offsets


def best_offset_theory(from_moles, to_moles, point_offsets=None):
    if point_offsets is None:
        point_offsets = []

    if not from_moles:
        raise ValueError('from_moles is empty')
    if not to_moles:
        raise ValueError('to_moles is empty')

    from_dict = mole_list_to_uuid_dict(from_moles)
    to_dict = mole_list_to_uuid_dict(to_moles)
    from_set = set(from_dict.keys())
    to_set = set(to_dict.keys())

    in_both = from_set & to_set

    if in_both:
        theory = []
        theory.extend((u, u) for u in in_both)
        point_offsets.extend(to_point_offsets(
            [(from_dict[m], to_dict[m]) for m in in_both]))
        new_from_moles = [from_dict[m] for m in from_set if m not in in_both]
        new_to_moles = [to_dict[m] for m in to_set if m not in in_both]
        from_uuid_points = mel.rotomap.moles.to_uuid_points(new_from_moles)
        to_uuid_points = mel.rotomap.moles.to_uuid_points(new_to_moles)

        theory += make_offset_field_theory(
            from_uuid_points, to_uuid_points, point_offsets)

        return theory
    else:
        return best_baseless_offset_theory(from_moles, to_moles)


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

        # Note that an attempt to lerp between _MAGIC_FIELD_ERROR and 'error'
        # based on len(point_offsets) has been tried. If the lerp saturates at
        # len(point_offsets)==25, then it seems to perform better in one image
        # out of hundreds. Otherwise _MAGIC_FIELD_ERROR is still best.

        if distance < _MAGIC_FIELD_ERROR:

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
                _DEBUG_RENDERER.arrow(to_point, to_point + inv_offset)
                _DEBUG_RENDERER.circle(to_point + inv_offset, inv_error)
                theory.append((uuid_, None))
        else:
            _DEBUG_RENDERER.arrow(point, point + offset)
            _DEBUG_RENDERER.circle(point + offset, error)
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
    """Return (value, error) sampled from supplied array of (point, value).

    Given a number of points in space, which have values associated with them -
    'point_values'; return a best guess for what the value might be at the
    supplied 'point'.

    Note that values are expected to be numpy.array's.

    Usage example:

        Sample based on only a single point, expect the supplied value with
        certainty.

        >>> pick_value_from_field(
        ...     numpy.array([0, 0]),
        ...     [(numpy.array([0, 0]), (1,))])
        (array([ 1.]), 0.0)

    :point: a numpy.array representing a 2d point to take a sample from.
    :point_values: an array of (point, value) to sample at supplied 'point'.
    :returns: a tuple, (sampled_value, estimated_error).

    """

    # Note that the idea of passing in 'points' and 'values' as separate
    # variables, already as numpy.arrays has been tried. This was in order to
    # avoid the array -> numpy.array listcomp conversions happening inside a
    # loop. It turns out that there is no noticable difference in speed, and it
    # slightly complicated all the existing client code. May as well leave
    # as-is.

    offsets = numpy.array([q - point for q, v in point_values])
    sq_distances = numpy.sum(offsets * offsets, axis=1)

    sqweights = 1.0 / (sq_distances + 1)
    sqweights /= numpy.sum(sqweights)

    values = numpy.array([x[1] for x in point_values])
    picked_value = numpy.dot(values.T, sqweights)

    # Note that inverse-distance instead of inverse-square-distance for
    # calculating error has been tried. This did not improve results as
    # measured by 'mel-debug bench-relate'. Also tried inverse-log10-distance,
    # and 'equal weights'.

    value_errors = numpy.linalg.norm(values - picked_value, axis=1)
    picked_error = numpy.dot(value_errors.T, sqweights)

    return picked_value, picked_error


def best_baseless_offset_theory(from_moles, to_moles):

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
