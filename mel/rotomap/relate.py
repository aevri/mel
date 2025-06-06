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
    x = mole["x"]
    y = mole["y"]
    if mole[mel.rotomap.moles.KEY_IS_CONFIRMED]:
        draw_canonical_mole(image, x, y, colour)
    else:
        draw_non_canonical_mole(image, x, y, colour)


def mole_list_to_uuid_dict(mole_list):
    return {m["uuid"]: m for m in mole_list}


def apply_theory(theory, to_moles):
    theory_to_original = {}
    for mole in to_moles:
        for from_, to in theory:
            if from_ is not None and mole["uuid"] == to:
                mole["uuid"] = from_
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


def best_theory(from_moles, to_moles, iterate):
    if not iterate:
        return best_offset_theory(from_moles, to_moles)

    to_moles = copy.deepcopy(to_moles)

    theory = None
    done = False
    theory_to_original = {}
    while not done:
        new_theory = reverse_theory(
            best_offset_theory(from_moles, to_moles), theory_to_original
        )
        done = new_theory == theory
        theory = new_theory
        if not done:
            theory_to_original.update(apply_theory(theory, to_moles))

    return theory


def best_offset_theory(from_moles, to_moles):
    if not from_moles:
        raise ValueError("from_moles is empty")
    if not to_moles:
        raise ValueError("to_moles is empty")

    theory = best_offset_field_theory(from_moles, to_moles)
    if theory is None:
        theory = best_baseless_offset_theory(from_moles, to_moles)
    return theory


def best_offset_field_theory(from_moles, to_moles):
    from_points, to_points, point_offsets, theory = offset_theory_points(
        from_moles, to_moles
    )

    if not point_offsets:
        return None

    return make_offset_field_theory(from_points, to_points, point_offsets, theory)


def offset_theory_points(from_moles, to_moles):
    """Return (from_points, to_points, point_offsets, theory) from input.

    Args:
        from_moles (list): A list of mole dicts to map from.
        to_moles (list): A list of mole dicts to map to.

    Returns:
        tuple: A tuple containing the from_uuid_points, to_uuid_points, point_offsets.
    """
    from_dict, to_dict, from_set, to_set, in_both = mole_list_overlap_info(
        from_moles, to_moles
    )

    theory = []
    theory.extend((u, u) for u in in_both)
    point_offsets = to_point_offsets([(from_dict[m], to_dict[m]) for m in in_both])
    new_from_moles = [from_dict[m] for m in from_set - in_both]
    new_to_moles = [to_dict[m] for m in to_set - in_both]
    from_uuid_points = mel.rotomap.moles.to_uuid_points(new_from_moles)
    to_uuid_points = mel.rotomap.moles.to_uuid_points(new_to_moles)

    return from_uuid_points, to_uuid_points, point_offsets, theory


def mole_list_overlap_info(from_moles, to_moles):
    from_dict = mole_list_to_uuid_dict(from_moles)
    to_dict = mole_list_to_uuid_dict(to_moles)
    from_set = set(from_dict.keys())
    to_set = set(to_dict.keys())
    in_both = from_set & to_set
    return from_dict, to_dict, from_set, to_set, in_both


def guess_mole_pos(from_uuid, from_moles, to_moles):
    """Return a numpy.array position guessing the location of uuid_, or None.

    Args:
        from_uuid (str): The UUID of the mole to guess the position of in to_moles.
        from_moles (list): A list of mole dicts to map from.
        to_moles (list): A list of mole dicts to map to.

    Returns:
        numpy.array or None: A numpy.array of the guessed position, or None if no guess.
    """
    from_points, to_points, point_offsets, _ = offset_theory_points(
        from_moles, to_moles
    )

    if not point_offsets:
        return None

    if from_uuid not in from_points:
        return None

    point = from_points[from_uuid]
    offset, error = pick_value_from_field(point, point_offsets)
    return (point + offset).astype(int)


def to_point_offsets(mole_pairs):
    point_offsets = []
    for from_mole, to_mole in mole_pairs:
        from_pos = mel.rotomap.moles.mole_to_point(from_mole)
        to_pos = mel.rotomap.moles.mole_to_point(to_mole)
        point_offsets.append((from_pos, to_pos - from_pos))
    return point_offsets


def make_offset_field_theory(from_uuid_points, to_uuid_points, point_offsets, theory):
    to_uuid_points = dict(to_uuid_points)
    inv_point_offsets = invert_point_offsets(point_offsets)
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
            inv_offset, inv_error = pick_value_from_field(to_point, inv_point_offsets)
            from_uuid, _ = nearest_uuid_point(to_point + inv_offset, from_uuid_points)

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
    return [(point + offset, -offset) for point, offset in point_offsets]


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
        ...     [(numpy.array([0, 0]), (1,))]) == ((1.0,), 0.0)
        True

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
            to_x = dest["x"] - source["x"]
            to_y = dest["y"] - source["y"]
            offset_dist_sq = to_x * to_x + to_y * to_y

            theory, dist_sq = make_offset_theory(
                from_moles, to_moles, (to_x, to_y), cutoff_sq
            )

            new_best = best_theory is None
            if not new_best and len(theory) < len(best_theory):
                new_best = True
            if not new_best and len(theory) == len(best_theory):
                if dist_sq < best_theory_dist_sq:
                    new_best = True
                if (
                    not new_best
                    and dist_sq == best_theory_dist_sq
                    and offset_dist_sq < best_theory_offset_dist_sq
                ):
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
            dist = _mole_distance_sq(a, b)
            if min_dist is None or dist < min_dist:
                min_dist = dist
    return min_dist


def make_offset_theory(from_moles, to_moles_in, offset, cutoff_sq):
    to_moles = list(to_moles_in)
    offset = numpy.array(offset)

    theory = []

    dist_sq_sum = 0

    for i, a in enumerate(from_moles):
        point = mel.rotomap.moles.mole_to_point(a)
        point += offset
        best_index, best_dist_sq = _nearest_mole_index_to_point(point, to_moles)
        if best_index is not None and best_dist_sq <= cutoff_sq:
            r_point = mel.rotomap.moles.mole_to_point(to_moles[best_index])
            r_point -= offset
            r_index, _ = _nearest_mole_index_to_point(r_point, from_moles)
            if i == r_index:
                theory.append((a["uuid"], to_moles[best_index]["uuid"]))
                del to_moles[best_index]
                dist_sq_sum += best_dist_sq
            else:
                theory.append((a["uuid"], None))
        else:
            theory.append((a["uuid"], None))

    for b in to_moles:
        theory.append((None, b["uuid"]))

    return theory, dist_sq_sum


def _nearest_mole_index_to_point(point, mole_list):
    best_index = None
    best_dist_sq = None
    for i, mole in enumerate(mole_list):
        dist_sq = mel.lib.math.distance_sq_2d(
            point, mel.rotomap.moles.mole_to_point(mole)
        )
        if best_index is None or dist_sq < best_dist_sq:
            best_index = i
            best_dist_sq = dist_sq
    return best_index, best_dist_sq


def _mole_distance_sq(from_mole, to_mole):
    return mel.lib.math.distance_sq_2d(
        mel.rotomap.moles.mole_to_point(from_mole),
        mel.rotomap.moles.mole_to_point(to_mole),
    )


# -----------------------------------------------------------------------------
# Copyright (C) 2018 Angelos Evripiotis.
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
