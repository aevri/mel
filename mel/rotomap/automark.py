"""Automatically mark moles on rotomap images."""

import copy

import numpy

import mel.lib.image
import mel.rotomap.detectmoles
import mel.rotomap.moles


def merge_in_radiuses(targets, radii_sources, error_distance, only_merge):

    match_uuids, _, added_uuids = match_moles_by_pos(
        targets, radii_sources, error_distance
    )

    target_to_radii_src = {
        from_uuid: to_uuid for from_uuid, to_uuid in match_uuids
    }
    radii_src_radius = {s["uuid"]: s["radius"] for s in radii_sources}
    target_uuid_radius = {
        t_uuid: radii_src_radius[target_to_radii_src[t_uuid]]
        for t_uuid in (t["uuid"] for t in targets)
        if t_uuid in target_to_radii_src
    }

    results = []
    for t in targets:
        t_copy = copy.deepcopy(t)
        results.append(t_copy)
        if "radius" not in t_copy:
            radius = target_uuid_radius.get(t_copy["uuid"], None)
            if radius is not None:
                t_copy["radius"] = radius

    if not only_merge:
        for r in radii_sources:
            if r["uuid"] in added_uuids:
                results.append(r)

    return results


def _array_nonempty(numpy_array):
    return all(numpy_array.shape)


def match_pos_vecs(from_pos_vec, to_pos_vec, error_distance):
    max_sqdist = error_distance ** 2

    # pylint: disable=no-member
    # Avoid this error:
    #
    #   E1101: Function 'subtract' has no 'outer' member (no-member)
    #
    distmatx = numpy.subtract.outer(from_pos_vec[:, 0], to_pos_vec[:, 0])
    distmaty = numpy.subtract.outer(from_pos_vec[:, 1], to_pos_vec[:, 1])
    # pylint: enable=no-member

    sqdistmat = numpy.square(distmatx) + numpy.square(distmaty)

    assert sqdistmat.shape == (len(from_pos_vec), len(to_pos_vec))

    # print(sqdistmat)

    matindex_to_fromindex = list(range(len(from_pos_vec)))
    matindex_to_toindex = list(range(len(to_pos_vec)))

    matches = []
    while _array_nonempty(sqdistmat):
        nextmin_i = numpy.argmin(sqdistmat)  # Gives us the 'flat index'.
        from_i, to_i = numpy.unravel_index(nextmin_i, sqdistmat.shape)
        nextmin = sqdistmat[from_i, to_i]
        if nextmin > max_sqdist:
            break
        # It seems pylint doesn't know about numpy indexing.
        # pylint: disable=invalid-sequence-index
        matches.append(
            (
                matindex_to_fromindex[from_i],
                matindex_to_toindex[to_i],
            )
        )
        del matindex_to_fromindex[from_i]
        del matindex_to_toindex[to_i]
        sqdistmat = numpy.delete(sqdistmat, from_i, axis=0)
        sqdistmat = numpy.delete(sqdistmat, to_i, axis=1)
        # pylint: enable=invalid-sequence-index

    # print()

    missing = matindex_to_fromindex
    added = matindex_to_toindex

    return matches, missing, added


def match_moles_by_pos(from_moles, to_moles, error_distance):

    if from_moles and not to_moles:
        return [], [m["uuid"] for m in from_moles], []
    elif not from_moles and to_moles:
        return [], [], [m["uuid"] for m in to_moles]
    elif not from_moles and not to_moles:
        return [], [], []

    from_pos_vec = mel.rotomap.moles.mole_list_to_pointvec(from_moles)
    to_pos_vec = mel.rotomap.moles.mole_list_to_pointvec(to_moles)

    vec_matches, vec_missing, vec_added = match_pos_vecs(
        from_pos_vec, to_pos_vec, error_distance
    )

    matches = [
        (from_moles[from_i]["uuid"], to_moles[to_i]["uuid"])
        for from_i, to_i in vec_matches
    ]
    missing = [from_moles[from_i]["uuid"] for from_i in vec_missing]
    added = [to_moles[to_i]["uuid"] for to_i in vec_added]

    return matches, missing, added


# -----------------------------------------------------------------------------
# Copyright (C) 2020 Angelos Evripiotis.
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
