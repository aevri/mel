"""Detect moles in an image, using deep neural nets."""

import numpy as np

import mel.rotomap.automark


def boxes_to_poslist(boxes):
    poslist = [
        [int(0.5 * (xmin + xmax)), int(0.5 * (ymin + ymax))]
        for xmin, ymin, xmax, ymax in boxes
    ]
    return np.array(poslist)


def calc_precision_recall(target_poslist, poslist, error_distance=5):
    if not len(poslist):
        return 0, 0
    vec_matches, vec_missing, vec_added = mel.rotomap.automark.match_pos_vecs(
        target_poslist, poslist, error_distance
    )
    precision = len(vec_matches) / (len(vec_matches) + len(vec_added))
    recall = len(vec_matches) / (len(vec_matches) + len(vec_missing))
    return precision, recall


# -----------------------------------------------------------------------------
# Copyright (C) 2023 Angelos Evripiotis.
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
