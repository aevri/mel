"""Manage image masks."""

import os

import cv2
import numpy


def path(mole_image_path):
    # Path might be a pathlib.Path, so convert to string first.
    return str(mole_image_path) + ".mask.png"


def load(mole_image_path):
    mask = cv2.imread(path(mole_image_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise Exception(f'Failed to load mask: "{path}"')
    return mask


def has_mask(mole_image_path):
    return os.path.isfile(path(mole_image_path))


def load_or_none(mole_image_path):
    if has_mask(mole_image_path):
        return load(mole_image_path)
    return None


def mask_biggest_region(mask):
    # Pick only the biggest connected region - there may be other things in the
    # image which have a similar colour profile. Assume that the biggest region
    # is the area that we're interested in.

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    max_area = 0
    max_index = None
    for i, c in enumerate(contours):
        if c is not None and len(c) > 5:
            area = cv2.contourArea(c)
            if max_index is None or area > max_area:
                max_area = area
                max_index = i

    mask = numpy.zeros(mask.shape, numpy.uint8)
    if max_index is not None:
        c = contours[max_index]
        cv2.drawContours(mask, [c], -1, 255, -1)

    return mask


def guess_mask_otsu(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    return mask_biggest_region(img)


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
