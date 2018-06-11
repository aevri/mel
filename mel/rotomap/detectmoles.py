"""Detect moles in an image."""

import cv2
import numpy

import mel.rotomap.moles


# Magic number to specify how close to the edge of an inclusion mask we can be
# before we should discard detected blobs. Generally it seems that we get a lot
# of falso positives near those edges.
_MASK_EXCLUSION_SQUARE_SIZE = 5


def draw_debug(image, mask):
    keypoints_, image = _keypoints(image, mask)
    image = cv2.drawKeypoints(
        image,
        keypoints_,
        numpy.array([]),
        (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return image


def moles(image, mask):
    moles_ = []
    for point in keypoints(image, mask):
        xy = point.pt

        # Exclude points that are near mask boundaries
        if mask is not None:
            if not _is_mask_region_all_set(
                    mask, xy, _MASK_EXCLUSION_SQUARE_SIZE):
                continue

        mel.rotomap.moles.add_mole(moles_, int(xy[0]), int(xy[1]))
        moles_[-1]['radius'] = point.size // 2
    return moles_


def _is_mask_region_all_set(mask, point, region_size):
    x, y = (int(i) for i in point)
    x_slice = slice(x - region_size, x + region_size)
    y_slice = slice(y - region_size, y + region_size)
    mask_slice = y_slice, x_slice

    # If we do the .mean() test on a region with no values then numpy will
    # print a RuntimeWarning about it. Therefore here we avoid calling .mean()
    # for this case.
    has_values = mask[mask_slice].size > 0

    return has_values and mask[mask_slice].mean() == 255


def keypoints(image, mask):
    return _keypoints(image, mask)[0]


def _keypoints(original_image, mask):
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    image = image[:, :, 1]

    image = cv2.bitwise_and(image, image, mask=mask)
    image = cv2.bitwise_not(image, image)

    # Note that the static analysis tool 'vulture' doesn't seem to be happy
    # with using attributes on 'params'. The only workaround appears to be
    # ignoring the whole file.
    params = cv2.SimpleBlobDetector_Params()
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    params.minThreshold = 0
    params.maxThreshold = 256

    params.filterByArea = True
    params.maxArea = 10000
    params.minArea = 25

    params.filterByInertia = True
    params.minInertiaRatio = 0.1

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints_ = detector.detect(image)

    return keypoints_, image


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
