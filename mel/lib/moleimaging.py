"""Routines for analysing images of moles."""


import math

import cv2

import mel.lib.image
import mel.lib.math


def find_mole(frame):
    # look for areas of high saturation, they are likely moles
    img = frame.copy()
    img = cv2.blur(img, (40, 40))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.split(img)[1]
    _, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    ringed, stats, _ = mel.lib.moleimaging.process_contours(img, frame)
    return ringed, stats


def calc_hist(image, channel, mask):
    hist = cv2.calcHist([image], [channel], mask, [8], [0, 256])
    hist = [int(x) for x in hist]
    hist_sum = sum(hist)
    hist = [100 * x / hist_sum for x in hist]
    return hist


def log10_zero(x):
    """Return the log10 of x, map log10(0) -> 0."""
    if x == 0:
        return 0
    else:
        return math.log10(x)


def biggest_contour(image):
    _, contours, _ = cv2.findContours(
        image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    max_area = 0
    max_index = None
    for i, c in enumerate(contours):
        if c is not None and len(c) > 5:
            area = cv2.contourArea(c)
            if max_index is None or area > max_area:
                max_area = area
                max_index = i

    return contours[max_index]


def process_contours(mole_regions, original):

    final = original.copy()
    stats = None

    _, contours, _ = cv2.findContours(
        mole_regions.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    mole_contour, mole_area = find_mole_contour(
        contours, mole_regions.shape[0:2]
    )

    ellipse = None

    if mole_contour is not None:
        if len(mole_contour) > 5:

            ellipse = cv2.fitEllipse(mole_contour)

            blue = (255, 0, 0)

            moments = cv2.moments(mole_contour)
            hu_moments = cv2.HuMoments(moments)
            hu_moments = [log10_zero(abs(float(x))) for x in hu_moments]

            hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
            hist = calc_hist(hsv, 1, mole_regions)
            hist_surrounding = calc_hist(hsv, 1, None)

            cv2.ellipse(final, ellipse, blue, 5)
            sqrt_area = math.sqrt(mole_area)
            aspect_ratio = (ellipse[1][0] / ellipse[1][1]) * 100
            ellipse_area = math.pi * ellipse[1][0] * ellipse[1][1] * 0.25

            if ellipse_area:
                coverage_percent = (mole_area / ellipse_area) * 100
            else:
                coverage_percent = 0

            stats = (sqrt_area, aspect_ratio, coverage_percent)
            stats += tuple(hist)
            stats += tuple(hist_surrounding)
            stats += tuple(hu_moments)

    return final, stats, ellipse


def find_mole_contour(contours, width_height):

    centre = (
        width_height[0] // 2,
        width_height[1] // 2,
    )

    mole_contour = None
    mole_area = None
    mole_distance = None

    for contour in contours:
        if contour is not None and len(contour) > 5:
            area = cv2.contourArea(contour)
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                dx = centre[0] - cx
                dy = centre[1] - cy
                distance = (dx ** 2 + dy ** 2) ** 0.5
                if mole_area and area > 10 * mole_area:
                    mole_contour = contour
                    mole_area = area
                    mole_distance = distance
                elif mole_distance is None or distance < mole_distance:
                    mole_contour = contour
                    mole_area = area
                    mole_distance = distance

    return mole_contour, mole_area


class MoleAcquirer(object):
    def __init__(self):
        super(MoleAcquirer, self).__init__()
        self._is_locked = False
        self._last_stats = None
        self._last_stats_diff = None

    def update(self, stats):
        if stats and self._last_stats:

            stats_diff = list(map(
                lambda x, y: x - y,
                self._last_stats,
                stats))

            self._last_stats = list(map(
                lambda x, y: mel.lib.math.lerp(x, y, 0.5),
                self._last_stats,
                stats))

            if self._last_stats_diff:
                self._last_stats_diff = list(map(
                    lambda x, y: mel.lib.math.lerp(x, y, 0.5),
                    self._last_stats_diff,
                    stats_diff))

                should_lock = all(
                    [int(x) == 0 for x in self._last_stats_diff])

                should_unlock = any(
                    [abs(int(x)) > 1 for x in self._last_stats_diff])

                if not self._is_locked and should_lock:
                    self._is_locked = True
                elif self._is_locked and should_unlock:
                    self._is_locked = False
            else:
                self._last_stats_diff = stats_diff
                self._is_locked = False
        elif stats:
            self._last_stats = stats
            self._is_locked = False
        else:
            self._is_locked = False

    @property
    def is_locked(self):
        return self._is_locked


def point_to_int_point(point):
    return (int(point[0]), int(point[1]))


def rotate_point_around_pivot(point, pivot, degrees):
    centre_point = (point[0] - pivot[0], point[1] - pivot[1])
    theta = degrees * (math.pi / 180.0)
    rotated_point = (
        centre_point[0] * math.cos(theta) - centre_point[1] * math.sin(theta),
        centre_point[0] * math.sin(theta) + centre_point[1] * math.cos(theta),
    )
    new_point = (rotated_point[0] + pivot[0], rotated_point[1] + pivot[1])
    return new_point


def draw_vertical_lines(image, left, top, right, bottom, color, width):
    cv2.line(image, (left, top), (left, bottom), color, width)
    cv2.line(image, (right, top), (right, bottom), color, width)


def draw_horizontal_lines(image, left, top, right, bottom, color, width):
    cv2.line(image, (left, top), (right, top), color, width)
    cv2.line(image, (left, bottom), (right, bottom), color, width)


def annotate_image(original, is_rot_sensitive):
    is_aligned = False
    center_xy = None
    angle_degs = None
    original_width = original.shape[1]
    original_height = original.shape[0]

    final = original.copy()
    img = original.copy()
    img = cv2.blur(img, (40, 40))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.split(img)[1]
    _, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(
        img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    mole_contour, _ = find_mole_contour(contours, img.shape[0:2])

    if mole_contour is not None:
        if len(mole_contour) > 5:
            ellipse = cv2.fitEllipse(mole_contour)
            guide_ellipse = (ellipse[0], ellipse[1], 0)

            center_xy = ellipse[0]
            center_xy = point_to_int_point(center_xy)
            angle_degs = ellipse[2]

            top_xy = (center_xy[0], center_xy[1] - int(ellipse[1][1] / 2))
            ellipse_top_xy = rotate_point_around_pivot(
                top_xy, center_xy, angle_degs
            )
            ellipse_top_xy = point_to_int_point(ellipse_top_xy)

            yellow = (0, 255, 255)
            green = (0, 255, 0)
            red = (0, 0, 255)
            blue = (255, 0, 0)

            is_rotation_aligned = not is_rot_sensitive

            color = yellow
            thickness = 10
            if is_rot_sensitive:
                if angle_degs <= 20 or angle_degs >= 160:
                    is_rotation_aligned = True
                    color = green
                    thickness = 2
                else:
                    if angle_degs >= 45 and angle_degs <= 135:
                        color = red
                    cv2.ellipse(final, guide_ellipse, green, 10)
                    cv2.line(final, center_xy, top_xy, green, 10)
                    cv2.line(final, center_xy, ellipse_top_xy, color, 10)
            else:
                color = green
                thickness = 2

            cv2.ellipse(final, ellipse, color, thickness)

            bounds_half_width = original_width // 10
            bounds_half_height = original_height // 10
            bounds_center_x = original_width // 2
            bounds_center_y = original_height // 2
            bounds_left = bounds_center_x - bounds_half_width
            bounds_right = bounds_center_x + bounds_half_width
            bounds_top = bounds_center_y - bounds_half_height
            bounds_bottom = bounds_center_y + bounds_half_height

            is_position_aligned = True

            if center_xy[0] < bounds_left or center_xy[0] > bounds_right:
                is_position_aligned = False
                # show x guide
                draw_vertical_lines(
                    final,
                    bounds_left,
                    bounds_top,
                    bounds_right,
                    bounds_bottom,
                    blue,
                    10,
                )
                cv2.rectangle(final, center_xy, center_xy, blue, 10)

            if center_xy[1] < bounds_top or center_xy[1] > bounds_bottom:
                is_position_aligned = False
                # show y guide
                draw_horizontal_lines(
                    final,
                    bounds_left,
                    bounds_top,
                    bounds_right,
                    bounds_bottom,
                    blue,
                    10,
                )
                cv2.rectangle(final, center_xy, center_xy, blue, 10)

            if is_rotation_aligned and is_position_aligned:
                is_aligned = True

    original[:, :] = final[:, :]
    return is_aligned, center_xy, angle_degs


def find_mole_ellipse(original, centre, radius):

    lefttop = centre - (radius, radius)
    rightbottom = centre + (radius + 1, radius + 1)

    original = mel.lib.image.slice_square_or_none(
        original, lefttop, rightbottom
    )

    if original is None:
        return None

    image = original[:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.split(image)[1]
    image = cv2.equalizeHist(image)
    image = cv2.threshold(image, 252, 255, cv2.THRESH_BINARY)[1]
    image, _, ellipse = mel.lib.moleimaging.process_contours(image, original)

    if ellipse:
        ellipse = (
            (ellipse[0][0] + lefttop[0], ellipse[0][1] + lefttop[1]),
            ellipse[1],
            ellipse[2],
        )

    return ellipse


# -----------------------------------------------------------------------------
# Copyright (C) 2015-2017 Angelos Evripiotis.
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
