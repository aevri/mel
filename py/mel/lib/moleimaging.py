"""Routines for analysing images of moles."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import cv2

import mel.lib.math


def find_mole(frame):
    # look for areas of high saturation, they are likely moles
    img = frame.copy()
    img = cv2.blur(img, (40, 40))
    img = cv2.cvtColor(img, cv2.cv.CV_BGR2HSV)
    img = cv2.split(img)[1]
    _, img = cv2.threshold(img, 30, 255, cv2.cv.CV_THRESH_BINARY)
    ringed, stats = mel.lib.moleimaging.process_contours(img, frame)
    return ringed, stats


def process_contours(mole_regions, original):

    final = original.copy()
    stats = None

    contours, hierarchy = cv2.findContours(
        mole_regions.copy(),
        cv2.cv.CV_RETR_LIST,
        cv2.cv.CV_CHAIN_APPROX_NONE)

    mole_contour, mole_area = find_mole_contour(contours)
    if mole_contour is not None:
        if len(mole_contour) > 5:

            ellipse = cv2.fitEllipse(mole_contour)

            yellow = (0, 255, 255)
            green = (0, 255, 0)
            red = (0, 0, 255)
            blue = (255, 0, 0)

            cv2.ellipse(final, ellipse, blue, 5)
            stats = (math.sqrt(mole_area), ellipse[1][0], ellipse[1][1])

    return final, stats


def find_mole_contour(contours):
    mole_contour = None
    mole_area = None
    for contour in contours:
        if contour is not None:
            area = cv2.contourArea(contour)
            if mole_area is None or area > mole_area:
                mole_contour = contour
                mole_area = area

    return mole_contour, mole_area


class MoleAcquirer(object):

    def __init__(self):
        super(MoleAcquirer, self).__init__()
        self._is_locked = False
        self._was_locked = False
        self._last_stats = None
        self._last_stats_diff = None

    def update(self, stats):
        self._was_locked = self._is_locked
        if stats and self._last_stats:

            stats_diff = map(
                lambda x, y: x - y,
                self._last_stats,
                stats)

            self._last_stats = map(
                lambda x, y: mel.lib.math.lerp(x, y, 0.1),
                self._last_stats,
                stats)

            if self._last_stats_diff:
                self._last_stats_diff = map(
                    lambda x, y: mel.lib.math.lerp(x, y, 0.1),
                    self._last_stats_diff,
                    stats_diff)

                should_lock = all(
                    map(lambda x: int(x) == 0, self._last_stats_diff))

                should_unlock = any(
                    map(lambda x: abs(int(x)) > 1, self._last_stats_diff))

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

    def just_unlocked(self):
        return self._was_locked and not self._is_locked

    def just_locked(self):
        return not self._was_locked and self._is_locked

    @property
    def is_locked(self):
        return self._is_locked

    @property
    def last_stats(self):
        return self._last_stats
