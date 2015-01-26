"""Routines for analysing images of moles."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import cv2


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
