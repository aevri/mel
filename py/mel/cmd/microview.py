"""Show a live view through an attached microscope."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import cv2


def setup_parser(parser):
    pass


def process_args(args):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open video capture device.")

    # create an 800x600 window
    window_name = "output"
    cv2.namedWindow(window_name)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_width = 800
    window_height = 600
    cv2.resizeWindow(window_name, window_width, window_height)

    # loop until the user presses a key
    print("Press any key to exit.")
    is_finished = False
    last_stats = None
    last_stats_diff = None
    while not is_finished:
        key = cv2.waitKey(50)
        if key != -1:
            raise Exception('User aborted.')

        ret, frame = cap.read()
        if not ret:
            raise Exception("Could not read frame.")

        # highlight areas of high saturation, they are likely moles
        img = frame.copy()
        img = cv2.blur(img, (40, 40))
        img = cv2.cvtColor(img, cv2.cv.CV_BGR2HSV)
        img = cv2.split(img)[1]
        _, img = cv2.threshold(img, 30, 255, cv2.cv.CV_THRESH_BINARY)
        ringed, stats = _process_contours(img, frame)

        if stats and last_stats:
            stats_diff = map(lambda x, y: x - y, last_stats, stats)
            last_stats = map(lambda x, y: _lerp(x, y, 0.1), last_stats, stats)

            if last_stats_diff:
                last_stats_diff = map(
                    lambda x, y: _lerp(x, y, 0.1), last_stats_diff, stats_diff)

                print(
                    map(lambda x: int(x), last_stats),
                    map(lambda x: int(x), last_stats_diff)
                )
            else:
                last_stats_diff = stats_diff


        elif stats:
            last_stats = stats

        # show the image with 'non-mole' areas encircled
        cv2.imshow(window_name, ringed)


def _lerp(origin, target, factor_0_to_1):
    towards = target - origin
    return origin + (towards * factor_0_to_1)


def _find_mole_contour(contours):
    mole_contour = None
    mole_area = None
    for contour in contours:
        if contour is not None:
            area = cv2.contourArea(contour)
            if mole_area is None or area > mole_area:
                mole_contour = contour
                mole_area = area

    return mole_contour, mole_area


def _process_contours(mole_regions, original):

    final = original.copy()
    stats = None

    contours, hierarchy = cv2.findContours(
        mole_regions.copy(),
        cv2.cv.CV_RETR_LIST,
        cv2.cv.CV_CHAIN_APPROX_NONE)

    mole_contour, mole_area = _find_mole_contour(contours)
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
