"""Show a live view through an attached microscope."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
        ringed = _process_contours(img, frame)

        # show the image with 'non-mole' areas encircled
        cv2.imshow(window_name, ringed)


def _find_mole_contour(contours):
    mole_contour = None
    mole_area = None
    for contour in contours:
        if contour is not None:
            area = cv2.contourArea(contour)
            if mole_area is None or area > mole_area:
                mole_contour = contour
                mole_area = area

    return mole_contour


def _process_contours(mole_regions, original):

    final = original.copy()

    contours, hierarchy = cv2.findContours(
        mole_regions.copy(),
        cv2.cv.CV_RETR_LIST,
        cv2.cv.CV_CHAIN_APPROX_NONE)

    mole_contour = _find_mole_contour(contours)
    if mole_contour is not None:
        if len(mole_contour) > 5:

            ellipse = cv2.fitEllipse(mole_contour)

            yellow = (0, 255, 255)
            green = (0, 255, 0)
            red = (0, 0, 255)
            blue = (255, 0, 0)

            cv2.ellipse(final, ellipse, blue, 5)

    return final
