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

    window_name = "output"
    cv2.namedWindow(window_name)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_width = 800
    window_height = 600
    cv2.resizeWindow(window_name, window_width, window_height)

    is_finished = False
    while not is_finished:
        key = cv2.waitKey(50)

        if key != -1:
            raise Exception('User aborted.')

        ret, frame = cap.read()
        if not ret:
            raise Exception("Could not read frame.")

        cv2.imshow(window_name, frame)
