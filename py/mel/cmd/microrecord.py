"""Record videos of individual moles live from a microscope."""

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

    # read first frame to get dimensions
    ret, frame = cap.read()
    if not ret:
        raise Exception("Could not read frame.")
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    # create an 800x600 window
    window_name = "output"
    cv2.namedWindow(window_name)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_width = frame_width
    window_height = frame_height
    cv2.resizeWindow(window_name, window_width, window_height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter(
        'mole.avi',
        fourcc,
        25.0,
        (frame_width, frame_height))

    is_finished = False
    while not is_finished:
        key = cv2.waitKey(50)
        if key != -1:
            raise Exception('User aborted.')

        ret, frame = cap.read()
        if not ret:
            raise Exception("Could not read frame.")

        cv2.imshow(window_name, frame)
        out.write(frame)
