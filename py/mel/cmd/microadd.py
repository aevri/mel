"""Capture images from an attached microscope and add to existing moles."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import datetime
import os
import numpy

import mel.lib.moleimaging


def setup_parser(parser):
    parser.add_argument(
        'PATH',
        type=str,
        help="Path to the mole to add new microscope images to.")


def show_image_in_window(image, window_name):
    cv2.namedWindow(window_name)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_width = 800
    window_height = 600
    cv2.resizeWindow(window_name, window_width, window_height)
    cv2.imshow(window_name, image)


def load_context_image(path):

    # Paths should alpha-sort to recent last, pick the first jpg
    children = reversed(sorted(os.listdir(path)))
    for name in children:
        if name.lower().endswith('.jpg'):
            return cv2.imread(
                os.path.join(path, name))

    raise Exception("No images in {}".format(path))


def process_args(args):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open video capture device.")

    context_image = load_context_image(args.PATH)
    show_image_in_window(context_image, 'context')

    # create an 800x600 output window
    window_name = "output"
    cv2.namedWindow(window_name)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_width = 800
    window_height = 600
    cv2.resizeWindow(window_name, window_width, window_height)

    # loop until the user presses a key
    print("Press any key to exit.")
    mole_acquirer = mel.lib.moleimaging.MoleAcquirer()
    while True:
        key = cv2.waitKey(50)
        if key != -1:
            raise Exception('User aborted.')

        ret, frame = cap.read()
        if not ret:
            raise Exception("Could not read frame.")

        is_rot_sensitive = True
        ringed, stats = mel.lib.moleimaging.find_mole(frame)
        asys_image = numpy.copy(frame)
        is_aligned = mel.lib.moleimaging.annotate_image(
            asys_image,
            is_rot_sensitive)

        mole_acquirer.update(stats)

        cv2.imshow(window_name, asys_image)
        if mole_acquirer.is_locked and is_aligned:
            # show the image with mole encircled
            print("locked and aligned")
            break

    # write the mole image
    filename = mel.lib.common.make_now_datetime_string() + ".jpg"
    dirname = os.path.join(args.PATH, '__micro__')
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    file_path = os.path.join(dirname, filename)
    cv2.imwrite(file_path, frame)
