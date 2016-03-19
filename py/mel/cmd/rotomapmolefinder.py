"""Find a mole in a rotomap."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2

import mel.lib.common
import mel.lib.image
import mel.lib.math

import mel.rotomap.display


def setup_parser(parser):
    parser.add_argument(
        'PATH',
        type=str,
        help="Path to the rotomap image.")
    parser.add_argument(
        '--rot90',
        type=int,
        default=None,
        help="Rotate images 90 degrees clockwise this number of times.")
    parser.add_argument(
        '--uuid',
        default=None,
        help="UUID of the mole to debug.")


def debug_display(image):
    name = 'debugdisplay'
    # width = 800
    # height = 600
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(name, width, height)
    cv2.imshow(name, image)
    while cv2.waitKey(50) == -1:
        pass


def process_args(args):
    image = mel.rotomap.display.load_image(args.PATH, args.rot90)
    moles = mel.rotomap.display.load_image_moles(args.PATH)
    display = mel.lib.ui.MultiImageDisplay('debug', 1200, 650)

    def add_debug_image(debugging_image):
        if len(debugging_image.shape) == 2:
            debugging_image = cv2.cvtColor(
                debugging_image, cv2.cv.CV_GRAY2BGR)

        if display.row_len() >= 5:
            display.new_row()
        display.add_image(debugging_image)

    if args.uuid is not None:
        mole_map = {m["uuid"]: m for m in moles}
        mel.lib.moleimaging.find_mole_ellipse(
            image, mole_map[args.uuid], 100, add_debug_image)
    else:
        i = 0
        for m in moles:
            mel.lib.moleimaging.find_mole_ellipse(
                image, m, 100, lambda x: None)[0]
            i += 1
            if i % 8 == 0:
                display.new_row()

    while cv2.waitKey(50) == -1:
        pass
