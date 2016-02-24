"""Edit a 'rotomap' series of images."""

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
        help="Path to the rotomap image directory.")
    parser.add_argument(
        '--display-width',
        type=int,
        default=800,
        help="Width of the preview display window.")
    parser.add_argument(
        '--display-height',
        type=int,
        default=600,
        help="Width of the preview display window.")
    parser.add_argument(
        '--rot90',
        type=int,
        default=None,
        help="Rotate images 90 degrees clockwise this number of times.")


def process_args(args):
    display = mel.rotomap.display.Display(
        args.PATH, args.display_width, args.display_height, args.rot90)

    left = 63234
    right = 63235

    # This must be a list in order for it to be referenced from the the
    # closure, in Python 3 we'll use "nonlocal".
    mole_uuid = [None]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                image_x, image_y = display.windowxy_to_imagexy(x, y)
                display.show_zoomed(image_x, image_y)
            elif flags & cv2.EVENT_FLAG_ALTKEY:
                image_x, image_y = display.windowxy_to_imagexy(x, y)
                if flags & cv2.EVENT_FLAG_SHIFTKEY:
                    mole_uuid[0] = display.get_mole_uuid(image_x, image_y)
                    print(mole_uuid[0])
                else:
                    display.set_mole_uuid(image_x, image_y, mole_uuid[0])
            elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                image_x, image_y = display.windowxy_to_imagexy(x, y)
                display.remove_mole(image_x, image_y)
            else:
                image_x, image_y = display.windowxy_to_imagexy(x, y)
                display.add_mole(image_x, image_y)

    display.set_mouse_callback(mouse_callback)

    print("Press left for previous image, right for next image.")
    print("Click on a point to add a mole there and save.")
    print("Ctrl-click on a point to zoom in on it.")
    print("Press space to restore original zoom.")
    print("Press enter to toggle mole markers.")
    print("Press any other key to quit.")

    is_finished = False
    while not is_finished:
        key = cv2.waitKey(50)
        if key != -1:
            if key == left:
                display.show_prev()
                print(display.current_image_path())
            elif key == right:
                display.show_next()
                print(display.current_image_path())
            elif key == ord(' '):
                display.show_fitted()
            elif key == 13:
                display.toggle_markers()
            else:
                is_finished = True

    display.clear_mouse_callback()
