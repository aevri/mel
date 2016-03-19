"""Edit a 'rotomap' series of images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import Tkinter
import copy

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
        default=None,
        help="Width of the preview display window.")
    parser.add_argument(
        '--display-height',
        type=int,
        default=None,
        help="Width of the preview display window.")
    parser.add_argument(
        '--rot90',
        type=int,
        default=None,
        help="Rotate images 90 degrees clockwise this number of times.")


def process_args(args):

    display_width = args.display_width
    display_height = args.display_height

    if display_width is None or display_height is None:
        tkroot = Tkinter.Tk()
        if display_width is None:
            display_width = tkroot.winfo_screenwidth() - 50
        if display_height is None:
            display_height = tkroot.winfo_screenheight() - 150

    display = mel.rotomap.display.Display(
        args.PATH, display_width, display_height, args.rot90)

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
    print("Shift-click on a point to delete it.")
    print("Alt-Shift-click on a point to copy it's uuid.")
    print("Alt-click on a point to paste the copied uuid.")
    print("Press 'c' to copy the moles in the displayed image.")
    print("Press 'a' to auto-paste the copied moles in the displayed image.")
    print("Press space to restore original zoom.")
    print("Press enter to toggle mole markers.")
    print("Press any other key to quit.")

    copied_moles = None

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
            elif key == ord('c'):
                copied_moles = display.get_moles()
            elif key == ord('a'):
                guessed_moles = guess_mole_positions(
                    copied_moles,
                    display.get_moles(),
                    display.get_image())
                display.set_moles(guessed_moles)
            elif key == 13:
                display.toggle_markers()
            else:
                is_finished = True

    display.clear_mouse_callback()


def guess_mole_positions(previous_moles, current_moles, current_image):
    prev_uuids = set(m['uuid'] for m in previous_moles)
    curr_uuids = set(m['uuid'] for m in current_moles)
    matched_uuids = prev_uuids.intersection(curr_uuids)

    new_moles = copy.deepcopy(current_moles)

    offset = None
    if matched_uuids:
        prev_dict = {m['uuid']: m for m in previous_moles}
        curr_dict = {m['uuid']: m for m in current_moles}
        for m in matched_uuids:
            prevpos = mel.lib.moleimaging.molepos_to_nparray(prev_dict[m])
            currpos = mel.lib.moleimaging.molepos_to_nparray(curr_dict[m])
            offset = currpos - prevpos

    for mole in previous_moles:
        if mole['uuid'] not in matched_uuids:
            new_m = copy.deepcopy(mole)
            if offset is not None:
                pos = mel.lib.moleimaging.molepos_to_nparray(new_m)
                pos += offset
                mel.lib.moleimaging.set_molepos_to_nparray(new_m, pos)

            ellipse = mel.lib.moleimaging.find_mole_ellipse(
                current_image, new_m, 50)
            if ellipse is not None:
                mel.lib.moleimaging.set_molepos_to_nparray(new_m, ellipse[0])

            new_moles.append(new_m)

    return new_moles
