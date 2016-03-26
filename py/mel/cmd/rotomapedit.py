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
        '--images',
        nargs='+',
        action='append',
        required=True,
        help="A list of paths to images, specify multiple times for multiple "
             "sets.")
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


def process_args(args):

    display_width = args.display_width
    display_height = args.display_height

    if display_width is None or display_height is None:
        tkroot = Tkinter.Tk()
        if display_width is None:
            display_width = tkroot.winfo_screenwidth() - 50
        if display_height is None:
            display_height = tkroot.winfo_screenheight() - 150

    editor = mel.rotomap.display.Editor(
        args.images, display_width, display_height)

    left = 63234
    right = 63235
    up = 63232
    down = 63233

    # This must be a list in order for it to be referenced from the the
    # closure, in Python 3 we'll use "nonlocal".
    mole_uuid = [None]
    is_move_mode = [False]

    def mouse_callback(event, mouse_x, mouse_y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                editor.show_zoomed(mouse_x, mouse_y)
            elif flags & cv2.EVENT_FLAG_ALTKEY:
                if flags & cv2.EVENT_FLAG_SHIFTKEY:
                    mole_uuid[0] = editor.get_mole_uuid(mouse_x, mouse_y)
                    print(mole_uuid[0])
                else:
                    editor.set_mole_uuid(mouse_x, mouse_y, mole_uuid[0])
            elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                editor.remove_mole(mouse_x, mouse_y)
            else:
                if not is_move_mode[0]:
                    editor.add_mole(mouse_x, mouse_y)
                else:
                    editor.move_nearest_mole(mouse_x, mouse_y)

    editor.display.set_mouse_callback(mouse_callback)

    print("Press left for previous image, right for next image.")
    print("Press up for previous map, down for next map.")
    print("Click on a point to add or move a mole there and save.")
    print("Ctrl-click on a point to zoom in on it.")
    print("Shift-click on a point to delete it.")
    print("Alt-Shift-click on a point to copy it's uuid.")
    print("Alt-click on a point to paste the copied uuid.")
    print("Press 'm' to toggle move mode.")
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
                editor.show_prev()
                print(editor.moledata.current_image_path())
            elif key == right:
                editor.show_next()
                print(editor.moledata.current_image_path())
            elif key == up:
                editor.show_prev_map()
                print(editor.moledata.current_image_path())
            elif key == down:
                editor.show_next_map()
                print(editor.moledata.current_image_path())
            elif key == ord(' '):
                editor.show_fitted()
            elif key == ord('c'):
                copied_moles = editor.moledata.moles
            elif key == ord('m'):
                is_move_mode[0] = not is_move_mode[0]
            elif key == ord('a'):
                guessed_moles = guess_mole_positions(
                    copied_moles,
                    editor.moledata.moles,
                    editor.moledata.get_image())
                editor.set_moles(guessed_moles)
            elif key == 13:
                editor.toggle_markers()
            else:
                is_finished = True

    editor.display.clear_mouse_callback()


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
            prevpos = mel.rotomap.moles.molepos_to_nparray(prev_dict[m])
            currpos = mel.rotomap.moles.molepos_to_nparray(curr_dict[m])
            offset = currpos - prevpos

    for mole in previous_moles:
        if mole['uuid'] not in matched_uuids:
            new_m = copy.deepcopy(mole)
            pos = mel.rotomap.moles.molepos_to_nparray(new_m)
            if offset is not None:
                pos += offset
                mel.rotomap.moles.set_molepos_to_nparray(new_m, pos)

            ellipse = mel.lib.moleimaging.find_mole_ellipse(
                current_image, pos, 50)
            if ellipse is not None:
                mel.rotomap.moles.set_molepos_to_nparray(new_m, ellipse[0])

            new_moles.append(new_m)

    return new_moles
