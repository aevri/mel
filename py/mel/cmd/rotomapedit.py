"""Edit a 'rotomap' series of images."""


import copy

import cv2

import mel.lib.common
import mel.lib.image
import mel.lib.math
import mel.lib.ui

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
    parser.add_argument(
        '--follow',
        type=str,
        default=None,
        help="UUID of a mole to follow.")


def process_args(args):

    editor = mel.rotomap.display.Editor(
        args.images, args.display_width, args.display_height)

    mel.lib.ui.bring_python_to_front()

    if args.follow:
        editor.follow(args.follow)

    mole_uuid = None
    is_move_mode = False

    def mouse_callback(event, mouse_x, mouse_y, flags, _param):
        del _param
        nonlocal mole_uuid
        nonlocal is_move_mode
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                editor.show_zoomed(mouse_x, mouse_y)
            elif flags & cv2.EVENT_FLAG_ALTKEY:
                if flags & cv2.EVENT_FLAG_SHIFTKEY:
                    mole_uuid = editor.get_mole_uuid(mouse_x, mouse_y)
                    print(mole_uuid)
                else:
                    editor.set_mole_uuid(mouse_x, mouse_y, mole_uuid)
            elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                editor.remove_mole(mouse_x, mouse_y)
            else:
                if not is_move_mode:
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
            if key == mel.lib.ui.WAITKEY_LEFT_ARROW:
                editor.show_prev()
                if args.follow:
                    editor.follow(args.follow)
                print(editor.moledata.current_image_path())
            elif key == mel.lib.ui.WAITKEY_RIGHT_ARROW:
                editor.show_next()
                if args.follow:
                    editor.follow(args.follow)
                print(editor.moledata.current_image_path())
            elif key == mel.lib.ui.WAITKEY_UP_ARROW:
                editor.show_prev_map()
                if args.follow:
                    editor.follow(args.follow)
                print(editor.moledata.current_image_path())
            elif key == mel.lib.ui.WAITKEY_DOWN_ARROW:
                editor.show_next_map()
                if args.follow:
                    editor.follow(args.follow)
                print(editor.moledata.current_image_path())
            elif key == ord(' '):
                editor.show_fitted()
            elif key == ord('c'):
                copied_moles = editor.moledata.moles
            elif key == ord('m'):
                is_move_mode = not is_move_mode
            elif key == ord('a'):
                guessed_moles = guess_mole_positions(
                    copied_moles,
                    editor.moledata.moles,
                    editor.moledata.get_image())
                editor.set_moles(guessed_moles)
            elif key == ord('f'):
                editor.toggle_faded_markers()
            elif key == 13:
                editor.toggle_markers()
            else:
                is_finished = True

    editor.display.clear_mouse_callback()


def guess_mole_positions(previous_moles, current_moles, current_image):
    prev_uuids = set(m['uuid'] for m in previous_moles)
    curr_uuids = set(m['uuid'] for m in current_moles)
    matched_uuids = prev_uuids.intersection(curr_uuids)

    prev_moles_for_mapping = [
        m for m in previous_moles
        if m['uuid'] in matched_uuids
    ]

    image_rect = (0, 0, current_image.shape[1], current_image.shape[0])

    new_moles = copy.deepcopy(current_moles)
    for mole in previous_moles:
        if mole['uuid'] not in matched_uuids:
            new_m = copy.deepcopy(mole)
            pos = mel.rotomap.moles.molepos_to_nparray(new_m)

            # XXX: assume that current_image and prev_image have the same
            #      dimensions
            moles_for_mapping = mel.rotomap.moles.get_best_moles_for_mapping(
                pos, prev_moles_for_mapping, image_rect)

            if moles_for_mapping:
                pos = mel.rotomap.moles.mapped_pos(
                    pos, moles_for_mapping, current_moles)
                mel.rotomap.moles.set_molepos_to_nparray(new_m, pos)

            ellipse = mel.lib.moleimaging.find_mole_ellipse(
                current_image, pos, 50)
            if ellipse is not None:
                mel.rotomap.moles.set_molepos_to_nparray(new_m, ellipse[0])

            new_moles.append(new_m)

    return new_moles
