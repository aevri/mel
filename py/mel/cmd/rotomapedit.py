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


class MoveController():

    def __init__(self):
        self.status = 'Move mode'

    def on_mouse_event(self, editor, event, mouse_x, mouse_y, flags, _param):
        del _param
        if event == cv2.EVENT_LBUTTONDOWN:
            editor.move_nearest_mole(mouse_x, mouse_y)
            return True

    def on_key(self, editor, key):
        pass


class FollowController():

    def __init__(self, editor, follow, mole_uuid_list):
        self.mole_uuid_list = mole_uuid_list
        if follow:
            self.mole_uuid_list[0] = follow
            editor.follow(self.mole_uuid_list[0])

        self.is_paste_mode = False
        self.update_status()

    def on_mouse_event(self, editor, event, mouse_x, mouse_y, flags, _param):
        del _param
        if event == cv2.EVENT_LBUTTONDOWN:
            editor.crud_mole(self.mole_uuid_list[0], mouse_x, mouse_y)
            editor.follow(self.mole_uuid_list[0])

    def pre_key(self, editor, key):
        self._prev_moles = editor.moledata.moles

    def on_key(self, editor, key):
        arrows = [
            mel.lib.ui.WAITKEY_LEFT_ARROW,
            mel.lib.ui.WAITKEY_RIGHT_ARROW,
            mel.lib.ui.WAITKEY_UP_ARROW,
            mel.lib.ui.WAITKEY_DOWN_ARROW,
        ]
        if key in arrows:
            update_follow(
                editor,
                self.mole_uuid_list[0],
                self._prev_moles,
                self.is_paste_mode)
            return True
        elif key == ord('p'):
            self.is_paste_mode = not self.is_paste_mode
            self.update_status()
            editor.set_status(self.status)
            editor.show_current()
            return True

    def update_status(self):
        if self.is_paste_mode:
            self.status = 'follow paste mode'
        else:
            self.status = 'follow mode'


class Controller():

    def __init__(self, editor, follow):
        self.mole_uuid_list = [None]

        self.follow_controller = FollowController(
            editor, follow, self.mole_uuid_list)
        self.move_controller = MoveController()
        self.sub_controller = None

        self.copied_moles = None

    def on_mouse_event(self, editor, event, mouse_x, mouse_y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                editor.show_zoomed(mouse_x, mouse_y)
            elif flags & cv2.EVENT_FLAG_ALTKEY:
                if flags & cv2.EVENT_FLAG_SHIFTKEY:
                    self.mole_uuid_list[0] = editor.get_mole_uuid(
                        mouse_x, mouse_y)
                    print(self.mole_uuid_list[0])
                else:
                    editor.set_mole_uuid(
                        mouse_x, mouse_y, self.mole_uuid_list[0])
            elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                editor.remove_mole(mouse_x, mouse_y)
            else:
                if self.sub_controller:
                    if self.sub_controller.on_mouse_event(
                            editor, event, mouse_x, mouse_y, flags, _param):
                        return
                editor.add_mole(mouse_x, mouse_y)

    def on_key(self, editor, key):
        if self.sub_controller:
            try:
                self.sub_controller.pre_key(editor, key)
            except AttributeError:
                pass

        if key == mel.lib.ui.WAITKEY_LEFT_ARROW:
            editor.show_prev()
            print(editor.moledata.current_image_path())
        elif key == mel.lib.ui.WAITKEY_RIGHT_ARROW:
            editor.show_next()
            print(editor.moledata.current_image_path())
        elif key == mel.lib.ui.WAITKEY_UP_ARROW:
            editor.show_prev_map()
            print(editor.moledata.current_image_path())
        elif key == mel.lib.ui.WAITKEY_DOWN_ARROW:
            editor.show_next_map()
            print(editor.moledata.current_image_path())
        elif key == ord(' '):
            editor.show_fitted()
        elif key == ord('c'):
            self.copied_moles = editor.moledata.moles
        elif key == ord('o'):
            is_follow = self.sub_controller is self.follow_controller
            if not is_follow and self.mole_uuid_list[0]:
                self.sub_controller = self.follow_controller
                editor.set_status(self.sub_controller.status)
                print(self.mole_uuid_list[0])
            else:
                self.sub_controller = None
                editor.set_status('')
            editor.show_current()
        elif key == ord('m'):
            if not self.sub_controller == self.move_controller:
                self.sub_controller = self.move_controller
                editor.set_status(self.sub_controller.status)
            else:
                self.sub_controller = None
                editor.set_status('')
            editor.show_current()
        elif key == ord('a'):
            guessed_moles = guess_mole_positions(
                self.copied_moles,
                editor.moledata.moles,
                editor.moledata.get_image())
            editor.set_moles(guessed_moles)
        elif key == ord('f'):
            editor.toggle_faded_markers()
        elif key == 13:
            editor.toggle_markers()

        if self.sub_controller:
            try:
                self.sub_controller.on_key(editor, key)
            except AttributeError:
                pass


def process_args(args):

    editor = mel.rotomap.display.Editor(
        args.images, args.display_width, args.display_height)

    mel.lib.ui.bring_python_to_front()

    controller = Controller(editor, args.follow)

    def mouse_callback(*args):
        controller.on_mouse_event(editor, *args)

    editor.display.set_mouse_callback(
        mouse_callback)

    print("Press left for previous image, right for next image.")
    print("Press up for previous map, down for next map.")
    print("Click on a point to add or move a mole there and save.")
    print("Ctrl-click on a point to zoom in on it.")
    print("Shift-click on a point to delete it.")
    print("Alt-Shift-click on a point to copy it's uuid.")
    print("Alt-click on a point to paste the copied uuid.")
    print("Press 'o' to toggle follow mode.")
    print("Press 'm' to toggle move mode.")
    print("Press 'c' to copy the moles in the displayed image.")
    print("Press 'a' to auto-paste the copied moles in the displayed image.")
    print("Press space to restore original zoom.")
    print("Press enter to toggle mole markers.")
    print("Press 'q' to quit.")

    is_finished = False
    while not is_finished:
        key = cv2.waitKey(50)
        if key != -1:
            if key == ord('q'):
                is_finished = True
            else:
                controller.on_key(editor, key)

    editor.display.clear_mouse_callback()


def update_follow(editor, follow_uuid, prev_moles, is_paste_mode):
    guess_pos = None
    editor.follow(follow_uuid)

    if mel.rotomap.moles.uuid_mole_index(
            editor.moledata.moles, follow_uuid) is None:

        guessed_moles = guess_mole_positions(
            prev_moles,
            editor.moledata.moles,
            editor.moledata.get_image())

        follow_index = mel.rotomap.moles.uuid_mole_index(
            guessed_moles, follow_uuid)

        if follow_index is not None:
            guess_pos = mel.rotomap.moles.molepos_to_nparray(
                guessed_moles[follow_index])

            guess_pos = (int(guess_pos[0]), int(guess_pos[1]))

            print(guess_pos)
            editor.show_zoomed_display(
                guess_pos[0], guess_pos[1])

            if is_paste_mode:
                editor.add_mole_display(
                    guess_pos[0], guess_pos[1], follow_uuid)

    return guess_pos


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
