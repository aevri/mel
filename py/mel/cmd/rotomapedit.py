"""Edit a 'rotomap' series of images."""


import copy
import uuid

import cv2

import mel.lib.common
import mel.lib.image
import mel.lib.math
import mel.lib.ui

import mel.rotomap.display
import mel.rotomap.mask


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

    def on_lbutton_down_noflags(self, editor, mouse_x, mouse_y):
        editor.move_nearest_mole(mouse_x, mouse_y)
        return True

    def on_key(self, editor, key):
        pass


class FollowController():

    def __init__(self, editor, follow, mole_uuid_list):
        self.mole_uuid_list = mole_uuid_list
        if follow:
            self.mole_uuid_list[0] = follow
            editor.skip_to_mole(self.mole_uuid_list[0])
            editor.follow(self.mole_uuid_list[0])

        self.is_paste_mode = False
        self.update_status()

    def on_lbutton_down_noflags(self, editor, mouse_x, mouse_y):
        editor.crud_mole(self.mole_uuid_list[0], mouse_x, mouse_y)
        editor.follow(self.mole_uuid_list[0])
        return True

    def pre_key(self, editor, key):
        self._prev_moles = editor.moledata.moles

    def on_key(self, editor, key):
        if key in mel.lib.ui.WAITKEY_ARROWS:
            update_follow(
                editor,
                self.mole_uuid_list[0],
                self._prev_moles,
                self.is_paste_mode)
        elif key == ord('p'):
            self.is_paste_mode = not self.is_paste_mode
            self.update_status()
            editor.set_status(self.status)
            editor.show_current()

    def update_status(self):
        if self.is_paste_mode:
            self.status = 'follow paste mode'
        else:
            self.status = 'follow mode'


class MoleEditController():

    def __init__(self, editor, follow):
        self.mole_uuid_list = [None]

        self.follow_controller = FollowController(
            editor, follow, self.mole_uuid_list)
        self.move_controller = MoveController()
        self.sub_controller = None

        self.copied_moles = None
        self.previous_moles = None

    def on_mouse_event(self, editor, event, mouse_x, mouse_y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.on_lbutton_down(editor, mouse_x, mouse_y, flags)
        if event == cv2.EVENT_RBUTTONDOWN:
            self.on_rbutton_down(editor, mouse_x, mouse_y, flags)

    def on_lbutton_down(self, editor, mouse_x, mouse_y, flags):
        if flags & cv2.EVENT_FLAG_ALTKEY:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.mole_uuid_list[0] = editor.get_mole_uuid(mouse_x, mouse_y)
                print(self.mole_uuid_list[0])
            else:
                editor.set_mole_uuid(mouse_x, mouse_y, self.mole_uuid_list[0])
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            editor.remove_mole(mouse_x, mouse_y)
        else:
            if self.sub_controller:
                if self.sub_controller.on_lbutton_down_noflags(
                        editor, mouse_x, mouse_y):
                    return
            editor.add_mole(mouse_x, mouse_y)

    def on_rbutton_down(self, editor, mouse_x, mouse_y, flags):
        if flags & cv2.EVENT_FLAG_ALTKEY:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                pass
            else:
                editor.remap_uuid(
                    editor.get_mole_uuid(mouse_x, mouse_y),
                    self.mole_uuid_list[0])
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            editor.set_mole_uuid(
                mouse_x,
                mouse_y,
                uuid.uuid4().hex,
                is_canonical=False)

    def pre_key(self, editor, key):
        if key in mel.lib.ui.WAITKEY_ARROWS:
            self.previous_moles = copy.deepcopy(editor.moledata.moles)

        if self.sub_controller:
            try:
                self.sub_controller.pre_key(editor, key)
            except AttributeError:
                pass

    def on_key(self, editor, key):
        if key == ord('c'):
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
        elif key == ord('r'):
            guessed_moles = mel.rotomap.detectmoles.moles(
                editor.moledata.get_image(),
                editor.moledata.mask)
            editor.set_moles(guessed_moles)
            editor.moledata.save_moles()
        elif key == ord('t'):
            theory = mel.rotomap.relate.best_offset_theory(
                self.previous_moles,
                editor.moledata.moles)
            if theory:
                guessed_moles = copy.deepcopy(editor.moledata.moles)
                for mole in guessed_moles:
                    for p in theory:
                        if p[0] and p[1]:
                            if mole['uuid'] == p[1]:
                                mole['uuid'] = p[0]
                                break
                editor.set_moles(guessed_moles)
                editor.moledata.save_moles()
        elif key == ord('f'):
            editor.toggle_faded_markers()
        elif key == 13:
            editor.toggle_markers()

        if self.sub_controller:
            try:
                self.sub_controller.on_key(editor, key)
            except AttributeError:
                pass


class MaskEditController():

    def __init__(self):
        pass

    def on_mouse_event(self, editor, event, mouse_x, mouse_y, flags, param):
        enable = not(flags & cv2.EVENT_FLAG_SHIFTKEY)
        if event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                editor.set_mask(mouse_x, mouse_y, enable)
        elif event == cv2.EVENT_LBUTTONDOWN:
            editor.set_mask(mouse_x, mouse_y, enable)

    def pre_key(self, editor, key):
        pass

    def on_key(self, editor, key):
        if key == ord('a'):
            image = editor.moledata.image
            mask = editor.moledata.mask
            hist = mel.rotomap.mask.histogram_from_image_mask(image, mask)
            editor.moledata.mask = mel.rotomap.mask.guess_mask(image, hist)
            editor.moledata.save_mask()
            editor.show_current()


class MoleMarkController():

    def __init__(self):
        pass

    def on_mouse_event(self, editor, event, mouse_x, mouse_y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                editor.remove_mole(mouse_x, mouse_y)
            else:
                editor.add_mole(mouse_x, mouse_y)

    def pre_key(self, editor, key):
        pass

    def on_key(self, editor, key):
        if key == ord('a'):
            is_alt = editor.marked_mole_overlay.is_accentuate_marked_mode
            editor.marked_mole_overlay.is_accentuate_marked_mode = not is_alt
            editor.show_current()


class AutomoleDebugController():

    def __init__(self):
        pass

    def on_mouse_event(self, editor, event, mouse_x, mouse_y, flags, param):
        pass

    def pre_key(self, editor, key):
        pass

    def on_key(self, editor, key):
        pass


class AutoRelateDebugController():

    def __init__(self):
        pass

    def on_mouse_event(self, editor, event, mouse_x, mouse_y, flags, param):
        pass

    def pre_key(self, editor, key):
        if key in mel.lib.ui.WAITKEY_ARROWS:
            editor.set_from_moles(copy.deepcopy(editor.moledata.moles))

    def on_key(self, editor, key):
        pass


class Controller():

    def __init__(self, editor, follow):
        self.moleedit_controller = MoleEditController(editor, follow)
        self.maskedit_controller = MaskEditController()
        self.molemark_controller = MoleMarkController()
        self.automoledebug_controller = AutomoleDebugController()
        self.autorelatedebug_controller = AutoRelateDebugController()
        self.current_controller = self.moleedit_controller

    def on_mouse_event(self, editor, event, mouse_x, mouse_y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                editor.show_zoomed(mouse_x, mouse_y)
                return

        self.current_controller.on_mouse_event(
            editor, event, mouse_x, mouse_y, flags, param)

    def on_key(self, editor, key):
        self.current_controller.pre_key(editor, key)

        if key == mel.lib.ui.WAITKEY_LEFT_ARROW:
            editor.show_prev()
        elif key == mel.lib.ui.WAITKEY_RIGHT_ARROW:
            editor.show_next()
        elif key == mel.lib.ui.WAITKEY_UP_ARROW:
            editor.show_prev_map()
        elif key == mel.lib.ui.WAITKEY_DOWN_ARROW:
            editor.show_next_map()
        elif key == ord(' '):
            editor.show_fitted()
        elif key == ord('0'):
            # Switch to automole debug mode
            self.current_controller = self.automoledebug_controller
            editor.set_automoledebug_mode()
        elif key == ord('9'):
            # Switch to autorelate debug mode
            self.current_controller = self.autorelatedebug_controller
            editor.set_autorelatedebug_mode()
        elif key == ord('1'):
            # Switch to mole edit mode
            self.current_controller = self.moleedit_controller
            editor.set_editmole_mode()
        elif key == ord('2'):
            # Switch to mask edit mode
            self.current_controller = self.maskedit_controller
            editor.set_editmask_mode()
        elif key == ord('3'):
            # Switch to mole marking mode
            self.current_controller = self.molemark_controller
            editor.set_molemark_mode()

        if key in mel.lib.ui.WAITKEY_ARROWS:
            print(editor.moledata.current_image_path())

        self.current_controller.on_key(editor, key)


def process_args(args):

    editor = mel.rotomap.display.Editor(
        args.images, args.display_width, args.display_height)

    mel.lib.ui.bring_python_to_front()

    controller = Controller(editor, args.follow)

    def mouse_callback(*args):
        controller.on_mouse_event(editor, *args)

    editor.display.set_mouse_callback(
        mouse_callback)

    print("Press 'q' to quit.")
    print("Press left for previous image, right for next image.")
    print("Press up for previous map, down for next map.")
    print("Ctrl-click on a point to zoom in on it.")
    print("Press space to restore original zoom.")
    print()
    print("Press '1' for mole edit mode (the starting mode).")
    print("Press '2' for mask edit mode.")
    print("Press '3' for mole marking mode.")
    print("Press '0' for auto-mole debug mode.")
    print("Press '9' for auto-relate debug mode.")
    print()
    print("In 'mole edit' mode:")
    print("Click on a point to add or move a mole there and save.")
    print("Shift-click on a point to delete it.")
    print("Shift-right-click on a point to randomize the uuid.")
    print("Alt-Shift-click on a point to copy it's uuid.")
    print("Alt-click on a point to paste the copied uuid.")
    print("Alt-right-click on a point to replace the uuid in the whole map.")
    print("Press 'o' to toggle follow mode.")
    print("Press 'm' to toggle move mode.")
    print("Press 'c' to copy the moles in the displayed image.")
    print("Press 'a' to auto-paste the copied moles in the displayed image.")
    print("Press 'r' to auto-mark moles visible in the current mask.")
    print("Press 't' to auto-relate moles from the previously viewed image.")
    print("Press enter to toggle mole markers.")
    print()
    print("In 'mask edit' mode:")
    print("Click on a point to draw masking there.")
    print("Shift-click on a point to remove masking there.")
    print("Press 'a' to auto-mask based on the current mask.")
    print()
    print("In 'mole marking' mode:")
    print("Click on a point to add or move a mole there and save.")
    print("Shift-click on a point to delete it.")
    print("Press 'a' to accentuate marked moles, for considering removal.")

    for key in mel.lib.ui.yield_keys_until_quitkey():
        controller.on_key(editor, key)

    editor.display.clear_mouse_callback()


def update_follow(editor, follow_uuid, prev_moles, is_paste_mode):
    guess_pos = None
    editor.follow(follow_uuid)

    if mel.rotomap.moles.uuid_mole_index(
            editor.moledata.moles, follow_uuid) is None:

        guess_pos = mel.rotomap.relate.guess_mole_pos(
            follow_uuid,
            prev_moles,
            editor.moledata.moles)

        if guess_pos is not None:
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
            pos = mel.rotomap.moles.mole_to_point(new_m)

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
