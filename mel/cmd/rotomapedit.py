"""Edit a 'rotomap' series of images.

In all modes:

    Press 'q' to quit.
    Press 'Q' to quit with exit code 1.
    Press left for previous image, right for next image.
    Press up for previous map, down for next map.
    Ctrl-click on a point to zoom in on it.
    Press space to restore original zoom.

Mode selection:

    Press '1' for mole edit mode (the starting mode).
    Press '2' for mask edit mode.
    Press '3' for mole marking mode.
    Press '4' for image relating mode.
    Press '5' for bounding area mode.
    Press '0' for auto-mole debug mode.
    Press '9' for auto-relate debug mode.

In 'mole edit' mode:

    Click on a point to add or move a mole there and save.
    Shift-click on a point to delete it.
    Shift-right-click on a point to randomize the uuid.
    Alt-Shift-click on a point to copy it's uuid.
    Also, press 'end' or '+' when over a point to copy it's uuid.
    Alt-Shift-right-click over a point to make it canonical.
    Alt-click on a point to paste the copied uuid.
    Press 'o' to toggle follow mode.
    Press 'm' to toggle move mode.
    Press 'c' to copy the moles in the displayed image.
    Press 'a' to auto-paste the copied moles in the displayed image.
    Press 'r' to auto-mark moles visible in the current mask.
    Press 't' to auto-relate moles from the previously viewed image.
    Press enter to toggle mole markers.

In 'mask edit' mode:

    Click on a point to draw masking there.
    Shift-click on a point to remove masking there.
    Press 'a' to auto-mask based on the current mask.
    Press '<' to decrease the size of the mask tool.
    Press '>' to increase the size of the mask tool.
    Press '.' to reset the size of the mask tool to the default.

In 'mole marking' mode:

    Click on a point to add or move a mole there and save.
    Shift-click on a point to delete it.
    Press 'a' to accentuate marked moles, for considering removal.

In 'image relating' mode:

    Right-Click on a mole to copy its UUID.
    Click on a non-faded point to paste the UUID.
    Alt-click on a non-faded point to paste the UUID globally.
    Shift-click on a non-faded point to randomize the uuid.
    Press 'a' to apply auto-relate results to image.
    Press 'g' to apply auto-relate results globally to rotomap.
    Press 't' to toggle target mode, which emphasises mole images.
"""


import copy

import cv2
import numpy

import mel.lib.common
import mel.lib.image
import mel.lib.math
import mel.lib.ui

import mel.rotomap.display
import mel.rotomap.mask
import mel.rotomap.moles
import mel.rotomap.relate


# Radius within which we should look for moles, in later work perhaps we'll
# make this configurable by the user.
_MAGIC_MOLE_FINDER_RADIUS = 50


def setup_parser(parser):
    parser.add_argument(
        'ROTOMAP',
        type=mel.rotomap.moles.make_argparse_rotomap_directory,
        nargs='+',
        help="A list of paths to rotomaps.",
    )
    parser.add_argument(
        '--display-width',
        type=int,
        default=None,
        help="Width of the preview display window.",
    )
    parser.add_argument(
        '--display-height',
        type=int,
        default=None,
        help="Width of the preview display window.",
    )
    parser.add_argument(
        '--follow',
        type=str,
        default=None,
        help="UUID of a mole to follow, try to jump to it in the first set.",
    )
    parser.add_argument(
        '--copy-to-clipboard',
        action='store_true',
        help='Copy UUID to the clipboard, as well as printing. Mac OSX only.',
    )
    parser.add_argument(
        '--advance-n-frames',
        '--skip',
        type=int,
        metavar='N',
        default=None,
        help="Start with the image with the specified index, instead of 0.",
    )


class MoveController:
    def __init__(self):
        self.status = 'Move mode'

    def on_lbutton_down_noflags(self, editor, mouse_x, mouse_y):
        editor.move_nearest_mole(mouse_x, mouse_y)
        return True

    def on_key(self, editor, key):
        pass


class FollowController:
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
                self.is_paste_mode,
            )
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


class MoleEditController:
    def __init__(self, editor, follow, copy_to_clipboard):
        self.mole_uuid_list = [None]

        self.follow_controller = FollowController(
            editor, follow, self.mole_uuid_list
        )
        self.move_controller = MoveController()
        self.sub_controller = None

        self.copied_moles = None
        self.previous_moles = None

        self.mouse_x = 0
        self.mouse_y = 0

        self.copy_to_clipboard = copy_to_clipboard

    def on_mouse_event(self, editor, event, mouse_x, mouse_y, flags, param):
        self.mouse_x = mouse_x
        self.mouse_y = mouse_y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.on_lbutton_down(editor, mouse_x, mouse_y, flags)
        if event == cv2.EVENT_RBUTTONDOWN:
            self.on_rbutton_down(editor, mouse_x, mouse_y, flags)

    def on_lbutton_down(self, editor, mouse_x, mouse_y, flags):
        if flags & cv2.EVENT_FLAG_ALTKEY:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.mole_uuid_list[0] = editor.get_mole_uuid(mouse_x, mouse_y)
                print(self.mole_uuid_list[0])
                if self.copy_to_clipboard:
                    mel.lib.ui.set_clipboard_contents(self.mole_uuid_list[0])
            else:
                editor.set_mole_uuid(mouse_x, mouse_y, self.mole_uuid_list[0])
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            editor.remove_mole(mouse_x, mouse_y)
        else:
            if self.sub_controller:
                if self.sub_controller.on_lbutton_down_noflags(
                    editor, mouse_x, mouse_y
                ):
                    return
            editor.add_mole(mouse_x, mouse_y)

    def on_rbutton_down(self, editor, mouse_x, mouse_y, flags):
        if flags & cv2.EVENT_FLAG_ALTKEY:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                editor.confirm_mole(mouse_x, mouse_y)
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            editor.set_mole_uuid(
                mouse_x,
                mouse_y,
                mel.rotomap.moles.make_new_uuid(),
                is_canonical=False,
            )

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
                editor.moledata.get_image(),
            )
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
        elif key == ord('+'):
            self.mole_uuid_list[0] = editor.get_mole_uuid(
                self.mouse_x, self.mouse_y
            )
            print(self.mole_uuid_list[0])
            if self.copy_to_clipboard:
                mel.lib.ui.set_clipboard_contents(self.mole_uuid_list[0])

        if self.sub_controller:
            try:
                sub_controller_onkey = self.sub_controller.on_key
            except AttributeError:
                pass
            else:
                sub_controller_onkey(editor, key)


class MaskEditController:
    def __init__(self):
        pass

    def on_mouse_event(self, editor, event, mouse_x, mouse_y, flags, param):
        enable = not (flags & cv2.EVENT_FLAG_SHIFTKEY)
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
        elif key == ord('<'):
            editor.set_smaller_masker()
        elif key == ord('>'):
            editor.set_larger_masker()
        elif key == ord('.'):
            editor.set_default_masker()


class MoleMarkController:
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


class ImageRelateController:
    def __init__(self):
        self.copied_uuid = None

    def on_mouse_event(self, editor, event, mouse_x, mouse_y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                editor.set_mole_uuid(
                    mouse_x,
                    mouse_y,
                    mel.rotomap.moles.make_new_uuid(),
                    is_canonical=False,
                )
            else:
                if self.copied_uuid:
                    if flags & cv2.EVENT_FLAG_ALTKEY:
                        editor.remap_uuid(
                            editor.get_mole_uuid(mouse_x, mouse_y),
                            self.copied_uuid,
                        )
                    else:
                        editor.set_mole_uuid(
                            mouse_x, mouse_y, self.copied_uuid)
        elif event == cv2.EVENT_RBUTTONDOWN:
            image_pos = editor.display.windowxy_to_imagexy(mouse_x, mouse_y)
            nearest = mel.rotomap.moles.nearest_mole_index_distance
            from_index = from_distance = None
            if editor.from_moles is not None:
                from_index, from_distance = nearest(
                    editor.from_moles, *image_pos
                )

            to_index, to_distance = nearest(editor.moledata.moles, *image_pos)

            if from_distance is None:
                if to_index is not None:
                    self.copied_uuid = editor.moledata.moles[to_index]['uuid']
            elif to_distance is None:
                if from_index is not None:
                    self.copied_uuid = editor.from_moles[from_index]['uuid']
            else:
                if from_distance < to_distance:
                    self.copied_uuid = editor.from_moles[from_index]['uuid']
                else:
                    self.copied_uuid = editor.moledata.moles[to_index]['uuid']

    def pre_key(self, editor, key):
        if key in mel.lib.ui.WAITKEY_ARROWS:
            editor.set_from_moles(copy.deepcopy(editor.moledata.moles))

    def on_key(self, editor, key):
        if key == ord('a'):
            if editor.from_moles is not None:
                theory = mel.rotomap.relate.best_theory(
                    editor.from_moles, editor.moledata.moles, iterate=False
                )

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

        elif key == ord('g'):
            if editor.from_moles is not None:
                theory = mel.rotomap.relate.best_theory(
                    editor.from_moles, editor.moledata.moles, iterate=False
                )

                if theory:
                    for from_uuid, to_uuid in theory:
                        if from_uuid != to_uuid and from_uuid and to_uuid:
                            editor.remap_uuid(to_uuid, from_uuid)

                editor.moledata.save_moles()

        elif key == ord('t'):
            is_alt = editor.image_relate_overlay.is_target_mode
            editor.image_relate_overlay.is_target_mode = not is_alt
            editor.show_current()


class BoundingAreaController:
    def __init__(self):
        pass

    def on_mouse_event(self, editor, event, mouse_x, mouse_y, flags, param):
        pass

    def pre_key(self, editor, key):
        pass

    def on_key(self, editor, key):
        pass


class AutomoleDebugController:
    def __init__(self):
        pass

    def on_mouse_event(self, editor, event, mouse_x, mouse_y, flags, param):
        pass

    def pre_key(self, editor, key):
        pass

    def on_key(self, editor, key):
        pass


class AutoRelateDebugController:
    def __init__(self):
        pass

    def on_mouse_event(self, editor, event, mouse_x, mouse_y, flags, param):
        pass

    def pre_key(self, editor, key):
        if key in mel.lib.ui.WAITKEY_ARROWS:
            editor.set_from_moles(copy.deepcopy(editor.moledata.moles))

    def on_key(self, editor, key):
        pass


class Controller:
    def __init__(self, editor, follow, copy_to_clipboard):
        self.moleedit_controller = MoleEditController(
            editor, follow, copy_to_clipboard
        )
        self.maskedit_controller = MaskEditController()
        self.molemark_controller = MoleMarkController()
        self.imagerelate_controller = ImageRelateController()
        self.boundingarea_controller = BoundingAreaController()
        self.automoledebug_controller = AutomoleDebugController()
        self.autorelatedebug_controller = AutoRelateDebugController()
        self.current_controller = self.moleedit_controller

    def on_mouse_event(self, editor, event, mouse_x, mouse_y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                editor.show_zoomed(mouse_x, mouse_y)
                return

        self.current_controller.on_mouse_event(
            editor, event, mouse_x, mouse_y, flags, param
        )

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
        elif key == ord('4'):
            # Switch to image relating mode
            self.current_controller = self.imagerelate_controller
            editor.set_imagerelate_mode()
        elif key == ord('5'):
            # Switch to image relating mode
            self.current_controller = self.boundingarea_controller
            editor.set_boundingarea_mode()

        self.current_controller.on_key(editor, key)


def process_args(args):

    editor = mel.rotomap.display.Editor(
        args.ROTOMAP, args.display_width, args.display_height
    )

    mel.lib.ui.bring_python_to_front()

    if args.advance_n_frames:
        editor.show_next_n(args.advance_n_frames)

    controller = Controller(editor, args.follow, args.copy_to_clipboard)

    def mouse_callback(*args):
        controller.on_mouse_event(editor, *args)

    editor.display.set_mouse_callback(mouse_callback)

    try:
        for key in mel.lib.ui.yield_keys_until_quitkey(error_key='Q'):
            controller.on_key(editor, key)
    except mel.lib.ui.AbortKeyInterruptError:
        return 1
    finally:
        editor.display.clear_mouse_callback()


def update_follow(editor, follow_uuid, prev_moles, is_paste_mode):
    guess_pos = None
    editor.follow(follow_uuid)

    if (
        mel.rotomap.moles.uuid_mole_index(editor.moledata.moles, follow_uuid)
        is None
    ):

        guess_pos = mel.rotomap.relate.guess_mole_pos(
            follow_uuid, prev_moles, editor.moledata.moles
        )

        if guess_pos is not None:
            ellipse = mel.lib.moleimaging.find_mole_ellipse(
                editor.moledata.get_image().copy(),
                guess_pos,
                _MAGIC_MOLE_FINDER_RADIUS,
            )

            if ellipse is not None:
                guess_pos = numpy.array(ellipse[0], dtype=int)

            editor.show_zoomed_display(guess_pos[0], guess_pos[1])

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
                pos, prev_moles_for_mapping, image_rect
            )

            if moles_for_mapping:
                pos = mel.rotomap.moles.mapped_pos(
                    pos, moles_for_mapping, current_moles
                )
                mel.rotomap.moles.set_molepos_to_nparray(new_m, pos)

            ellipse = mel.lib.moleimaging.find_mole_ellipse(
                current_image, pos, _MAGIC_MOLE_FINDER_RADIUS
            )
            if ellipse is not None:
                mel.rotomap.moles.set_molepos_to_nparray(new_m, ellipse[0])

            new_moles.append(new_m)

    return new_moles


# -----------------------------------------------------------------------------
# Copyright (C) 2015-2017 Angelos Evripiotis.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------ END-OF-FILE ----------------------------------
