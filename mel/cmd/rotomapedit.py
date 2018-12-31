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
    Press '3' for bounding area mode.
    Press '4' for mole marking mode.
    Press '0' for auto-mole debug mode.

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
    Press enter to toggle mole markers.

In 'mask edit' mode:

    Click on a point to draw masking there.
    Shift-click on a point to remove masking there.
    Press '<' to decrease the size of the mask tool.
    Press '>' to increase the size of the mask tool.
    Press '.' to reset the size of the mask tool to the default.

In 'mole marking' mode:

    Click on a point to add or move a mole there and save.
    Shift-click on a point to delete it.
    Press 'a' to accentuate marked moles, for considering removal.
"""


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
        if self.sub_controller:
            try:
                self.sub_controller.pre_key(editor, key)
            except AttributeError:
                pass

    def on_key(self, editor, key):
        if key == ord('o'):
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
        if key == ord('<'):
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


class Controller:
    def __init__(self, editor, follow, copy_to_clipboard):
        self.moleedit_controller = MoleEditController(
            editor, follow, copy_to_clipboard
        )
        self.maskedit_controller = MaskEditController()
        self.molemark_controller = MoleMarkController()
        self.boundingarea_controller = BoundingAreaController()
        self.automoledebug_controller = AutomoleDebugController()
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
        elif key == ord('1'):
            # Switch to mole edit mode
            self.current_controller = self.moleedit_controller
            editor.set_editmole_mode()
        elif key == ord('2'):
            # Switch to mask edit mode
            self.current_controller = self.maskedit_controller
            editor.set_editmask_mode()
        elif key == ord('3'):
            # Switch to bounding area mode
            self.current_controller = self.boundingarea_controller
            editor.set_boundingarea_mode()
        elif key == ord('4'):
            # Switch to mole marking mode
            self.current_controller = self.molemark_controller
            editor.set_molemark_mode()

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


# -----------------------------------------------------------------------------
# Copyright (C) 2015-2018 Angelos Evripiotis.
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
