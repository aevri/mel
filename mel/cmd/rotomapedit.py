"""Edit a 'rotomap' series of images.

In all modes:

    Press 'q' to quit.
    Press 'Q' to quit with exit code 1.
    Press left for previous image, right for next image.
    Press up for previous map, down for next map.
    Ctrl-click on a point to zoom in on it.
    Press 'z' or 'x' to adjust the zoom level.
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
    Press 'i' to 'rotomap identify' in the current image.
    Press 'c' to to confirm all moles in the current image.
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

import argparse
import contextlib
import os.path

import numpy

import mel.lib.common
import mel.lib.fs
import mel.lib.fullscreenui
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
        "ROTOMAP",
        type=mel.rotomap.moles.make_argparse_rotomap_directory,
        nargs="+",
        help="A list of paths to rotomaps.",
    )
    parser.add_argument(
        "--follow",
        type=str,
        default=None,
        help="UUID of a mole to follow, try to jump to it in the first set.",
    )
    parser.add_argument(
        "--copy-to-clipboard",
        action="store_true",
        help="Copy UUID to the clipboard, as well as printing. Mac OSX only.",
    )
    parser.add_argument(
        "--advance-n-frames",
        "--skip",
        type=int,
        metavar="N",
        default=None,
        help="Start with the image with the specified index, instead of 0.",
    )
    parser.add_argument(
        "--visit-list-file",
        type=argparse.FileType(),
        metavar="PATH",
        help=(
            "Use keys to jump through this list of this form: "
            "'path/to/jpg:hash:optional co-ords'."
        ),
    )


class MoveController:
    def __init__(self):
        self.status = "Move mode"

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
        # Import pygame as late as possible, to avoid displaying its
        # startup-text where it is not actually used.
        import pygame

        arrows = [
            pygame.K_UP,
            pygame.K_DOWN,
            pygame.K_LEFT,
            pygame.K_RIGHT,
        ]
        if key in arrows:
            update_follow(
                editor,
                self.mole_uuid_list[0],
                self._prev_moles,
                self.is_paste_mode,
            )
        elif key == pygame.K_p:
            self.is_paste_mode = not self.is_paste_mode
            self.update_status()
            editor.set_status(self.status)
            editor.show_current()

    def update_status(self):
        if self.is_paste_mode:
            self.status = "follow paste mode"
        else:
            self.status = "follow mode"


class MoleEditController:
    def __init__(self, editor, follow, copy_to_clipboard):
        self.mole_uuid_list = [None]

        self.follow_controller = FollowController(editor, follow, self.mole_uuid_list)
        self.move_controller = MoveController()
        self.sub_controller = None

        self.mouse_x = 0
        self.mouse_y = 0

        self.copy_to_clipboard = copy_to_clipboard

    def on_mouse_event(self, editor, event):
        # Import pygame as late as possible, to avoid displaying its
        # startup-text where it is not actually used.
        import pygame

        self.mouse_x, self.mouse_y = pygame.mouse.get_pos()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.on_lbutton_down(editor, self.mouse_x, self.mouse_y)
            elif event.button == 3:
                self.on_rbutton_down(editor, self.mouse_x, self.mouse_y)

    def on_lbutton_down(self, editor, mouse_x, mouse_y):
        # Import pygame as late as possible, to avoid displaying its
        # startup-text where it is not actually used.
        import pygame

        key_mods = pygame.key.get_mods()
        if key_mods & pygame.KMOD_ALT:
            if key_mods & pygame.KMOD_SHIFT:
                self.mole_uuid_list[0] = editor.get_mole_uuid(mouse_x, mouse_y)
                print(self.mole_uuid_list[0])
                if self.copy_to_clipboard:
                    mel.lib.ui.set_clipboard_contents(self.mole_uuid_list[0])
            else:
                editor.set_mole_uuid(mouse_x, mouse_y, self.mole_uuid_list[0])
        elif key_mods & pygame.KMOD_SHIFT:
            editor.remove_mole(mouse_x, mouse_y)
        else:
            if self.sub_controller and self.sub_controller.on_lbutton_down_noflags(
                editor, mouse_x, mouse_y
            ):
                return
            editor.add_mole(mouse_x, mouse_y)

    def on_rbutton_down(self, editor, mouse_x, mouse_y):
        # Import pygame as late as possible, to avoid displaying its
        # startup-text where it is not actually used.
        import pygame

        key_mods = pygame.key.get_mods()
        if key_mods & pygame.KMOD_ALT:
            if key_mods & pygame.KMOD_SHIFT:
                editor.confirm_mole(mouse_x, mouse_y)
        elif key_mods & pygame.KMOD_SHIFT:
            editor.set_mole_uuid(
                mouse_x,
                mouse_y,
                mel.rotomap.moles.make_new_uuid(),
                is_canonical=False,
            )

    def pre_key(self, editor, key):
        if self.sub_controller:
            with contextlib.suppress(AttributeError):
                self.sub_controller.pre_key(editor, key)

    def on_key(self, editor, key):
        # Import pygame as late as possible, to avoid displaying its
        # startup-text where it is not actually used.
        import pygame

        if key == pygame.K_o:
            is_follow = self.sub_controller is self.follow_controller
            if not is_follow and self.mole_uuid_list[0]:
                self.sub_controller = self.follow_controller
                editor.set_status(self.sub_controller.status)
            else:
                self.sub_controller = None
                editor.set_status("")
            editor.show_current()
        elif key == pygame.K_m:
            if self.sub_controller != self.move_controller:
                self.sub_controller = self.move_controller
                editor.set_status(self.sub_controller.status)
            else:
                self.sub_controller = None
                editor.set_status("")
            editor.show_current()
        elif key == pygame.K_f:
            editor.toggle_faded_markers()
        elif key == pygame.K_RETURN:
            editor.toggle_markers()
        elif key == pygame.K_PLUS:
            self.mole_uuid_list[0] = editor.get_mole_uuid(self.mouse_x, self.mouse_y)
            print(self.mole_uuid_list[0])
            if self.copy_to_clipboard:
                mel.lib.ui.set_clipboard_contents(  # noqa: F823
                    self.mole_uuid_list[0]
                )
        elif key == pygame.K_i:
            # Auto-identify
            #
            # Import mel.rotomap.identifynn as late as possible, because it has
            # some expensive dependencies.
            import mel.rotomap.identifynn

            identifier = mel.rotomap.identifynn.make_identifier()
            target = editor.moledata.current_image_path()
            frame = mel.rotomap.moles.RotomapFrame(os.path.abspath(target))
            new_moles = identifier.get_new_moles(frame)
            mel.rotomap.moles.save_image_moles(new_moles, str(frame.path))
            editor.moledata.reload()
            editor.show_current()
        elif key == pygame.K_c:
            editor.confirm_all()

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

    def on_mouse_event(self, editor, event):
        # Import pygame as late as possible, to avoid displaying its
        # startup-text where it is not actually used.
        import pygame

        key_mods = pygame.key.get_mods()
        enable = not (key_mods & pygame.KMOD_SHIFT)
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
            # is_mouse_button_pressed = pygame.mouse.get_pressed(num_buttons=3)
            is_mouse_button_pressed = pygame.mouse.get_pressed()
            if is_mouse_button_pressed[0]:
                editor.set_mask(mouse_x, mouse_y, enable)

    def pre_key(self, editor, key):
        pass

    def on_key(self, editor, key):
        # Import pygame as late as possible, to avoid displaying its
        # startup-text where it is not actually used.
        import pygame

        key_mods = pygame.key.get_mods()
        shift = key_mods & pygame.KMOD_SHIFT
        if shift:
            if key == pygame.K_COMMA:
                editor.set_smaller_masker()
            elif key == pygame.K_PERIOD:
                editor.set_larger_masker()
        elif key == pygame.K_PERIOD:
            editor.set_default_masker()


class MoleMarkController:
    def __init__(self):
        pass

    def on_mouse_event(self, editor, event):
        # Import pygame as late as possible, to avoid displaying its
        # startup-text where it is not actually used.
        import pygame

        if event.type != pygame.MOUSEBUTTONDOWN:
            return

        key_mods = pygame.key.get_mods()
        mouse_x, mouse_y = pygame.mouse.get_pos()

        if event.button == 1:
            if key_mods & pygame.KMOD_SHIFT:
                if key_mods & pygame.KMOD_ALT:
                    nearest_mole = editor.get_nearest_mole(mouse_x, mouse_y)
                    if nearest_mole is not None:
                        nearest_mole["kind"] = "non-mole"
                        nearest_mole["looks_like"] = "mole"
                        editor.moledata.save_moles()
                        editor.show_current()
                else:
                    editor.remove_mole(mouse_x, mouse_y)
            else:
                if key_mods & pygame.KMOD_ALT:
                    nearest_mole = editor.get_nearest_mole(mouse_x, mouse_y)
                    if nearest_mole is not None:
                        nearest_mole["kind"] = "mole"
                        nearest_mole["looks_like"] = "non-mole"
                        editor.moledata.save_moles()
                        editor.show_current()
                else:
                    editor.add_mole(mouse_x, mouse_y)
        elif event.button == 3:
            nearest_mole = editor.get_nearest_mole(mouse_x, mouse_y)
            if nearest_mole is not None:
                if key_mods & pygame.KMOD_ALT:
                    if key_mods & pygame.KMOD_SHIFT:
                        nearest_mole["kind"] = "non-mole"
                        nearest_mole["looks_like"] = "unsure"
                    else:
                        nearest_mole["kind"] = "mole"
                        nearest_mole["looks_like"] = "unsure"
                else:
                    if key_mods & pygame.KMOD_SHIFT:
                        nearest_mole["kind"] = "non-mole"
                        nearest_mole["looks_like"] = "non-mole"
                    else:
                        nearest_mole["kind"] = "mole"
                        nearest_mole["looks_like"] = "mole"
                editor.moledata.save_moles()
                editor.show_current()

    def pre_key(self, editor, key):
        pass

    def on_key(self, editor, key):
        # Import pygame as late as possible, to avoid displaying its
        # startup-text where it is not actually used.
        import pygame

        if key == pygame.K_a:
            is_alt = editor.marked_mole_overlay.is_accentuate_marked_mode
            editor.marked_mole_overlay.is_accentuate_marked_mode = not is_alt
            editor.show_current()


class BoundingAreaController:
    def __init__(self):
        pass

    def on_mouse_event(self, editor, event):
        pass

    def pre_key(self, editor, key):
        pass

    def on_key(self, editor, key):
        pass


class AutomoleDebugController:
    def __init__(self):
        pass

    def on_mouse_event(self, editor, event):
        pass

    def pre_key(self, editor, key):
        pass

    def on_key(self, editor, key):
        pass


class VisitList:
    def __init__(self, items):
        self._items = items
        self._index = 0

    def back(self):
        self._index = (self._index + len(self._items) - 1) % len(self._items)
        return self.current()

    def forward(self):
        self._index = (self._index + 1) % len(self._items)
        return self.current()

    def current(self):
        return self._items[self._index]

    def __bool__(self):
        return bool(self._items)


class Controller:
    def __init__(self, editor, follow, copy_to_clipboard, visit_list, logger):
        self._visit_list = VisitList(visit_list)

        self._logger = logger
        self._melroot = mel.lib.fs.find_melroot()
        self.moleedit_controller = MoleEditController(editor, follow, copy_to_clipboard)
        self.maskedit_controller = MaskEditController()
        self.molemark_controller = MoleMarkController()
        self.boundingarea_controller = BoundingAreaController()
        self.automoledebug_controller = AutomoleDebugController()
        self.current_controller = self.moleedit_controller
        logger.reset(mode="editmole")

        self.zooms = [1.0, 0.75, 0.5, 0.25, 2.0, 1.75, 1.5]
        self.zoom_index = 0

        self._reset_logger_new_image(editor)

    def on_mouse_event(self, editor, event):
        # Import pygame as late as possible, to avoid displaying its
        # startup-text where it is not actually used.
        import pygame

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            key_mods = pygame.key.get_mods()
            if key_mods & pygame.KMOD_CTRL:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                editor.show_zoomed(mouse_x, mouse_y)
                return

        self.current_controller.on_mouse_event(editor, event)

    def _reset_logger_new_image(self, editor):
        self._logger.reset(
            path=os.path.relpath(
                os.path.abspath(editor.moledata.image_path),
                start=self._melroot,
            )
        )

    def on_key(self, editor, key):
        # Import pygame as late as possible, to avoid displaying its
        # startup-text where it is not actually used.
        import pygame

        self.current_controller.pre_key(editor, key)

        if key == pygame.K_LEFT:
            editor.show_prev()
            self._reset_logger_new_image(editor)
        elif key == pygame.K_RIGHT:
            editor.show_next()
            self._reset_logger_new_image(editor)
        elif key == pygame.K_UP:
            editor.show_prev_map()
            self._reset_logger_new_image(editor)
        elif key == pygame.K_DOWN:
            editor.show_next_map()
            self._reset_logger_new_image(editor)
        elif key == pygame.K_SPACE:
            editor.show_fitted()
        elif key == pygame.K_0:
            # Switch to automole debug mode
            self.current_controller = self.automoledebug_controller
            editor.set_automoledebug_mode()
            self._logger.reset(mode="debug")
        elif key == pygame.K_1:
            # Switch to mole edit mode
            self.current_controller = self.moleedit_controller
            editor.set_editmole_mode()
            self._logger.reset(mode="editmole")
        elif key == pygame.K_2:
            # Switch to mask edit mode
            self.current_controller = self.maskedit_controller
            editor.set_editmask_mode()
            self._logger.reset(mode="editmask")
        elif key == pygame.K_3:
            # Switch to bounding area mode
            self.current_controller = self.boundingarea_controller
            editor.set_boundingarea_mode()
            self._logger.reset(mode="boundingarea")
        elif key == pygame.K_4:
            # Switch to mole marking mode
            self.current_controller = self.molemark_controller
            self._logger.reset(mode="molemark")
            editor.set_molemark_mode()
        elif key == pygame.K_b:
            # Go back in the visit list
            if self._visit_list:
                to_visit = self._visit_list.back()
                editor.visit(to_visit)
        elif key == pygame.K_n:
            # Go to the next in the visit list
            if self._visit_list:
                to_visit = self._visit_list.forward()
                editor.visit(to_visit)
        elif key == pygame.K_z:
            self.zoom_index += 1
            self.zoom_index %= len(self.zooms)
            zoom = self.zooms[self.zoom_index]
            editor.set_status(f"Zoom {zoom}")
            editor.set_zoom_level(zoom)
        elif key == pygame.K_x:
            self.zoom_index += len(self.zooms) - 1
            self.zoom_index %= len(self.zooms)
            zoom = self.zooms[self.zoom_index]
            editor.set_status(f"Zoom {zoom}")
            editor.set_zoom_level(zoom)

        self.current_controller.on_key(editor, key)


def process_args(args):
    # Import pygame as late as possible, to avoid displaying its
    # startup-text where it is not actually used.
    import pygame

    visit_list = []
    if args.visit_list_file:
        visit_list = args.visit_list_file.read().splitlines()

    with (
        mel.lib.common.timelogger_context("rotomap-edit") as logger,
        mel.lib.fullscreenui.fullscreen_context() as screen,
    ):
        editor = mel.rotomap.display.Editor(args.ROTOMAP, screen)

        if args.advance_n_frames:
            editor.show_next_n(args.advance_n_frames)

        controller = Controller(
            editor, args.follow, args.copy_to_clipboard, visit_list, logger
        )

        for event in mel.lib.fullscreenui.yield_events_until_quit(screen):
            if event.type == pygame.KEYDOWN:
                controller.on_key(editor, event.key)
            elif event.type in (
                pygame.MOUSEBUTTONDOWN,
                pygame.MOUSEMOTION,
            ):
                controller.on_mouse_event(editor, event)


def update_follow(editor, follow_uuid, prev_moles, is_paste_mode):
    guess_pos = None
    editor.follow(follow_uuid)

    if mel.rotomap.moles.uuid_mole_index(editor.moledata.moles, follow_uuid) is None:
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
                editor.add_mole_display(guess_pos[0], guess_pos[1], follow_uuid)

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
