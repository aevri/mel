"""Organise images into rotomaps."""

import os
import shutil

import mel.lib.common
import mel.lib.fs
import mel.lib.fullscreenui


def setup_parser(parser):
    parser.add_argument(
        "IMAGES", nargs="+", help="A list of paths to images sets or images."
    )


def process_args(args):
    print("Press left arrow or right arrow to change image.")
    print("Press backspace to delete image.")
    print("Press 'g' to group images before current to a folder.")
    print("Press 'q' to quit.")
    print("Ctrl-click on a point to zoom in on it.")
    print("Press space to restore original zoom.")

    # Import pygame as late as possible, to avoid displaying its
    # startup-text where it is not actually used.
    import pygame

    with mel.lib.common.timelogger_context(
        "rotomap-organise"
    ) as logger, mel.lib.fullscreenui.fullscreen_context() as screen:
        display = OrganiserDisplay(
            logger, screen, mel.lib.fs.expand_dirs_to_jpegs(args.IMAGES)
        )

        display.reset_logger()
        for event in mel.lib.fullscreenui.yield_events_until_quit(screen):
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    display.next_image()
                    display.reset_logger()
                elif event.key == pygame.K_LEFT:
                    display.prev_image()
                    display.reset_logger()
                elif event.key == pygame.K_BACKSPACE:
                    display.delete_image()
                elif event.key == pygame.K_g:
                    logger.reset(mode="group")
                    destination = input("group destination: ")
                    display.group_images(destination)
                elif event.key == pygame.K_SPACE:
                    display.set_fitted()
                    display.show()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                key_mods = pygame.key.get_mods()
                if key_mods & pygame.KMOD_CTRL:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    display.show_zoomed(mouse_x, mouse_y)
                    display.show()


class OrganiserDisplay(mel.lib.fullscreenui.LeftRightDisplay):
    """Display images in a window, supply controls for organising."""

    def __init__(self, logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._melroot = mel.lib.fs.find_melroot()
        self._logger = logger

    def reset_logger(self):
        if self.image_path is not None:
            self._logger.reset(
                mode="view",
                path=os.path.relpath(
                    os.path.abspath(self.image_path),
                    start=self._melroot,
                ),
            )

    def delete_image(self):
        if self._image_list:
            os.remove(self._image_list[self._index])
            del self._image_list[self._index]
            self._index -= 1
            self.next_image()

    def group_images(self, destination):
        if self._image_list:
            if not os.path.exists(destination):
                os.makedirs(destination)
            for image_path in self._image_list[: self._index + 1]:
                shutil.move(image_path, destination)
            del self._image_list[: self._index + 1]
            self._index = -1
            self.next_image()
            self.reset_logger()


# -----------------------------------------------------------------------------
# Copyright (C) 2016-2020 Angelos Evripiotis.
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
