"""Organise images into rotomaps."""

import os
import shutil


import mel.lib.common
import mel.lib.fs
import mel.lib.fullscreenui
import mel.lib.ui


def setup_parser(parser):
    parser.add_argument(
        "IMAGES", nargs="+", help="A list of paths to images sets or images."
    )


def process_args(args):

    print("Press left arrow or right arrow to change image.")
    print("Press backspace to delete image.")
    print("Press 'g' to group images before current to a folder.")
    print("Press 'q' to quit.")

    # Import pygame as late as possible, to avoid displaying its
    # startup-text where it is not actually used.
    import pygame

    with mel.lib.fullscreenui.fullscreen_context() as surface:
        display = OrganiserDisplay(
            surface, mel.lib.fs.expand_dirs_to_jpegs(args.IMAGES)
        )

        for event in mel.lib.fullscreenui.yield_events_until_quit():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    display.next_image()
                elif event.key == pygame.K_LEFT:
                    display.prev_image()
                elif event.key == pygame.K_BACKSPACE:
                    display.delete_image()
                elif event.key == pygame.K_g:
                    destination = input("group destination: ")
                    display.group_images(destination)


class OrganiserDisplay(mel.lib.fullscreenui.LeftRightDisplay):
    """Display images in a window, supply controls for organising."""

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
