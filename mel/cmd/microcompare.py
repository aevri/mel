"""Compare existing microscope images of a mole."""

import os

import mel.lib.common
import mel.lib.datetime
import mel.lib.fullscreenui
import mel.lib.image
import mel.lib.moleimaging


def setup_parser(parser):
    parser.add_argument(
        "PATH", type=str, help="Path to the mole to compare images from."
    )


def get_comparison_images(path):
    micro_path = os.path.join(path, "__micro__")

    # List all the 'jpg' files in the micro dir
    # TODO: support more than just '.jpg'
    names = [x for x in os.listdir(micro_path) if x.lower().endswith(".jpg")]
    names.sort()
    paths = [os.path.join(micro_path, x) for x in names]
    images = [mel.lib.image.load_image(x) for x in paths]

    for i, (path, img) in enumerate(zip(paths, images, strict=False)):
        if img is None:
            raise ValueError(f"Failed to load file: {path}")
        images[i] = mel.lib.image.montage_vertical(
            10, img, mel.lib.image.render_text_as_image(names[i])
        )[:]

    return images


def process_args(args):
    images = get_comparison_images(args.PATH)
    if not images:
        raise Exception(f"No microscope images at {args.PATH}")

    print("Press left arrow or right arrow to change image in the left slot.")
    print("Press space to swap left slot and right slot.")
    print("Press 'q' to quit.")

    # Import pygame as late as possible, to avoid displaying its
    # startup-text where it is not actually used.
    import pygame

    with mel.lib.fullscreenui.fullscreen_context() as screen:
        display = ImageCompareDisplay(screen, args.PATH, images)

        for event in mel.lib.fullscreenui.yield_events_until_quit(screen):
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    display.next_image()
                elif event.key == pygame.K_LEFT:
                    display.prev_image()
                elif event.key == pygame.K_SPACE:
                    display.swap_images()


class ImageCompareDisplay:
    """Display two images in a window, supply controls for comparing a list."""

    def __init__(self, screen, name, image_list):
        if not image_list:
            raise ValueError("image_list must be a list with at least one image.")

        self._display = screen
        self._image_list = image_list
        self._display_list = [image_list[0], image_list[-1]]
        self._index = 0
        self._show_display_list()

    def next_image(self):
        self._index = (self._index + 1) % len(self._image_list)
        self._display_list[0] = self._image_list[self._index]
        self._show_display_list()

    def prev_image(self):
        num_images = len(self._image_list)
        self._index = (self._index + num_images - 1) % len(self._image_list)
        self._display_list[0] = self._image_list[self._index]
        self._show_display_list()

    def swap_images(self):
        self._display_list.reverse()
        self._show_display_list()

    def _show_display_list(self):
        montage = mel.lib.image.montage_horizontal(10, *self._display_list)
        self._display.show_opencv_image(montage)


# -----------------------------------------------------------------------------
# Copyright (C) 2016-2021 Angelos Evripiotis.
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
