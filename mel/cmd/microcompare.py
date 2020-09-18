"""Compare existing microscope images of a mole."""

import os

import mel.lib.common
import mel.lib.datetime
import mel.lib.image
import mel.lib.moleimaging
import mel.lib.ui


def setup_parser(parser):
    parser.add_argument(
        "PATH", type=str, help="Path to the mole to compare images from."
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=None,
        help="Width of the preview display window.",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=None,
        help="Width of the preview display window.",
    )


def get_comparison_images(path):

    micro_path = os.path.join(path, "__micro__")

    # List all the 'jpg' files in the micro dir
    # TODO: support more than just '.jpg'
    names = [x for x in os.listdir(micro_path) if x.lower().endswith(".jpg")]
    names.sort()
    paths = [os.path.join(micro_path, x) for x in names]
    images = [mel.lib.image.load_image(x) for x in paths]

    for i, (path, img) in enumerate(zip(paths, images)):
        if img is None:
            raise ValueError("Failed to load file: {}".format(path))
        else:
            images[i] = mel.lib.image.montage_vertical(
                10, img, mel.lib.image.render_text_as_image(names[i])
            )[:]

    return images


def process_args(args):
    images = get_comparison_images(args.PATH)
    if not images:
        raise Exception("No microscope images at {}".format(args.PATH))

    display = ImageCompareDisplay(
        args.PATH, images, args.display_width, args.display_height
    )

    mel.lib.ui.bring_python_to_front()

    print("Press left arrow or right arrow to change image in the left slot.")
    print("Press space to swap left slot and right slot.")
    print("Press 'q' to quit.")

    for key in mel.lib.ui.yield_keys_until_quitkey():
        if key == mel.lib.ui.WAITKEY_RIGHT_ARROW:
            display.next_image()
        elif key == mel.lib.ui.WAITKEY_LEFT_ARROW:
            display.prev_image()
        elif key == ord(" "):
            display.swap_images()


class ImageCompareDisplay:
    """Display two images in a window, supply controls for comparing a list."""

    def __init__(self, name, image_list, width=None, height=None):
        if not image_list:
            raise ValueError(
                "image_list must be a list with at least one image."
            )

        self._display = mel.lib.ui.ImageDisplay(name, width, height)
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
        self._display.show_image(montage)


# -----------------------------------------------------------------------------
# Copyright (C) 2016-2018 Angelos Evripiotis.
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
