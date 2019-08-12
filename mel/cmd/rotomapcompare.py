"""Compare marked moles across rotomaps.

Controls:

    'left arrow' or 'right arrow' to change image in the left slot.
    'up arrow' or 'down arrow' to change rotomap in the left slot.
    'space' to swap left slot and right slot.
    'p' or 'n' to change the mole being examined in both slots.
    'a' to toggle crosshairs on/off.

    'q' to quit.
"""

import collections
import functools

import cv2
import numpy

import mel.lib.common
import mel.lib.datetime
import mel.lib.image
import mel.lib.moleimaging
import mel.lib.ui

import mel.rotomap.display
import mel.rotomap.moles


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


def process_args(args):

    uuid_to_rotomaps_imagepos_list = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )

    for rotomap in args.ROTOMAP:
        for frame in rotomap.yield_frames():
            for uuid_, point in frame.moledata.uuid_points.items():
                uuid_to_rotomaps_imagepos_list[uuid_][rotomap.path].append(
                    (frame.path, point)
                )

    # We can't compare moles that are only in one rotomap, cull these.
    uuid_to_rotomaps_imagepos_list = {
        key: value
        for key, value in uuid_to_rotomaps_imagepos_list.items()
        if len(value) > 1
    }

    if not uuid_to_rotomaps_imagepos_list:
        raise Exception("Nothing to compare.")

    # Ensure we're not using a defaultdict, otherwise we might miss a KeyError.
    uuid_to_rotomaps_imagepos_list = dict(uuid_to_rotomaps_imagepos_list)

    uuid_order = tuple(sorted(iter(uuid_to_rotomaps_imagepos_list)))

    index = 0
    uuid_ = uuid_order[index]
    path_images_tuple = tuple(uuid_to_rotomaps_imagepos_list[uuid_].values())
    display = ImageCompareDisplay(
        '.', path_images_tuple, args.display_width, args.display_height
    )

    mel.lib.ui.bring_python_to_front()

    for key in mel.lib.ui.yield_keys_until_quitkey():
        if key == mel.lib.ui.WAITKEY_RIGHT_ARROW:
            display.next_image()
        elif key == mel.lib.ui.WAITKEY_LEFT_ARROW:
            display.prev_image()
        elif key == mel.lib.ui.WAITKEY_UP_ARROW:
            display.prev_rotomap()
        elif key == mel.lib.ui.WAITKEY_DOWN_ARROW:
            display.next_rotomap()
        elif key == ord('n'):
            num_uuids = len(uuid_to_rotomaps_imagepos_list)
            index += 1
            index %= num_uuids
            uuid_ = uuid_order[index]
            path_images_tuple = tuple(
                uuid_to_rotomaps_imagepos_list[uuid_].values()
            )
            display.reset(path_images_tuple)
        elif key == ord('p'):
            num_uuids = len(uuid_to_rotomaps_imagepos_list)
            index -= 1
            index %= num_uuids
            uuid_ = uuid_order[index]
            path_images_tuple = tuple(
                uuid_to_rotomaps_imagepos_list[uuid_].values()
            )
            display.reset(path_images_tuple)
        elif key == ord(' '):
            display.swap_images()
        elif key == ord('a'):
            display.toggle_crosshairs()


class ImageCompareDisplay:
    """Display two images in a window, supply controls for comparing a list."""

    def __init__(self, name, path_images_tuple, width=None, height=None):
        self._should_draw_crosshairs = True
        self._display = mel.lib.ui.ImageDisplay(name, width, height)
        self.reset(path_images_tuple)

    def reset(self, path_images_tuple):
        if not path_images_tuple:
            raise ValueError(
                "path_images_tuple must be a tuple with at least one thing."
            )

        for group in path_images_tuple:
            if not group:
                raise ValueError("path_images_tuple not have empty groups.")

        self._rotomaps = path_images_tuple

        self._rotomap_cursors = [0] * len(self._rotomaps)

        self._indices = [0, -1]

        self._show()

    def next_image(self):
        ix = self._indices[0]
        num_images = len(self._rotomaps[ix])
        self._rotomap_cursors[ix] += 1
        self._rotomap_cursors[ix] %= num_images
        self._show()

    def prev_image(self):
        ix = self._indices[0]
        num_images = len(self._rotomaps[ix])
        self._rotomap_cursors[ix] -= 1
        self._rotomap_cursors[ix] %= num_images
        self._show()

    def next_rotomap(self):
        num_rotomaps = len(self._rotomaps)
        self._indices[0] += 1
        self._indices[0] %= num_rotomaps
        self._show()

    def prev_rotomap(self):
        num_rotomaps = len(self._rotomaps)
        self._indices[0] -= 1
        self._indices[0] %= num_rotomaps
        self._show()

    def swap_images(self):
        self._indices.reverse()
        self._show()

    def toggle_crosshairs(self):
        self._should_draw_crosshairs = not self._should_draw_crosshairs
        self._show()

    def _path_pos(self, index):
        image_index = self._rotomap_cursors[index]
        return self._rotomaps[index][image_index]

    def _show(self):
        image_width = self._display.width // 2
        image_height = self._display.height
        image_size = numpy.array((image_width, image_height))
        images = [
            captioned_mole_image(
                *self._path_pos(i), image_size, self._should_draw_crosshairs
            )
            for i in self._indices
        ]
        montage = mel.lib.image.montage_horizontal(10, *images)
        self._display.show_image(montage)


def captioned_mole_image(path, pos, size, should_draw_crosshairs):

    image, caption_shape = _cached_captioned_mole_image(
        str(path), tuple(pos), tuple(size)
    )

    if should_draw_crosshairs:
        image_crosshairs = image.copy()
        xpos = image.shape[1] // 2
        ypos = (image.shape[0] - caption_shape[0]) // 2

        mel.rotomap.display.draw_crosshair(image_crosshairs, xpos, ypos)
        image = cv2.addWeighted(image, 0.75, image_crosshairs, 0.25, 0.0)

    return image


@functools.lru_cache()
def _cached_captioned_mole_image(path, pos, size):
    image = mel.lib.image.load_image(path)
    size = numpy.array(size)
    image = mel.lib.image.centered_at(image, pos, size)
    caption = mel.lib.image.render_text_as_image(str(path))
    return (
        mel.lib.image.montage_vertical(10, image, caption),
        caption.shape,
    )


# -----------------------------------------------------------------------------
# Copyright (C) 2018 Angelos Evripiotis.
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
