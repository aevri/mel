"""Compare marked moles across rotomaps.

Controls:

    'left arrow' or 'right arrow' to change image in the left slot.
    'up arrow' or 'down arrow' to change rotomap in the left slot.
    'space' to swap left slot and right slot.
    'p' or 'n' to change the mole being examined in both slots.
    'a' to toggle crosshairs on/off.
    'c' to mark the moles as changed in the newest rotomap.
    'u' to mark the moles as unchanged in the newest rotomap.

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


_PosInfo = collections.namedtuple('_PosInfo', 'path pos ellipse_xpos')


def setup_parser(parser):
    parser.add_argument(
        'ROTOMAP',
        type=mel.rotomap.moles.make_argparse_rotomap_directory,
        nargs='+',
        help=(
            "A list of paths to rotomaps. The last rotomap is considered "
            "'the target', and only UUIDs from that one will be compared."
        )
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
    target_rotomap = args.ROTOMAP[-1]
    target_uuids = target_rotomap.calc_uuids()

    uuid_to_rotomaps_imagepos_list = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )

    for rotomap in args.ROTOMAP:
        for frame in rotomap.yield_frames():
            if 'ellipse' not in frame.metadata:
                raise Exception(
                    f'{frame} has no ellipse metadata, '
                    'try running "rotomap calc-space"'
                )
            ellipse = frame.metadata['ellipse']
            elspace = mel.lib.ellipsespace.Transform(ellipse)
            for uuid_, point in frame.moledata.uuid_points.items():
                if uuid_ not in target_uuids:
                    continue
                posinfo = _PosInfo(
                    path=frame.path,
                    pos=point,
                    ellipse_xpos=elspace.to_space(point)[0],
                )
                uuid_to_rotomaps_imagepos_list[uuid_][rotomap.path].append(
                    posinfo
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

    def unchanged_status_keyfunc(uuid_):
        is_unchanged = is_lesion_unchanged(target_rotomap, uuid_)
        if is_unchanged is None:
            return 1
        if not is_unchanged:
            return 0
        return 2

    uuid_order = list(uuid_to_rotomaps_imagepos_list)
    uuid_order.sort(key=unchanged_status_keyfunc)

    index = 0
    uuid_ = uuid_order[index]
    path_images_tuple = tuple(uuid_to_rotomaps_imagepos_list[uuid_].values())
    display = ImageCompareDisplay(
        '.', path_images_tuple, args.display_width, args.display_height
    )
    is_unchanged = is_lesion_unchanged(target_rotomap, uuid_)
    if is_unchanged is not None:
        display.indicate_changed(not is_unchanged)

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
            is_unchanged = is_lesion_unchanged(target_rotomap, uuid_)
            if is_unchanged is not None:
                display.indicate_changed(not is_unchanged)
        elif key == ord('p'):
            num_uuids = len(uuid_to_rotomaps_imagepos_list)
            index -= 1
            index %= num_uuids
            uuid_ = uuid_order[index]
            path_images_tuple = tuple(
                uuid_to_rotomaps_imagepos_list[uuid_].values()
            )
            display.reset(path_images_tuple)
            is_unchanged = is_lesion_unchanged(target_rotomap, uuid_)
            if is_unchanged is not None:
                display.indicate_changed(not is_unchanged)
        elif key == ord(' '):
            display.swap_images()
        elif key == ord('a'):
            display.toggle_crosshairs()
        elif key == ord('c'):
            is_unchanged = False
            mark_lesion(target_rotomap, uuid_, is_unchanged=False)
            display.indicate_changed()
        elif key == ord('u'):
            is_unchanged = True
            mark_lesion(target_rotomap, uuid_, is_unchanged=True)
            display.indicate_changed(False)
        elif key == ord('z'):
            display.adjust_zoom(1.025)
        elif key == ord('x'):
            display.adjust_zoom(1 / 1.025)


def is_lesion_unchanged(rotomap, uuid_):
    """Mark the provided uuid changed status in the lesions datafile."""
    for l in rotomap.lesions:
        if l["uuid"] == uuid_:
            return l[mel.rotomap.moles.KEY_IS_UNCHANGED]
    return None


def mark_lesion(rotomap, uuid_, *, is_unchanged):
    """Mark the provided uuid changed status in the lesions datafile."""
    target_lesion = None
    for l in rotomap.lesions:
        if l["uuid"] == uuid_:
            target_lesion = l
    if target_lesion is None:
        target_lesion = {"uuid": uuid_}
        rotomap.lesions.append(target_lesion)
    target_lesion[mel.rotomap.moles.KEY_IS_UNCHANGED] = is_unchanged
    mel.rotomap.moles.save_rotomap_dir_lesions_file(
        rotomap.path, rotomap.lesions
    )


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
        self._zooms = [1 for _ in path_images_tuple]

        self._rotomap_cursors = [0] * len(self._rotomaps)
        for i, rotomap in enumerate(self._rotomaps):
            centre_index, _ = min(
                enumerate(self._rotomaps[i]),
                key=lambda x: x[1].ellipse_xpos * x[1].ellipse_xpos,
            )
            self._rotomap_cursors[i] = centre_index

        self._indices = [0, -1]

        self._should_indicate_changed = None

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

    def indicate_changed(self, should_indicate_changed=True):
        self._should_indicate_changed = should_indicate_changed
        self._show()

    def adjust_zoom(self, zoom_multiplier):
        ix = self._indices[0]
        self._zooms[ix] *= zoom_multiplier
        self._show()

    def _path_pos_zoom(self, index):
        image_index = self._rotomap_cursors[index]
        posinfo = self._rotomaps[index][image_index]
        zoom = self._zooms[index]
        return posinfo.path, posinfo.pos, zoom

    def _show(self):
        image_width = self._display.width // 2
        image_height = self._display.height
        image_size = numpy.array((image_width, image_height))
        border_colour = None
        if self._should_indicate_changed is not None:
            if self._should_indicate_changed:
                border_colour = (0, 0, 255)
            else:
                border_colour = (0, 255, 0)

        images = [
            captioned_mole_image(
                *self._path_pos_zoom(i),
                image_size,
                self._should_draw_crosshairs,
                border_colour,
            )
            for i in self._indices
        ]
        montage = mel.lib.image.montage_horizontal(10, *images)
        self._display.show_image(montage)


def captioned_mole_image(
    path, pos, zoom, size, should_draw_crosshairs, border_colour=None
):

    image, caption_shape = _cached_captioned_mole_image(
        str(path), tuple(pos), zoom, tuple(size)
    )

    if should_draw_crosshairs:
        image_crosshairs = image.copy()
        xpos = image.shape[1] // 2
        ypos = (image.shape[0] - caption_shape[0]) // 2

        mel.rotomap.display.draw_crosshair(image_crosshairs, xpos, ypos)
        image = cv2.addWeighted(image, 0.75, image_crosshairs, 0.25, 0.0)

    if border_colour is not None:
        cv2.rectangle(image, (0, 0), (image.shape[1], 10), border_colour, -1)

    return image


@functools.lru_cache()
def _cached_captioned_mole_image(path, pos, zoom, size):
    image = mel.lib.image.load_image(path)
    image = mel.lib.image.scale_image(image, zoom)
    pos = tuple(int(v * zoom) for v in pos)
    size = numpy.array(size)
    image = mel.lib.image.centered_at(image, pos, size)
    caption = mel.lib.image.render_text_as_image(str(path))
    return (
        mel.lib.image.montage_vertical(10, image, caption),
        caption.shape,
    )


# -----------------------------------------------------------------------------
# Copyright (C) 2018-2019 Angelos Evripiotis.
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
