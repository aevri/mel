"""Compare marked moles across rotomaps.

Controls:

    'left arrow' or 'right arrow' to change image in the left slot.
    'up arrow' or 'down arrow' to change rotomap in the left slot.
    'space' to swap left slot and right slot.
    'p' or 'n' to change the mole being examined in both slots.
    'a' to toggle crosshairs on/off.
    'c' to mark the moles as changed in the newest rotomap.
    'u' to mark the moles as unchanged in the newest rotomap.

    'l' to zoom and rotate the left slot to roughly align with the right slot.

    'z' to zoom in on the left slot.
    'x' to zoom out on the left slot.

    'j' to rotate left on the left slot.
    'k' to rotate right on the left slot.

    'm' to toggle displaying mole markers.

    'q' to quit.
"""

import collections
import functools
import math
import os

import cv2
import numpy

import mel.lib.common
import mel.lib.datetime
import mel.lib.image
import mel.lib.math
import mel.lib.moleimaging
import mel.rotomap.display
import mel.rotomap.moles

_PosInfo = collections.namedtuple("_PosInfo", "path pos ellipse_xpos uuid uuid_points")


def setup_parser(parser):
    parser.add_argument(
        "ROTOMAP",
        type=mel.rotomap.moles.make_argparse_rotomap_directory,
        nargs="+",
        help=(
            "A list of paths to rotomaps. The last rotomap is considered "
            "'the target', and only UUIDs from that one will be compared."
        ),
    )


def process_args(args):
    target_rotomap = args.ROTOMAP[-1]
    target_uuids = target_rotomap.calc_uuids()

    uuid_to_rotomaps_imagepos_list = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )

    for rotomap in args.ROTOMAP:
        for frame in rotomap.yield_frames():
            if "ellipse" not in frame.metadata:
                raise Exception(
                    f'{frame} has no ellipse metadata, try running "rotomap calc-space"'
                )
            ellipse = frame.metadata["ellipse"]
            elspace = mel.lib.ellipsespace.Transform(ellipse)
            for uuid_, point in frame.moledata.uuid_points.items():
                if uuid_ not in target_uuids:
                    continue
                posinfo = _PosInfo(
                    path=frame.path,
                    pos=point,
                    ellipse_xpos=elspace.to_space(point)[0],
                    uuid=uuid_,
                    uuid_points=frame.moledata.uuid_points,
                )
                uuid_to_rotomaps_imagepos_list[uuid_][rotomap.path].append(posinfo)

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
    uuid_ = uuid_order[0]

    # Import pygame as late as possible, to avoid displaying its
    # startup-text where it is not actually used.
    import pygame

    path_images_tuple = tuple(uuid_to_rotomaps_imagepos_list[uuid_].values())
    with (
        mel.lib.common.timelogger_context("rotomap-compare") as logger,
        mel.lib.fullscreenui.fullscreen_context() as screen,
    ):
        display = ImageCompareDisplay(logger, screen, path_images_tuple, uuid_)

        on_keydown = _make_on_keydown(
            display,
            uuid_order,
            target_rotomap,
            uuid_to_rotomaps_imagepos_list,
        )

        for event in mel.lib.fullscreenui.yield_events_until_quit(screen):
            if event.type == pygame.KEYDOWN:
                on_keydown(event)


def _make_on_keydown(
    display, uuid_order, target_rotomap, uuid_to_rotomaps_imagepos_list
):
    # Import pygame as late as possible, to avoid displaying its
    # startup-text where it is not actually used.
    import pygame

    index = 0
    uuid_ = uuid_order[index]
    is_unchanged = is_lesion_unchanged(target_rotomap, uuid_)
    if is_unchanged is not None:
        display.indicate_changed(not is_unchanged)

    def on_keydown(event):
        key = event.key
        nonlocal index
        if key == pygame.K_RIGHT:
            display.next_image()
        elif key == pygame.K_LEFT:
            display.prev_image()
        elif key == pygame.K_UP:
            display.prev_rotomap()
        elif key == pygame.K_DOWN:
            display.next_rotomap()
        elif key == pygame.K_n:
            num_uuids = len(uuid_to_rotomaps_imagepos_list)
            index += 1
            index %= num_uuids
            uuid_ = uuid_order[index]
            path_images_tuple = tuple(uuid_to_rotomaps_imagepos_list[uuid_].values())
            display.reset(path_images_tuple, uuid_)
            is_unchanged = is_lesion_unchanged(target_rotomap, uuid_)
            if is_unchanged is not None:
                display.indicate_changed(not is_unchanged)
        elif key == pygame.K_p:
            num_uuids = len(uuid_to_rotomaps_imagepos_list)
            index -= 1
            index %= num_uuids
            uuid_ = uuid_order[index]
            path_images_tuple = tuple(uuid_to_rotomaps_imagepos_list[uuid_].values())
            display.reset(path_images_tuple, uuid_)
            is_unchanged = is_lesion_unchanged(target_rotomap, uuid_)
            if is_unchanged is not None:
                display.indicate_changed(not is_unchanged)
        elif key == pygame.K_SPACE:
            display.swap_images()
        elif key == pygame.K_a:
            display.toggle_crosshairs()
        elif key == pygame.K_c:
            is_unchanged = False
            uuid_ = uuid_order[index]
            mark_lesion(target_rotomap, uuid_, is_unchanged=False)
            display.indicate_changed()
        elif key == pygame.K_u:
            is_unchanged = True
            uuid_ = uuid_order[index]
            mark_lesion(target_rotomap, uuid_, is_unchanged=True)
            display.indicate_changed(False)
        elif key == pygame.K_z:
            display.adjust_zoom(1.025)
        elif key == pygame.K_x:
            display.adjust_zoom(1 / 1.025)
        elif key == pygame.K_l:
            display.auto_align_and_show()
        elif key == pygame.K_j:
            display.adjust_rotation(2)
        elif key == pygame.K_k:
            display.adjust_rotation(-2)
        elif key == pygame.K_m:
            display.toggle_mole_display()

    return on_keydown


def is_lesion_unchanged(rotomap, uuid_):
    """Mark the provided uuid changed status in the lesions datafile."""
    for lesion in rotomap.lesions:
        if lesion["uuid"] == uuid_:
            return lesion[mel.rotomap.moles.KEY_IS_UNCHANGED]
    return None


def mark_lesion(rotomap, uuid_, *, is_unchanged):
    """Mark the provided uuid changed status in the lesions datafile."""
    target_lesion = None
    for lesion in rotomap.lesions:
        if lesion["uuid"] == uuid_:
            target_lesion = lesion
    if target_lesion is None:
        target_lesion = {"uuid": uuid_}
        rotomap.lesions.append(target_lesion)
    target_lesion[mel.rotomap.moles.KEY_IS_UNCHANGED] = is_unchanged
    mel.rotomap.moles.save_rotomap_dir_lesions_file(rotomap.path, rotomap.lesions)


class ImageCompareDisplay:
    """Display two images in a window, supply controls for comparing a list."""

    def __init__(self, logger, screen, path_images_tuple, uuid_):
        self._logger = logger
        self._image_path = None
        self._uuid = None
        self._draw_moles = False
        self._should_draw_crosshairs = True
        self._display = screen
        self._melroot = mel.lib.fs.find_melroot()
        self.reset(path_images_tuple, uuid_)

    def _reset_logger(self):
        self._logger.reset(
            mode="compare",
            path=str(
                os.path.relpath(
                    os.path.abspath(self._image_path),
                    start=self._melroot,
                )
            )
            + ":"
            + self._uuid,
        )

    def reset(self, path_images_tuple, uuid_):
        if not path_images_tuple:
            raise ValueError(
                "path_images_tuple must be a tuple with at least one thing."
            )

        for group in path_images_tuple:
            if not group:
                raise ValueError("path_images_tuple not have empty groups.")

        self._uuid = uuid_
        self._rotomaps = path_images_tuple
        self._zooms = [1 for _ in path_images_tuple]
        self._rotations = [0 for _ in path_images_tuple]

        self._rotomap_cursors = [0] * len(self._rotomaps)
        for i, rotomap in enumerate(self._rotomaps):
            centre_index, _ = min(
                enumerate(self._rotomaps[i]),
                key=lambda x: x[1].ellipse_xpos * x[1].ellipse_xpos,
            )
            self._rotomap_cursors[i] = centre_index

        self._indices = [0, -1]

        self._should_indicate_changed = None

        self.auto_align_and_show()
        self._reset_logger()

    def next_image(self):
        ix = self._indices[0]
        num_images = len(self._rotomaps[ix])
        self._rotomap_cursors[ix] += 1
        self._rotomap_cursors[ix] %= num_images
        self.auto_align_and_show()

    def prev_image(self):
        ix = self._indices[0]
        num_images = len(self._rotomaps[ix])
        self._rotomap_cursors[ix] -= 1
        self._rotomap_cursors[ix] %= num_images
        self.auto_align_and_show()

    def next_rotomap(self):
        num_rotomaps = len(self._rotomaps)
        self._indices[0] += 1
        self._indices[0] %= num_rotomaps
        self.auto_align_and_show()

    def prev_rotomap(self):
        num_rotomaps = len(self._rotomaps)
        self._indices[0] -= 1
        self._indices[0] %= num_rotomaps
        self.auto_align_and_show()

    def swap_images(self):
        self._indices.reverse()
        self._show()

    def toggle_crosshairs(self):
        self._should_draw_crosshairs = not self._should_draw_crosshairs
        self._show()

    def toggle_mole_display(self):
        self._draw_moles = not self._draw_moles
        self._show()

    def indicate_changed(self, should_indicate_changed=True):
        self._should_indicate_changed = should_indicate_changed
        self._show()

    def adjust_zoom(self, zoom_multiplier):
        ix = self._indices[0]
        self._zooms[ix] *= zoom_multiplier
        self._show()

    def adjust_rotation(self, rotation_modifier):
        ix = self._indices[0]
        self._rotations[ix] += rotation_modifier
        self._show()

    def _posinfo(self, index):
        ix = self._indices[index]
        image_index = self._rotomap_cursors[ix]
        return self._rotomaps[ix][image_index]

    def auto_align_and_show(self):
        left_posinfo = self._posinfo(0)
        right_posinfo = self._posinfo(1)
        target_uuid = left_posinfo.uuid
        assert right_posinfo.uuid == target_uuid

        common_uuids = set(left_posinfo.uuid_points) & set(right_posinfo.uuid_points)
        common_uuids.remove(target_uuid)
        if not common_uuids:
            self._show()
            return

        left_target_pos = next(
            pos
            for uuid_, pos in left_posinfo.uuid_points.items()
            if uuid_ == target_uuid
        )

        nearest_common_uuid = min(
            common_uuids,
            key=lambda u: mel.lib.math.distance_sq_2d(
                left_posinfo.uuid_points[u], left_target_pos
            ),
        )

        left_dist = math.sqrt(
            mel.lib.math.distance_sq_2d(
                left_posinfo.uuid_points[nearest_common_uuid], left_target_pos
            )
        )

        right_target_pos = next(
            pos
            for uuid_, pos in right_posinfo.uuid_points.items()
            if uuid_ == target_uuid
        )

        right_dist = math.sqrt(
            mel.lib.math.distance_sq_2d(
                right_posinfo.uuid_points[nearest_common_uuid],
                right_target_pos,
            )
        )

        self._zooms[self._indices[0]] = right_dist / left_dist
        self._zooms[self._indices[1]] = 1.0

        left_angle = mel.lib.math.angle(
            left_posinfo.uuid_points[nearest_common_uuid] - left_target_pos
        )
        right_angle = mel.lib.math.angle(
            right_posinfo.uuid_points[nearest_common_uuid] - right_target_pos
        )

        self._rotations[self._indices[0]] = right_angle - left_angle
        self._rotations[self._indices[1]] = 1.0

        self._show()

    def _path_pos_zoom_rotation_moles(self, index):
        image_index = self._rotomap_cursors[index]
        posinfo = self._rotomaps[index][image_index]
        zoom = self._zooms[index]
        rotation = self._rotations[index]
        return posinfo.path, posinfo.pos, zoom, rotation, posinfo.uuid_points

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
                *self._path_pos_zoom_rotation_moles(i),
                image_size,
                self._should_draw_crosshairs,
                border_colour,
                self._draw_moles,
            )
            for i in self._indices
        ]
        self._image_path = self._path_pos_zoom_rotation_moles(self._indices[-1])[0]
        montage = mel.lib.image.montage_horizontal(10, *images)
        self._display.show_opencv_image(montage)


def captioned_mole_image(
    path,
    pos,
    zoom,
    rotation_degs,
    uuid_points,
    size,
    should_draw_crosshairs,
    border_colour=None,
    draw_moles=False,
):
    points = None
    if draw_moles and uuid_points is not None:
        points = tuple(tuple(p) for p in uuid_points.values())

    image, caption_shape = _cached_captioned_mole_image(
        str(path), tuple(pos), zoom, tuple(size), rotation_degs, points
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


@functools.lru_cache
def _cached_captioned_mole_image(path, pos, zoom, size, rotation_degs, points):
    image = mel.lib.image.load_image(path)
    image = mel.lib.image.scale_image(image, zoom)
    colors = [[255, 0, 0], [255, 128, 128], [255, 0, 0]]
    if points is not None:
        for x, y in points:
            x *= zoom
            x = int(x)
            y *= zoom
            y = int(y)
            mel.rotomap.display.draw_mole(image, x, y, colors)
    pos = tuple(int(v * zoom) for v in pos)
    size = numpy.array(size)
    max_size = 2 * max(size)
    max_size = numpy.array([max_size, max_size])
    max_size_2 = 2 * max_size
    image = mel.lib.image.centered_at(image, pos, max_size_2)
    image = mel.lib.image.rotated(image, rotation_degs)
    image = mel.lib.image.centered_at(image, max_size, size)
    caption = mel.lib.image.render_text_as_image(str(path))
    return (
        mel.lib.image.montage_vertical(10, image, caption),
        caption.shape,
    )


# -----------------------------------------------------------------------------
# Copyright (C) 2018-2021 Angelos Evripiotis.
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
