"""Image processing routines."""


import cv2
import numpy

import mel.lib.common


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise Exception(f'Failed to load image: "{path}"')

    return image


def calc_letterbox(width, height, fit_width, fit_height):
    """Return (x, y, width, height) to fit image into.

    Usage example:
        >>> calc_letterbox(4, 2, 2, 1)
        (0, 0, 2, 1)
        >>> calc_letterbox(2, 1, 4, 2)
        (1, 0, 2, 1)
    """
    if width < fit_width and height < fit_height:
        scale = 1
    else:
        scale_x = fit_width / width
        scale_y = fit_height / height
        scale = min(scale_x, scale_y)

    new_width = int(width * scale)
    new_height = int(height * scale)

    x = (fit_width - new_width) // 2
    y = (fit_height - new_height) // 2

    return (x, y, new_width, new_height)


def letterbox(image, width, height):
    x, y, new_width, new_height = calc_letterbox(
        image.shape[1], image.shape[0], width, height)
    resized_image = cv2.resize(
        image,
        (new_width, new_height))
    letterboxed = mel.lib.common.new_image(
        height, width)
    mel.lib.common.copy_image_into_image(
        resized_image, letterboxed, y, x)
    return letterboxed


def calc_montage_horizontal(border_size, *frames):
    """Return total[], pos1[], pos2[], ... for a horizontal montage.

    Usage example:
        >>> calc_montage_horizontal(1, [2,1], [3,2])
        ([8, 4], [1, 1], [4, 1])
    """
    num_frames = len(frames)
    total_width = sum(f[0] for f in frames) + (border_size * num_frames + 1)
    max_height = max(f[1] for f in frames)
    total_height = max_height + (2 * border_size)

    x = border_size
    pos_list = []
    for f in frames:
        y = border_size + (max_height - f[1]) // 2
        pos_list.append([x, y])
        x += f[0] + border_size

    result = [[total_width, total_height]]
    result.extend(pos_list)
    return tuple(result)


def calc_montage_vertical(border_size, *frames):
    """Return total[], pos1[], pos2[], ... for a vertical montage.

    Usage example:
        >>> calc_montage_vertical(1, [2,1], [3,2])
        ([5, 6], [1, 1], [1, 3])
    """
    geometry = calc_montage_horizontal(
        border_size,
        *[list(reversed(f)) for f in frames])

    return tuple([g[1], g[0]] for g in geometry)


def arrange_images(total_width, total_height, *images_positions):
    """Return a composited image based on the (image, pos) arguments."""
    result = mel.lib.common.new_image(total_height, total_width)

    for image, pos in images_positions:
        mel.lib.common.copy_image_into_image(
            image, result, pos[1], pos[0])

    return result


def montage_horizontal_inner_border(divider_size, *image_list):
    """Return a new image, of the supplied images in a row.

    Will keep a separator of 'divider_size' pixels between each image. The
    images are vertically centered.

    Usage example:

        Arrange varied images in a row, with a 50 pixel separator:
        >>> blue = numpy.full((600, 800, 3), (255, 0, 0), dtype=numpy.uint8)
        >>> grey_half = numpy.full((300, 800, 1), (128), dtype=numpy.uint8)
        >>> red = numpy.full((600, 800, 3), (0, 0, 255), dtype=numpy.uint8)
        >>> image = montage_horizontal_inner_border(50, blue, grey_half, red)

    Counter example:

        Forget to specify image list:
        >>> mel.lib.image.montage_horizontal_inner_border(50)
        Traceback (most recent call last):
            ...
        ValueError: Must provide image_list
    """
    if not image_list:
        raise ValueError('Must provide image_list')

    height = max(i.shape[0] for i in image_list)
    width = sum(i.shape[1] for i in image_list)
    splits = len(image_list) - 1

    width += divider_size * splits

    output = mel.lib.common.new_image(height, width)

    x = 0
    for image in image_list:
        iheight = image.shape[0]
        iwidth = image.shape[1]
        y = (height - iheight) // 2
        output[y:y + iheight, x:x + iwidth] = image[:, :]
        x += iwidth + divider_size

    return output


def montage_horizontal(border_size, *image_list):
    geometry = calc_montage_horizontal(
        border_size,
        *[list(reversed(i.shape[:2])) for i in image_list])

    size_xy = geometry[0]
    geometry = geometry[1:]

    return arrange_images(
        size_xy[0],
        size_xy[1],
        *list(zip(image_list, geometry)))


def montage_vertical(border_size, *image_list):
    geometry = calc_montage_vertical(
        border_size,
        *[list(reversed(i.shape[:2])) for i in image_list])

    size_xy = geometry[0]
    geometry = geometry[1:]

    return arrange_images(
        size_xy[0],
        size_xy[1],
        *list(zip(image_list, geometry)))


def render_text_as_image(
        text,
        font_face=None,
        font_scale=None,
        thickness=None,
        color=None):

    if font_face is None:
        font_face = cv2.FONT_HERSHEY_DUPLEX
    if font_scale is None:
        font_scale = 1
    if thickness is None:
        thickness = 1
    if color is None:
        color = (255, 255, 255)

    (width, height), baseline = cv2.getTextSize(
        text, font_face, font_scale, thickness)

    baseline += thickness

    image = mel.lib.common.new_image(height + baseline, width)
    textpos = (0, height)
    cv2.putText(image, text, textpos, font_face, font_scale, color)
    return image


def calc_centering_offset(centre_xy, dst_size_xy):
    dst_centre = [i // 2 for i in dst_size_xy]
    offset = [i[1] - i[0] for i in zip(centre_xy, dst_centre)]
    return offset


def centered_at(image, src_pos, dst_rect):

    dst_selection, src_selection = calc_centered_at_selections(
        get_image_rect(image), src_pos, dst_rect)

    result = mel.lib.common.new_image(*numpy.flipud(dst_rect))
    result[dst_selection] = image[src_selection]

    return result


def calc_centered_at_selections(src_rect, src_pos, dst_rect):
    """Return (dst, src) selections for centering at src_pos.

    :src_rect: A numpy.array of src's (width, height)
    :src_pos: A numpy.array of the (x, y) in source space to centre at
    :dst_rect: A numpy.array of dst's (width, height)
    :returns: A tuple (dst, src) of selections for centering

    For example, the slices can be used like this to write the source at the
    required location:

        result[dst_slices] = image[src_slices]
    """
    dst_mid = dst_rect // 2

    # Calculate the dst geometry, unclipped
    dst_start = dst_mid - src_pos
    dst_end = dst_start + src_rect

    # Project the dst clip rect into source space and clip the src rect to it
    src_start = numpy.clip(-dst_start, 0, src_rect)
    src_end = numpy.clip(dst_rect - dst_start, 0, src_rect)

    # Clip the dst rect
    numpy.clip(dst_start, 0, dst_rect, dst_start)
    numpy.clip(dst_end, 0, dst_rect, dst_end)

    dst_selection = positions_to_selection(dst_start, dst_end)
    src_selection = positions_to_selection(src_start, src_end)

    return dst_selection, src_selection


def positions_to_selection(top_left_inclusive, bottom_right_exclusive):
    """Return a selection object for an image as specified by positions.

    Note that the bottom right point will not be included in the selection.

    :top_left_inclusive: a numpy.array of (top, left) co-ordinates
    :bottom_right_exclusive: a numpy.array of (bottom, right) co-ordinates
    :returns: A tuple (rows, columns) slices as a selection object
    """
    return (
        slice(top_left_inclusive[1], bottom_right_exclusive[1]),
        slice(top_left_inclusive[0], bottom_right_exclusive[0])
    )


def slice_square_or_none(image, lefttop, rightbottom):
    """Return a slice of the supplied image or None.

    :image: a NumPy array representing an OpenCV image, stored in yx order.
    :lefttop: a NumPy array of xy co-ordinates, the inclusive top-left.
    :rightbottom: a NumPy array of xy co-ordinates, the exclusive bottom-right.
    :returns: a NumPy array representing an OpenCV image, stored in yx order.
    """
    height_width = image.shape[:2]
    width_height = (height_width[1], height_width[0])

    clipped_lefttop = numpy.clip(lefttop, (0, 0), width_height)
    clipped_rightbottom = numpy.clip(rightbottom, (0, 0), width_height)

    if not numpy.allclose(lefttop, clipped_lefttop):
        return None

    if not numpy.allclose(rightbottom, clipped_rightbottom):
        return None

    # Note that images are stored in yx order, not xy.
    return image[
        lefttop[1]:rightbottom[1],
        lefttop[0]:rightbottom[0],
    ]


def recentered_at(image, x, y):
    """Return a new image, centered at new position on a black background.

    Where new content needs to be shifted into the image, it will appear black.

    :image: An OpenCV image.
    :x: The horizontal co-ordinate to put at the centre of the new image.
    :y: The vertical co-ordinate to put at the centre of the new image.
    :returns: A new OpenCV image.
    """
    return centered_at(
        image,
        numpy.array((x, y)),
        get_image_rect(image))


def get_image_rect(image):
    """Return the (width, height) of the supplied 'image'.

    :image: A numpy.ndarray representing an image.
    :returns: A numpy.ndarray representing (width, height) of the image.
    """
    return numpy.flipud(image.shape[:2])


def rotated(image, degrees):
    """Return a new image, rotated by specified amount, on a black background.

    Where new content needs to be shifted into the image, it will appear black.

    :image: An OpenCV image.
    :degrees: The degrees of the rotation about the centre.
    :returns: A new OpenCV image.
    """
    height, width = image.shape[0:2]

    rot = cv2.getRotationMatrix2D((width // 2, height // 2), degrees, 1.0)

    return cv2.warpAffine(image, rot, (width, height))


def rotated180(image):
    return cv2.flip(image, -1)
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
