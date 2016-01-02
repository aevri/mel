"""Image processing routines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2

import mel.lib.common


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
    for index, f in enumerate(frames):
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


def montage_horizontal(border_size, *image_list):
    geometry = calc_montage_horizontal(
        border_size,
        *[list(reversed(i.shape[:2])) for i in image_list])

    size_xy = geometry[0]
    geometry = geometry[1:]

    return arrange_images(
        size_xy[0],
        size_xy[1],
        *zip(image_list, geometry))


def montage_vertical(border_size, *image_list):
    geometry = calc_montage_vertical(
        border_size,
        *[list(reversed(i.shape[:2])) for i in image_list])

    size_xy = geometry[0]
    geometry = geometry[1:]

    return arrange_images(
        size_xy[0],
        size_xy[1],
        *zip(image_list, geometry))


def calc_centering_offset(centre_xy, src_size_xy, dst_size_xy):
    """Return translation (x, y) for centering with the supplied geometry.

    The supplied 'src_size_xy' represent the dimensions of the image to
    translate, fitting withing the bounds of 'dst_size_xy'.

    It is assumed that the source image will start drawing at (0, 0) in the
    destination region.

    Usage examples:
        >>> calc_centering_offset((10, 10), (100, 100), (40, 40))
        (0, 0)

        >>> calc_centering_offset((40, 40), (100, 100), (40, 40))
        (20, 20)

        >>> calc_centering_offset((50, 50), (100, 100), (40, 40))
        (30, 30)

    """
    result = []
    for i in (0, 1):
        result.append(
            mel.lib.math.clamp(
                centre_xy[i] - (dst_size_xy[i] // 2),
                0,
                src_size_xy[i] - 1))
    return tuple(result)


def translated_and_clipped(image, x, y, dst_width, dst_height):
    result = mel.lib.common.new_image(dst_height, dst_width)
    image_shape = image.shape
    src_width = image_shape[1] - x
    src_height = image_shape[0] - y
    width = min(src_width, dst_width)
    height = min(src_height, dst_height)
    result[0:height, 0:width] = image[y:y + height, x:x + width]
    return result
