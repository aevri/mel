"""Image processing routines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

import mel.lib.common


def montage_horizontal(left_image, right_image):
    if left_image.shape != right_image.shape:
        raise ValueError('image shapes must be identical')

    # calculate the bounds of the montage
    border_size = 50
    total_border_size = numpy.array([border_size * 2, border_size * 3])
    image_shape = left_image.shape
    total_image_size = numpy.array([image_shape[0], image_shape[1] * 2])
    total_montage_size = total_border_size + total_image_size

    montage_image = mel.lib.common.new_image(
        total_montage_size[0], total_montage_size[1])

    # write the images into the montage image
    x = border_size
    mel.lib.common.copy_image_into_image(
        left_image, montage_image, border_size, x)
    x += border_size
    x += image_shape[1]
    mel.lib.common.copy_image_into_image(
        right_image, montage_image, border_size, x)

    return montage_image
