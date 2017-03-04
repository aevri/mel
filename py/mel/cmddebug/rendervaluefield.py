"""Render a 'value field' as an image, to make the idea easier to inspect.

The 'value field' idea from mel.rotomap.pick_value_from_field() is now
important for relating moles together. This visualisation helps check it's
behaviour.

In this image, there is a blue point in the top-left, and a green point in the
bottom-right. Where the error gets high, you can see that redness is
introduced.

"""

import cv2
import numpy

import mel.rotomap.relate


def setup_parser(parser):
    pass


def process_args(args):
    width = 512
    height = 512
    shape = (height, width, 3)
    image = numpy.empty(shape, dtype=int)

    point_values = [
        ((0, 0), (255, 0)),
        ((width, height), (0, 255)),
    ]

    for row in range(height):
        for col in range(width):
            value, error = mel.rotomap.relate.pick_value_from_field(
                numpy.array((col, row)),
                point_values)
            image[row, col, :] = (value[0], value[1], error)

    cv2.imwrite('valuefield.png', image)
