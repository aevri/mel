"""Render a 'value field' as an image, to make the idea easier to inspect.

The 'value field' idea from mel.rotomap.pick_value_from_field() is now
important for relating moles together. This visualisation helps check it's
behaviour.

In this image, there is a blue point in the top-left, and a green point in the
bottom-right. Where the error gets high, you can see that redness is
introduced.
"""

import numpy

import mel.lib.common
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

    mel.lib.common.write_image('valuefield.png', image)
# -----------------------------------------------------------------------------
# Copyright (C) 2017 Angelos Evripiotis.
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
