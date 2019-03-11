"""Generate a mel repo for developing and testing."""

import pathlib
import random

import cv2

import mel.lib.common

import mel.cmd.error


def setup_parser(parser):
    parser.add_argument(
        'PATH',
        help='Where to create this generated repo.',
        type=pathlib.Path)


def process_args(args):
    melroot = args.PATH
    if not melroot.exists():
        melroot.mkdir(parents=False)
    if _iterable_len(melroot.iterdir()):
        raise mel.cmd.error.UsageError('Target directory must be empty.')

    melrootfile = melroot / 'melroot'
    melrootfile.touch()

    leftleg_lower = melroot / 'rotomaps' / 'parts' / 'LeftLeg' / 'Lower'
    leftleg_lower.mkdir(parents=True)
    dir1 = leftleg_lower / '2018_01_01'
    dir1.mkdir()

    width = 3042
    height = 4032
    num_moles = 10
    write_fake_image(
        dir1 / '0.jpg', width=width, height=height, num_moles=num_moles)


def write_fake_image(path, width, height, num_moles):
    image = mel.lib.common.new_image(height, width)

    # Set the background to a non-organic green colour.
    image[:, :] = [0, 255, 0]

    # Draw some 'skin'.
    left = width // 10
    right = left * 9
    top = height // 10
    bottom = top * 9
    image[top:bottom, left:right, :] = 255

    # Create a slightly irregular border, so that e.g. calc-space doesn't get
    # tripped up by the unnaturally small number of vertices.
    image[top-1:top, left-1:right+1, :] = 255
    image[bottom:bottom+1, left-1:right+1, :] = 255
    image[top+1:bottom-1, left-1:left, :] = 255
    image[top+1:bottom-1, right:right+1, :] = 255

    # Draw the 'moles'.
    skin_width = left * 8
    skin_height = top * 8
    # Note that it's important that the 'moles' have some saturation, or
    # automark won't pick them up.
    brown = (0, 150, 100)
    max_radius = 50
    for _ in range(num_moles):
        x = random.randrange(skin_width) + left
        y = random.randrange(skin_height) + top
        radius = random.randrange(max_radius)
        cv2.circle(image, (x, y), radius, brown, -1)

    mel.lib.common.write_image(path, image)


def _iterable_len(it):
    return sum(1 for _ in it)


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
