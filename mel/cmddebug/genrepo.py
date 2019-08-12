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

    skin_rect = Rect(
        left=width // 10,
        width=int(width * 0.8),
        top=height // 10,
        height=int(height * 0.8)
    )
    num_moles = 10
    min_radius = 20
    max_radius = 50
    moles = gen_moles(num_moles, skin_rect, min_radius, max_radius)
    jitter = 10

    num_images = 20

    for i in range(num_images):
        write_fake_image(
            dir1 / (str(i) + '.jpg'),
            width=width,
            height=height,
            skin_rect=skin_rect,
            moles=moles,
            jitter=jitter)


def gen_moles(num_moles, skin_rect, min_radius, max_radius):
    moles = []
    for _ in range(num_moles):
        x = random.randrange(skin_rect.width) + skin_rect.left
        y = random.randrange(skin_rect.height) + skin_rect.top
        radius = random.randrange(min_radius, max_radius)
        moles.append(Mole(x, y, radius))
    return moles


class Mole:

    def __init__(self, x, y, radius):
        self.x, self.y, self.radius = x, y, radius


class Rect:

    def __init__(self, left, width, top, height):
        self.left = left
        self.right = left + width
        self.top = top
        self.bottom = top + height
        self.width = width
        self.height = height


def write_fake_image(path, width, height, skin_rect, moles, jitter):
    image = mel.lib.common.new_image(height, width)

    # Set the background to a non-organic green colour.
    image[:, :] = [0, 255, 0]

    s = skin_rect

    # Draw some 'skin'.
    left = width // 10
    right = left * 9
    top = height // 10
    bottom = top * 9
    image[s.top:s.bottom, s.left:s.right, :] = 255

    # Create a slightly irregular border, so that e.g. calc-space doesn't get
    # tripped up by the unnaturally small number of vertices.
    image[s.top-1:s.top, s.left-1:s.right+1, :] = 255
    image[s.bottom:s.bottom+1, s.left-1:s.right+1, :] = 255
    image[s.top+1:s.bottom-1, s.left-1:s.left, :] = 255
    image[s.top+1:s.bottom-1, s.right:s.right+1, :] = 255

    # Draw the 'moles'.
    # Note that it's important that the 'moles' have some saturation, or
    # automark won't pick them up.
    brown = (0, 150, 100)
    for m in moles:
        j_x = random.randrange(jitter)
        j_y = random.randrange(jitter)
        j_radius = random.randrange(jitter)
        cv2.circle(
            image,
            (m.x + j_x, m.y + j_y),
            m.radius + j_radius,
            brown,
            -1)

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
