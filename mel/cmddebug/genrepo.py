"""Generate a mel repo for developing and testing."""

import pathlib

import mel.lib.common

import mel.cmd.error

import mel.rotomap.fake


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

    width = 300
    height = 400
    num_moles = 10
    moles = mel.rotomap.fake.random_moles(num_moles)
    image = mel.rotomap.fake.render_moles(
        moles, image_width=width, image_height=height
    )
    mel.lib.common.write_image(dir1 / '0.jpg', image)


def _iterable_len(it):
    return sum(1 for _ in it)


# -----------------------------------------------------------------------------
# Copyright (C) 2018-2020 Angelos Evripiotis.
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
