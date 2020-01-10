"""Generate a mel repo for developing and testing."""

import pathlib
import random

import mel.lib.common

import mel.cmd.error

import mel.rotomap.fake


def setup_parser(parser):
    parser.add_argument(
        "PATH", help="Where to create this generated repo.", type=pathlib.Path
    )


def process_args(args):
    melroot = args.PATH
    if not melroot.exists():
        melroot.mkdir(parents=False)
    if _iterable_len(melroot.iterdir()):
        raise mel.cmd.error.UsageError("Target directory must be empty.")

    melrootfile = melroot / "melroot"
    melrootfile.touch()

    leftleg_lower = melroot / "rotomaps" / "parts" / "LeftLeg" / "Lower"
    leftleg_lower.mkdir(parents=True)
    dir0 = leftleg_lower / "2017_01_01"
    dir1 = leftleg_lower / "2018_01_01"
    dir0.mkdir()
    dir1.mkdir()

    width = 300
    height = 400
    num_moles = 10
    moles = mel.rotomap.fake.random_moles(num_moles)

    for dir_path in (dir0, dir1):
        num_images = random.randint(10, 20)
        for i in range(num_images):
            rot_0_to_1 = i / num_images
            rot_0_to_1 += random.random() / num_images
            image, visible_moles = mel.rotomap.fake.render_moles(
                moles,
                image_width=width,
                image_height=height,
                rot_0_to_1=rot_0_to_1,
            )
            image_path = dir_path / f"{i:02}.jpg"
            mel.lib.common.write_image(image_path, image)
            mel.rotomap.moles.save_image_moles(visible_moles, image_path)


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
