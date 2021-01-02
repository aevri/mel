"""Generate a mel repo for developing and testing."""

import pathlib
import random

import mel.cmd.error
import mel.lib.common
import mel.rotomap.fake


def setup_parser(parser):
    parser.add_argument(
        "PATH", help="Where to create this generated repo.", type=pathlib.Path
    )
    parser.add_argument(
        "--num-rotomaps",
        "-n",
        help="How many rotomaps to make.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num-parts",
        "-p",
        help="How many parts to make.",
        type=int,
        default=1,
    )


def process_args(args):
    melroot = args.PATH
    if not melroot.exists():
        melroot.mkdir(parents=False)
    if _iterable_len(melroot.iterdir()):
        raise mel.cmd.error.UsageError("Target directory must be empty.")

    melrootfile = melroot / "melroot"
    melrootfile.touch()

    width = 300
    height = 400

    make_fake_micro(melroot)

    part_names = ["LeftLeg"]
    part_names.extend([f"Part{i}" for i in range(2, args.num_parts + 1)])

    for part in part_names:
        leftleg_lower = melroot / "rotomaps" / "parts" / part / "Lower"
        leftleg_lower.mkdir(parents=True)

        num_moles = 10
        moles = mel.rotomap.fake.random_moles(num_moles)

        for dirnum in range(args.num_rotomaps):
            dir_path = leftleg_lower / f"{2018 - dirnum}_01_01"
            dir_path.mkdir()
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


def make_fake_micro(melroot):
    micro_parts_path = melroot / "micro" / "data"
    micro_parts_path.mkdir(parents=True)


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
