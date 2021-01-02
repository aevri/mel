"""Pre-calculate features for training 'filter-marks' with."""

import tqdm

import mel.lib.common
import mel.rotomap.filtermarks
import mel.rotomap.mask
import mel.rotomap.moles

# TODO: Check for masking errors, where the mole is obscured by the mask.
# TODO: Make the mask green.


def setup_parser(parser):

    parser.add_argument(
        "FRAMES",
        type=mel.rotomap.moles.make_argparse_image_moles,
        nargs="+",
        help="Path to rotomap or image to copy from.",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print information about the processing.",
    )

    # parser.add_argument(
    #     "--size-half",
    #     "-s",
    #     default=mel.rotomap.filtermarks.DEFAULT_SIZE // 2,
    #     type=int,
    #     help="The 'radius' around the items, pixels. i.e. x2 to calc width.",
    # )

    parser.add_argument(
        "--batch-size",
        default=mel.rotomap.filtermarks.DEFAULT_BATCH_SIZE,
        type=int,
        help="How many image patches to process at once. Higher is better.",
    )


def process_args(args):
    for image_mole_iter in tqdm.tqdm(args.FRAMES):
        for image_path, moles in image_mole_iter:
            try:
                mel.rotomap.filtermarks.pretrain_image(
                    image_path, moles, args.batch_size
                )
            except Exception:
                raise Exception("Error while processing {}".format(image_path))


# -----------------------------------------------------------------------------
# Copyright (C) 2020 Angelos Evripiotis.
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
