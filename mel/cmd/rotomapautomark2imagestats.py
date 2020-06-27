"""Calc image stats for training a model to mark moles on rotomap images."""

import tqdm

import mel.rotomap.detectmolesnn


def setup_parser(parser):
    parser.add_argument(
        "IMAGES", nargs="+", help="A list of paths to images to automark."
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print information about the processing.",
    )


def process_args(args):
    to_tensor = mel.rotomap.detectmolesnn.to_tensor
    get_masked_image = mel.rotomap.detectmolesnn.get_masked_image

    progress_bar_iter = tqdm.tqdm(
        (to_tensor(get_masked_image(path)) for path in args.IMAGES),
        total=len(args.IMAGES),
    )

    mean, std = mel.rotomap.detectmolesnn.calc_images_mean_std(
        progress_bar_iter
    )
    print("Mean:", mean)
    print("Standard deviation:", std)


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
