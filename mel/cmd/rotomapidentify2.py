"""Guess which mole is which in a rotomap image."""

import os
import warnings


def setup_parser(parser):
    parser.add_argument(
        "TARGET",
        nargs="+",
        help="Paths to images to identify.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print information about the processing.",
    )
    parser.add_argument(
        "--extra-stem",
        help="Add an extra bit to the filename stem, e.g. '0.jpg.EXTRA.json'.",
    )


def process_args(args):
    # This is an expensive import, so only do it when necessary.
    import mel.rotomap.identifynn2

    warnings.filterwarnings(
        "ignore",
        message="Creating a tensor from a list of numpy.ndarrays is extremely slow.",
    )

    identifier = mel.rotomap.identifynn2.make_identifier()
    for target in args.TARGET:
        if args.verbose:
            print("Processing", target, "..")

        new_moles = identifier.get_new_moles(
            target, extra_stem=args.extra_stem
        )

        mel.rotomap.moles.save_image_moles(
            new_moles, target, extra_stem=args.extra_stem
        )


# -----------------------------------------------------------------------------
# Copyright (C) 2023 Angelos Evripiotis.
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