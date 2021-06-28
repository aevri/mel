"""Automatically mark moles on rotomap images."""

import mel.lib.image
import mel.rotomap.automark
import mel.rotomap.detectmoles
import mel.rotomap.mask
import mel.rotomap.moles


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
    parser.add_argument(
        "--only-merge",
        action="store_true",
        help="Don't add new moles, only merge in updates to existing ones.",
    )
    parser.add_argument(
        "--error-distance",
        default=0,
        type=int,
        help="Radius to merge moles within.",
    )
    parser.add_argument(
        "--extra-stem",
        help="Add an extra bit to the filename stem, e.g. '0.jpg.EXTRA.json'.",
    )


def process_args(args):
    for path in args.IMAGES:
        if args.verbose:
            print(path)
        image = mel.lib.image.load_image(path)
        mask = mel.rotomap.mask.load(path)
        guessed_moles = mel.rotomap.detectmoles.moles(image, mask)
        loaded_moles = mel.rotomap.moles.load_image_moles(path)

        moles = mel.rotomap.automark.merge_in_radiuses(
            loaded_moles,
            radii_sources=guessed_moles,
            error_distance=args.error_distance,
            only_merge=args.only_merge,
        )

        mel.rotomap.moles.save_image_moles(
            moles, path, extra_stem=args.extra_stem
        )


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
