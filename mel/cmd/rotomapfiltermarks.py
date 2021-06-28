"""Automatically remove marked regions that are probably not moles."""

import sys

import mel.lib.fs
import mel.rotomap.filtermarks
import mel.rotomap.moles


def setup_parser(parser):

    parser.add_argument(
        "FRAMES",
        nargs="+",
        help="Path to rotomap or image to filter.",
    )

    parser.add_argument(
        "--classifier-dir",
        help="Path to the classifier base dir, relative to the root of the "
        "mel repo. Names are relative to this."
        f"Defaults to {mel.lib.fs.DEFAULT_CLASSIFIER_PATH}.",
        default=mel.lib.fs.DEFAULT_CLASSIFIER_PATH,
    )

    parser.add_argument(
        "--model-name",
        help="Name of the model to use, relative to the classifier dir. "
        f"Defaults to {mel.lib.fs.DEFAULT_MOLE_MARK_MODEL_NAME}.",
        default="filtermarks.pth",
    )

    parser.add_argument(
        "--dataconfig-name",
        help="Name of the dataconfig to use, relative to the classifier dir. "
        f"Defaults to {mel.lib.fs.DEFAULT_MOLE_MARK_DATACONFIG_NAME}.",
        default="filtermarks.json",
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Don't save results of processing, just print.",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print information about the processing.",
    )

    parser.add_argument(
        "--include-canonical",
        action="store_true",
        help="Don't excluded canonical moles from processing.",
    )

    parser.add_argument(
        "--softmax-threshold",
        type=float,
        default=0.5,
        help="Use this threshold on softmax",
    )

    parser.add_argument(
        "--extra-stem",
        help="Add an extra bit to the filename stem, e.g. '0.jpg.EXTRA.json'.",
    )


def process_args(args):
    try:
        melroot = mel.lib.fs.find_melroot()
    except mel.lib.fs.NoMelrootError:
        print("Not in a mel repo, could not find melroot", file=sys.stderr)
        return 1

    if args.verbose:
        print("Initialising classifier .. ", end="", file=sys.stderr)
    classifier_path = melroot / args.classifier_dir
    is_mole = mel.rotomap.filtermarks.make_is_mole_func(
        classifier_path,
        # args.dataconfig_name,
        args.model_name,
        args.softmax_threshold,
    )
    if args.verbose:
        print("done", file=sys.stderr)

    image_moles_iter = (
        (
            image_path,
            mel.rotomap.moles.load_image_moles(
                image_path, extra_stem=args.extra_stem
            ),
        )
        for image_path in args.FRAMES
    )
    for image_path, moles in image_moles_iter:
        if args.verbose:
            print(image_path, file=sys.stderr)
        image, _ = mel.rotomap.filtermarks.open_image_for_classifier(
            image_path
        )
        try:
            filtered_moles = mel.rotomap.filtermarks.filter_marks(
                is_mole, image, moles, args.include_canonical
            )
        except Exception as e:
            raise Exception(
                "Error while processing {}".format(image_path)
            ) from e

        num_filtered = len(moles) - len(filtered_moles)
        if args.verbose:
            print(f"Filtered {num_filtered} unlikely moles.", file=sys.stderr)

        if not args.dry_run and num_filtered:
            mel.rotomap.moles.save_image_moles(
                filtered_moles, image_path, extra_stem=args.extra_stem
            )


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
