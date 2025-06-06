"""Merge data to data in an 'extra stem' namespace."""

import mel.rotomap.automark
import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument("EXTRA_STEM", help="The 'extra stem' namespace to merge to.")
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
        "--error-distance",
        default=0,
        type=int,
        help="Radius to merge moles within.",
    )
    parser.add_argument(
        "--compare-uuids",
        action="store_true",
        help="Also compare UUIDs for matching.",
    )


def process_args(args):
    total_matched = 0
    total_missing = 0
    total_added = 0
    for path in args.IMAGES:
        from_moles = mel.rotomap.moles.load_image_moles(path)
        to_moles = mel.rotomap.moles.load_image_moles(path, extra_stem=args.EXTRA_STEM)
        (
            match_uuids,
            _missing_uuids,
            added_uuids,
        ) = mel.rotomap.automark.match_moles_by_pos(
            from_moles, to_moles, args.error_distance
        )
        if args.verbose:
            print(
                path,
                ":",
                len(match_uuids),
                len(_missing_uuids),
                len(added_uuids),
            )
        if args.compare_uuids:
            total_missing += len(_missing_uuids)
            total_added += len(added_uuids)
            for from_uuid, to_uuid in match_uuids:
                if from_uuid == to_uuid:
                    total_matched += 1
                else:
                    total_missing += 1
                    total_added += 1
        else:
            total_matched += len(match_uuids)
            total_missing += len(_missing_uuids)
            total_added += len(added_uuids)
    print("matched:", total_matched)
    print("missing:", total_missing)
    print("added:", total_added)

    # Calculate recall
    if total_matched + total_missing == 0:
        recall = "N/A (no relevant items)"
    else:
        recall = f"{total_matched / (total_matched + total_missing):.0%}"

    # Calculate precision
    if total_matched + total_added == 0:
        precision = "N/A (no retrieved items)"
    else:
        precision = f"{total_matched / (total_matched + total_added):.0%}"

    print(f"Recall: {recall}")
    print(f"Precision: {precision}")


# -----------------------------------------------------------------------------
# Copyright (C) 2022 Angelos Evripiotis.
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
