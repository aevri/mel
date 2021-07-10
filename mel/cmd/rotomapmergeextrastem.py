"""Merge data to data in an 'extra stem' namespace."""

import copy

import mel.rotomap.automark
import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        "EXTRA_STEM", help="The 'extra stem' namespace to merge to."
    )
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


def process_args(args):
    for path in args.IMAGES:
        if args.verbose:
            print(path)

        from_moles = mel.rotomap.moles.load_image_moles(path)
        to_moles = mel.rotomap.moles.load_image_moles(
            path, extra_stem=args.EXTRA_STEM
        )

        moles = _merge(
            from_moles=from_moles,
            to_moles=to_moles,
            error_distance=args.error_distance,
        )

        mel.rotomap.moles.save_image_moles(
            moles, path, extra_stem=args.EXTRA_STEM
        )


def _merge(from_moles, to_moles, error_distance):

    (
        match_uuids,
        _missing_uuids,
        added_uuids,
    ) = mel.rotomap.automark.match_moles_by_pos(
        from_moles, to_moles, error_distance
    )

    old_to_new_uuids = {
        to_uuid: from_uuid for from_uuid, to_uuid in match_uuids
    }

    results = []
    for to_m in to_moles:
        old_uuid = to_m["uuid"]
        new_uuid = old_to_new_uuids.get(old_uuid, None)
        mole = copy.deepcopy(to_m)
        if new_uuid is None:
            mole["uuid"] = mel.rotomap.moles.make_new_uuid()
            mole[mel.rotomap.moles.KEY_IS_CONFIRMED] = False
        else:
            mole["uuid"] = new_uuid
            mole[mel.rotomap.moles.KEY_IS_CONFIRMED] = True
        results.append(mole)

    return results


# -----------------------------------------------------------------------------
# Copyright (C) 2021 Angelos Evripiotis.
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
