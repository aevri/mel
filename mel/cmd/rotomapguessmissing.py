"""Guess missing moles based on canonical moles common between images."""

import argparse
import pathlib

import mel.rotomap.moles
import mel.rotomap.relate


def _existing_file_path(string):
    """Argparse type for validating that a file exists."""
    path = pathlib.Path(string)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File does not exist: {string}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Path is not a file: {string}")
    return path


def setup_parser(parser):
    parser.add_argument(
        "SRC_JPG",
        type=_existing_file_path,
        help="Path to the source image with known mole locations.",
    )
    parser.add_argument(
        "TGT_JPG",
        type=_existing_file_path,
        help="Path to the target image to add missing moles to.",
    )


def process_args(args):
    src_path = args.SRC_JPG
    tgt_path = args.TGT_JPG

    try:
        src_moles = mel.rotomap.moles.load_image_moles(src_path)
        tgt_moles = mel.rotomap.moles.load_image_moles(tgt_path)
    except Exception as e:
        print(f"Error loading moles: {e}")
        return 1

    src_canonical_moles = [
        m for m in src_moles if m[mel.rotomap.moles.KEY_IS_CONFIRMED]
    ]
    tgt_canonical_moles = [
        m for m in tgt_moles if m[mel.rotomap.moles.KEY_IS_CONFIRMED]
    ]

    if not src_canonical_moles:
        print("Error: No canonical moles found in source image")
        return 1
    if not tgt_canonical_moles:
        print("Error: No canonical moles found in target image")
        return 1

    # Find canonical moles present in source but missing in target
    src_canonical_uuids = {m["uuid"] for m in src_canonical_moles}
    tgt_all_uuids = {m["uuid"] for m in tgt_moles}

    missing_uuids = src_canonical_uuids - tgt_all_uuids

    if not missing_uuids:
        print(
            "No missing moles to guess - all source canonical moles already present in target"
        )
        return 0

    print(f"Found {len(missing_uuids)} missing canonical moles to guess locations for")

    # Guess positions for missing moles using canonical moles as reference
    guessed_count = 0
    for missing_uuid in missing_uuids:
        guessed_pos = mel.rotomap.relate.guess_mole_pos(
            missing_uuid, src_canonical_moles, tgt_canonical_moles
        )

        if guessed_pos is not None:
            # Add the missing mole to target as non-canonical
            new_mole = {
                "uuid": missing_uuid,
                "x": int(guessed_pos[0]),
                "y": int(guessed_pos[1]),
                mel.rotomap.moles.KEY_IS_CONFIRMED: False,
            }
            tgt_moles.append(new_mole)
            guessed_count += 1
            print(
                f"Guessed position for mole {missing_uuid} at ({guessed_pos[0]}, {guessed_pos[1]})"
            )
        else:
            print(f"Could not guess position for mole {missing_uuid}")

    if guessed_count > 0:
        try:
            mel.rotomap.moles.save_image_moles(tgt_moles, tgt_path)
            print(f"Successfully added {guessed_count} guessed moles to {tgt_path}")
        except Exception as e:
            print(f"Error saving moles: {e}")
            return 1
    else:
        print("No moles could be guessed")

    return 0


# -----------------------------------------------------------------------------
# Copyright (C) 2025 Angelos Evripiotis.
# Generated with assistance from Claude Code.
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
