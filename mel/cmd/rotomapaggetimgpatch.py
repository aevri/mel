"""Extract a patch of an image around a specific mole for use with VLMs."""

import argparse
import json
import pathlib

import cv2

import mel.lib.image
import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        "IMAGE",
        help="Path to the rotomap image.",
    )
    parser.add_argument(
        "OUTPUT",
        help="Path to save the output image patch.",
    )
    parser.add_argument(
        "--mole",
        required=True,
        help="UUID of the mole to extract a patch for.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=100,
        help="Margin around the mole in pixels (default: 100).",
    )


def process_args(args):
    image_path = pathlib.Path(args.IMAGE)
    output_path = pathlib.Path(args.OUTPUT)

    if not image_path.exists():
        print(f"Error: Input image does not exist: {image_path}")
        return 1

    try:
        # Load the image and mole data
        image = mel.lib.image.load_image(image_path)
        moles = mel.rotomap.moles.load_image_moles(image_path)

        # Find the specified mole
        mole_index = mel.rotomap.moles.uuid_mole_index(moles, args.mole)
        if mole_index is None:
            print(
                f"Error: Mole with UUID {args.mole} not found in {image_path}"
            )
            return 1

        mole = moles[mole_index]
        mole_x, mole_y = mole["x"], mole["y"]

        # Calculate patch boundaries with margin
        height, width = image.shape[:2]

        left = max(0, mole_x - args.margin)
        top = max(0, mole_y - args.margin)
        right = min(width, mole_x + args.margin)
        bottom = min(height, mole_y + args.margin)

        # Extract the patch
        patch = image[top:bottom, left:right]

        # Save the patch
        if not cv2.imwrite(str(output_path), patch):
            print(f"Error: Failed to save patch to {output_path}")
            return 1

        # Create JSON with moles information in the patch
        moles_in_patch = []
        for m in moles:
            m_x, m_y = m["x"], m["y"]
            if left <= m_x < right and top <= m_y < bottom:
                # Adjust coordinates to be relative to the patch
                patch_mole = m.copy()
                patch_mole["x"] = m_x - left
                patch_mole["y"] = m_y - top
                moles_in_patch.append(patch_mole)

        # Save JSON metadata using the same format as other mole json files
        json_path = f"{output_path}.json"
        mel.rotomap.moles.save_json(json_path, moles_in_patch)

        print(f"Extracted patch around mole {args.mole} to {output_path}")
        print(
            f"Saved metadata with {len(moles_in_patch)} moles to {json_path}"
        )

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


# -----------------------------------------------------------------------------
# Copyright (C) 2025 Angelos Evripiotis.
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
