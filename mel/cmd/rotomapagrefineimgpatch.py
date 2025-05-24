"""Refine detected mole coordinates using a VLM."""

import argparse
import json
import os
import pathlib
from typing import Dict, List

import anthropic
import cv2

import mel.lib.agent
import mel.lib.image
import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        "IMAGE",
        help="Path to the image patch created by ag-get-image-patch.",
    )
    parser.add_argument(
        "MOLES_JSON",
        help="Path to the JSON file containing the moles to refine.",
    )
    parser.add_argument(
        "--messages-json",
        help="Optional path to a JSON file containing previous messages from Claude.",
    )
    parser.add_argument(
        "--model",
        choices=[
            "claude-3-opus-20240229",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-haiku-20240307",
        ],
        default="claude-3-7-sonnet-20250219",
        help="Claude model to use for analysis (default: claude-3-7-sonnet-20250219).",
    )
    parser.add_argument(
        "--output-json",
        help="Path to save the refined moles JSON. If not provided, will use <IMAGE>.refined.json",
    )
    parser.add_argument(
        "--debug-output-prefix",
        help="Optional prefix for debug output images.",
    )


def process_args(args):
    image_path = pathlib.Path(args.IMAGE)
    if not image_path.exists():
        print(f"Error: Input image does not exist: {image_path}")
        return 1

    moles_json_path = pathlib.Path(args.MOLES_JSON)
    if not moles_json_path.exists():
        print(f"Error: Moles JSON file does not exist: {moles_json_path}")
        return 1

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "Error: Claude API key not provided. "
            "Please set the ANTHROPIC_API_KEY environment variable."
        )
        return 1

    # Load moles JSON
    try:
        with open(moles_json_path, "r") as f:
            moles = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to parse moles JSON file: {moles_json_path}")
        return 1

    print(f"Loaded {len(moles)} moles from {moles_json_path}")

    # Load previous messages if provided
    previous_messages = None
    if args.messages_json:
        messages_path = pathlib.Path(args.messages_json)
        if not messages_path.exists():
            print(
                f"Warning: Messages JSON file does not exist: {messages_path}"
            )
        else:
            try:
                with open(messages_path, "r") as f:
                    previous_messages = json.load(f)
                print(
                    f"Loaded {len(previous_messages)} messages from {messages_path}"
                )
            except json.JSONDecodeError:
                print(
                    f"Error: Failed to parse messages JSON file: {messages_path}"
                )
                return 1

    # Set output path
    output_path = args.output_json
    if not output_path:
        output_path = f"{image_path}.refined.json"

    # Refine mole coordinates
    print(f"Refining mole coordinates using {args.model}...")
    refined_moles = mel.lib.agent.refine_mole_coordinates(
        image_path,
        moles,
        api_key,
        args.model,
        previous_messages,
        args.debug_output_prefix,
    )

    # Compare original and refined coordinates
    print("\nCoordinate changes:")
    for i, (orig, refined) in enumerate(zip(moles, refined_moles)):
        orig_x, orig_y = orig["x"], orig["y"]
        new_x, new_y = refined["x"], refined["y"]
        dx, dy = new_x - orig_x, new_y - orig_y
        distance = (dx**2 + dy**2) ** 0.5

        if distance > 0:
            print(
                f"  Mole {orig.get('id', i+1)}: ({orig_x}, {orig_y}) -> ({new_x}, {new_y}) [moved {distance:.1f}px]"
            )
        else:
            print(
                f"  Mole {orig.get('id', i+1)}: No change ({orig_x}, {orig_y})"
            )

    # Save refined moles
    with open(output_path, "w") as f:
        json.dump(refined_moles, f, indent=2)

    print(f"\nSaved refined moles to {output_path}")

    # Create and save final annotated image
    # Load the original image
    original_image = mel.lib.image.load_image(image_path)

    # Create a copy for annotation
    final_image = original_image.copy()

    # Draw original moles in blue, refined in red
    for orig, refined in zip(moles, refined_moles):
        # Original position (blue)
        cv2.circle(final_image, (orig["x"], orig["y"]), 10, (255, 0, 0), 2)

        # Refined position (red)
        cv2.circle(
            final_image, (refined["x"], refined["y"]), 10, (0, 0, 255), 2
        )

        # Add ID label near the refined position
        mole_id = str(refined.get("id", ""))
        if mole_id:
            cv2.putText(
                final_image,
                mole_id,
                (refined["x"] + 15, refined["y"]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

    # Save the final annotated image
    final_image_path = f"{image_path}.refined.jpg"
    cv2.imwrite(final_image_path, final_image)
    print(f"Saved annotated image to {final_image_path}")

    return 0


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
