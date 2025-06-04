"""Resize images and their associated mole files, masks, and ellipses."""

import json
import os

import cv2

import mel.lib.image
import mel.rotomap.mask
import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Target width in pixels for resized images.",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Target height in pixels for resized images.",
    )
    parser.add_argument(
        "IMAGE",
        nargs="+",
        help="JPEG image files to resize along with their associated files.",
    )


def process_args(args):
    target_width = args.width
    target_height = args.height

    for image_path in args.IMAGE:
        if not image_path.endswith(".jpg"):
            print(f"Skipping non-JPEG file: {image_path}")
            continue

        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            continue

        print(f"Resizing {image_path}...")

        # Load original image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            continue

        orig_height, orig_width = image.shape[:2]
        scale_x = target_width / orig_width
        scale_y = target_height / orig_height

        # Resize the main image
        resized_image = cv2.resize(
            image, (target_width, target_height), interpolation=cv2.INTER_CUBIC
        )
        mel.lib.image.save_image(resized_image, image_path)

        # Resize mole coordinates
        _resize_mole_files(image_path, scale_x, scale_y)

        # Resize mask if it exists
        _resize_mask_file(image_path, target_width, target_height)

        # Resize ellipse metadata
        _resize_ellipse_metadata(image_path, scale_x, scale_y)


def _resize_mole_files(image_path, scale_x, scale_y):
    """Resize mole coordinates in .json files."""
    mole_file_path = image_path + ".json"
    if os.path.exists(mole_file_path):
        with open(mole_file_path) as f:
            moles = json.load(f)

        for mole in moles:
            mole["x"] = int(mole["x"] * scale_x)
            mole["y"] = int(mole["y"] * scale_y)

        mel.rotomap.moles.save_json(mole_file_path, moles)


def _resize_mask_file(image_path, target_width, target_height):
    """Resize mask file if it exists."""
    mask_path = mel.rotomap.mask.path(image_path)
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is not None:
            resized_mask = cv2.resize(
                mask,
                (target_width, target_height),
                interpolation=cv2.INTER_CUBIC,
            )
            mel.lib.image.save_image(resized_mask, mask_path)


def _resize_ellipse_metadata(image_path, scale_x, scale_y):
    """Resize ellipse coordinates in .meta.json file."""
    meta_path = image_path + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

        if "ellipse" in metadata:
            ellipse = metadata["ellipse"]
            # Ellipse format: [[centerX, centerY], [width, height], rotation]
            center = ellipse[0]
            extents = ellipse[1]
            rotation = ellipse[2]

            # Scale center coordinates
            center[0] *= scale_x
            center[1] *= scale_y

            # Scale ellipse dimensions
            extents[0] *= scale_x
            extents[1] *= scale_y

            # Rotation remains unchanged
            metadata["ellipse"] = [center, extents, rotation]

            mel.rotomap.moles.save_json(meta_path, metadata)


# -----------------------------------------------------------------------------
# Copyright (C) 2025 Angelos Evripiotis.
# Generated with assistance from Claude Code and Cursor.
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
