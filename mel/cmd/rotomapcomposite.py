"""Composite rotomap images by filling mask-hidden areas with a solid color."""

import argparse
import pathlib

import numpy as np

import mel.cmd.error
import mel.lib.image
import mel.rotomap.mask


def setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "IMAGE",
        nargs="+",
        help="JPEG image files to composite with their masks.",
    )
    parser.add_argument(
        "--fill-color",
        type=str,
        default="0,0,0",
        help="BGR fill color for masked-out areas, as B,G,R (default: 0,0,0).",
    )


def _parse_fill_color(fill_color_str: str) -> tuple[int, int, int]:
    """Parse a comma-separated BGR color string into a tuple of three ints."""
    parts = fill_color_str.split(",")
    if len(parts) != 3:
        msg = f"Fill color must have 3 comma-separated values, got: {fill_color_str}"
        raise ValueError(msg)
    values = tuple(int(p.strip()) for p in parts)
    for v in values:
        if not 0 <= v <= 255:
            msg = f"Fill color values must be 0-255, got: {v}"
            raise ValueError(msg)
    return values[0], values[1], values[2]


def process_args(args: argparse.Namespace) -> None:
    try:
        fill_color = _parse_fill_color(args.fill_color)
    except ValueError as err:
        msg = str(err)
        raise mel.cmd.error.UsageError(msg) from err

    for image_path in args.IMAGE:
        if not image_path.endswith(".jpg"):
            msg = f"Not a JPEG file: {image_path}"
            raise mel.cmd.error.UsageError(msg)

        if not pathlib.Path(image_path).exists():
            msg = f"File not found: {image_path}"
            raise mel.cmd.error.UsageError(msg)

        mask = mel.rotomap.mask.load_or_none(image_path)
        if mask is None:
            print(f"No mask found, skipping: {image_path}")
            continue

        image = mel.lib.image.load_image(image_path)

        keep_mask = mask > 127
        for c in range(3):
            image[:, :, c] = np.where(keep_mask, image[:, :, c], fill_color[c])

        mel.lib.image.save_image(image, image_path)
        print(f"Composited: {image_path}")


# -----------------------------------------------------------------------------
# Copyright (C) 2026 Angelos Evripiotis.
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
