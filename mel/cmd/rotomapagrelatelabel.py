"""Relate moles across images and label them with letters and numbers."""

import json
import pathlib
import string
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np

import mel.lib.image
import mel.rotomap.mask
import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        "IMAGES",
        nargs="+",
        help="Paths to the images to process.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Directory to save the labeled images to. If not specified, "
        "modified versions will be placed in the same directories as originals "
        "with '_labeled' added to filenames.",
    )


def process_args(args):
    image_paths = [pathlib.Path(p) for p in args.IMAGES]

    # Check that all images exist
    for path in image_paths:
        if not path.exists():
            print(f"Error: Input image does not exist: {path}")
            return 1

    # Load all moles from the images
    all_moles = []
    for path in image_paths:
        try:
            moles = mel.rotomap.moles.load_image_moles(path)
            all_moles.append((path, moles))
        except Exception as e:
            print(f"Error loading moles for {path}: {e}")
            return 1

    # Generate pooled labels
    canonical_moles, non_canonical_moles = generate_pooled_labels(all_moles)

    # Process each image
    for path, moles in all_moles:
        try:
            # Create labeled image
            labeled_image = create_labeled_image(
                path, moles, canonical_moles, non_canonical_moles
            )

            # Determine output path
            output_path = get_output_path(path, args.output_dir)

            # Save the labeled image
            cv2.imwrite(str(output_path), labeled_image)
            print(f"Saved labeled image to {output_path}")

            # Save JSON file with mole data
            json_output_path = get_json_output_path(path, args.output_dir)
            save_moles_json(
                json_output_path, moles, canonical_moles, non_canonical_moles
            )
            print(f"Saved mole data to {json_output_path}")

        except Exception as e:
            print(f"Error processing {path}: {e}")
            return 1

    return 0


def generate_pooled_labels(
    all_moles: List[Tuple[pathlib.Path, List[Dict]]]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Generate unique labels for all moles across images.

    Args:
        all_moles: List of (path, moles) tuples

    Returns:
        Tuple containing:
        - Dictionary mapping canonical UUIDs to letter labels (A-Z, AA, AB, etc.)
        - Dictionary mapping non-canonical UUIDs to numeric labels (1, 2, etc.)
    """
    # Collect all canonical mole UUIDs
    canonical_uuids: Set[str] = set()
    non_canonical_uuids: Set[str] = set()

    for _, moles in all_moles:
        for mole in moles:
            uuid = mole["uuid"]
            if mole[mel.rotomap.moles.KEY_IS_CONFIRMED]:
                canonical_uuids.add(uuid)
            else:
                non_canonical_uuids.add(uuid)

    # Generate labels for canonical moles (A-Z, then AA, AB, etc.)
    canonical_labels = {}

    # Single letter labels (A-Z)
    alphabet = list(string.ascii_uppercase)

    # Generate labels for all canonical UUIDs
    sorted_canonical_uuids = sorted(canonical_uuids)
    for i, uuid in enumerate(sorted_canonical_uuids):
        if i < 26:
            # Single letter (A-Z)
            label = alphabet[i]
        else:
            # Double letter (AA, AB, etc.)
            first_letter_idx = (i // 26) - 1
            second_letter_idx = i % 26
            label = alphabet[first_letter_idx] + alphabet[second_letter_idx]

        canonical_labels[uuid] = label

    # Generate numeric labels for non-canonical moles
    non_canonical_labels = {}
    for i, uuid in enumerate(sorted(non_canonical_uuids)):
        non_canonical_labels[uuid] = str(i + 1)

    return canonical_labels, non_canonical_labels


def create_labeled_image(
    image_path: pathlib.Path,
    moles: List[Dict],
    canonical_labels: Dict[str, str],
    non_canonical_labels: Dict[str, str],
) -> np.ndarray:
    """Create a labeled image with moles marked and labeled.

    Args:
        image_path: Path to the original image
        moles: List of mole dictionaries
        canonical_labels: Dictionary mapping canonical UUIDs to letter labels
        non_canonical_labels: Dictionary mapping non-canonical UUIDs to numeric labels

    Returns:
        A new image with moles labeled
    """
    # Load the original image
    image = mel.lib.image.load_image(image_path)

    # Create output image
    labeled_image = image.copy()

    # Apply mask if available to green out the background
    mask = mel.rotomap.mask.load_or_none(image_path)
    if mask is not None:
        # Create a green background where mask is 0
        green_background = np.zeros_like(labeled_image)
        green_background[:, :, 1] = 255  # Set green channel to max

        # Apply mask - keep original image where mask is 1, use green where mask is 0
        mask_3channel = np.stack([mask, mask, mask], axis=2)
        labeled_image = np.where(
            mask_3channel > 0, labeled_image, green_background
        )

    # Draw circles and labels for each mole
    for mole in moles:
        uuid = mole["uuid"]
        x, y = int(mole["x"]), int(mole["y"])

        # Determine label and color based on whether mole is canonical
        if mole["is_uuid_canonical"]:
            label = canonical_labels[uuid]
            color = (0, 0, 255)  # Red for canonical moles
        else:
            label = non_canonical_labels.get(uuid, "?")
            color = (255, 0, 0)  # Blue for non-canonical moles

        # Draw a circle around the mole
        cv2.circle(labeled_image, (x, y), 15, color, 2)

        # Draw the label next to the mole
        cv2.putText(
            labeled_image,
            label,
            (x + 20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

    return labeled_image


def get_output_path(
    input_path: pathlib.Path, output_dir: str = None
) -> pathlib.Path:
    """Determine the output path for a labeled image.

    Args:
        input_path: Original image path
        output_dir: Optional output directory

    Returns:
        Path to save the labeled image
    """
    if output_dir:
        output_dir_path = pathlib.Path(output_dir)
        output_dir_path.mkdir(exist_ok=True, parents=True)
        return (
            output_dir_path / f"{input_path.stem}_labeled{input_path.suffix}"
        )
    else:
        return (
            input_path.parent / f"{input_path.stem}_labeled{input_path.suffix}"
        )


def get_json_output_path(
    input_path: pathlib.Path, output_dir: str = None
) -> pathlib.Path:
    """Determine the output path for a JSON file with mole data.

    Args:
        input_path: Original image path
        output_dir: Optional output directory

    Returns:
        Path to save the JSON file
    """
    if output_dir:
        output_dir_path = pathlib.Path(output_dir)
        output_dir_path.mkdir(exist_ok=True, parents=True)
        return output_dir_path / f"{input_path.stem}_moles.json"
    else:
        return input_path.parent / f"{input_path.stem}_moles.json"


def save_moles_json(
    json_path: pathlib.Path,
    moles: List[Dict],
    canonical_labels: Dict[str, str],
    non_canonical_labels: Dict[str, str],
) -> None:
    """Save mole data to a JSON file.

    Args:
        json_path: Path to save the JSON file
        moles: List of mole dictionaries
        canonical_labels: Dictionary mapping canonical UUIDs to letter labels
        non_canonical_labels: Dictionary mapping non-canonical UUIDs to numeric labels
    """
    mole_data = []

    for mole in moles:
        uuid = mole["uuid"]
        is_canonical = mole[mel.rotomap.moles.KEY_IS_CONFIRMED]

        label = (
            canonical_labels.get(uuid)
            if is_canonical
            else non_canonical_labels.get(uuid)
        )

        mole_info = {
            "identifier": label,
            "canonical": is_canonical,
            "x": int(mole["x"]),
            "y": int(mole["y"]),
        }

        mole_data.append(mole_info)

    with open(json_path, "w") as f:
        json.dump(mole_data, f, indent=2)


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
