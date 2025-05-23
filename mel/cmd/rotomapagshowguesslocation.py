"""Show guess location for a mole in target image with purple grid overlay."""

import pathlib
import string
from typing import Dict, List, Set, Tuple, Optional

import cv2
import numpy as np

import mel.lib.image
import mel.rotomap.mask
import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        "SRC_JPG",
        help="Source image containing the mole with UUID.",
    )
    parser.add_argument(
        "TGT_JPG",
        help="Target image where the mole location should be guessed.",
    )
    parser.add_argument(
        "OUTPUT_SRC_JPG",
        help="Output path for labeled source image.",
    )
    parser.add_argument(
        "OUTPUT_TGT_JPG",
        help="Output path for labeled target image with guessed location.",
    )
    parser.add_argument(
        "UUID",
        help="UUID of the mole to locate in the target image.",
    )


def process_args(args):
    src_path = pathlib.Path(args.SRC_JPG)
    tgt_path = pathlib.Path(args.TGT_JPG)
    output_src_path = pathlib.Path(args.OUTPUT_SRC_JPG)
    output_tgt_path = pathlib.Path(args.OUTPUT_TGT_JPG)
    target_uuid = args.UUID

    # Check that input images exist
    if not src_path.exists():
        print(f"Error: Source image does not exist: {src_path}")
        return 1
    if not tgt_path.exists():
        print(f"Error: Target image does not exist: {tgt_path}")
        return 1

    # Load moles from both images
    try:
        src_moles = mel.rotomap.moles.load_image_moles(src_path)
        tgt_moles = mel.rotomap.moles.load_image_moles(tgt_path)
    except Exception as e:
        print(f"Error loading moles: {e}")
        return 1

    # Find the target mole in source image
    target_mole = None
    for mole in src_moles:
        if mole["uuid"] == target_uuid:
            target_mole = mole
            break

    if target_mole is None:
        print(f"Error: Mole with UUID {target_uuid} not found in source image")
        return 1

    # Find overlapping canonical moles between the two images
    overlapping_moles = find_overlapping_canonical_moles(src_moles, tgt_moles)

    if len(overlapping_moles) < 2:
        print(
            f"Error: Need at least 2 overlapping canonical moles, found {len(overlapping_moles)}"
        )
        return 1

    # Guess the location of the target mole in the target image
    guessed_x, guessed_y = guess_mole_location(
        target_mole, overlapping_moles, src_moles, tgt_moles
    )

    print(
        f"Guessed location for UUID {target_uuid}: ({guessed_x}, {guessed_y})"
    )

    # Generate pooled labels for all moles
    all_moles = [(src_path, src_moles), (tgt_path, tgt_moles)]
    canonical_moles, non_canonical_moles = generate_pooled_labels(all_moles)

    # Create labeled images with grid overlay
    try:
        # Create source image with labels and grid
        src_labeled = create_labeled_image_with_grid(
            src_path, src_moles, canonical_moles, non_canonical_moles
        )
        cv2.imwrite(str(output_src_path), src_labeled)
        print(f"Saved labeled source image to {output_src_path}")

        # Create target image with labels, grid, and guessed location
        tgt_labeled = create_labeled_image_with_grid(
            tgt_path,
            tgt_moles,
            canonical_moles,
            non_canonical_moles,
            guessed_location=(guessed_x, guessed_y),
        )
        cv2.imwrite(str(output_tgt_path), tgt_labeled)
        print(f"Saved labeled target image to {output_tgt_path}")

    except Exception as e:
        print(f"Error creating labeled images: {e}")
        return 1

    return 0


def find_overlapping_canonical_moles(
    src_moles: List[Dict], tgt_moles: List[Dict]
) -> List[str]:
    """Find UUIDs of canonical moles present in both images."""
    src_canonical_uuids = {
        mole["uuid"]
        for mole in src_moles
        if mole[mel.rotomap.moles.KEY_IS_CONFIRMED]
    }
    tgt_canonical_uuids = {
        mole["uuid"]
        for mole in tgt_moles
        if mole[mel.rotomap.moles.KEY_IS_CONFIRMED]
    }

    return list(src_canonical_uuids.intersection(tgt_canonical_uuids))


def guess_mole_location(
    target_mole: Dict,
    overlapping_uuids: List[str],
    src_moles: List[Dict],
    tgt_moles: List[Dict],
) -> Tuple[int, int]:
    """Guess the location of target_mole in target image using overlapping reference moles."""
    # Get positions of overlapping moles in both images
    src_ref_positions = {}
    tgt_ref_positions = {}

    for mole in src_moles:
        if mole["uuid"] in overlapping_uuids:
            src_ref_positions[mole["uuid"]] = (mole["x"], mole["y"])

    for mole in tgt_moles:
        if mole["uuid"] in overlapping_uuids:
            tgt_ref_positions[mole["uuid"]] = (mole["x"], mole["y"])

    # Use a simple transformation based on the closest reference moles
    # For now, use the centroid of reference moles and apply offset
    if len(overlapping_uuids) >= 2:
        # Calculate centroid of reference moles in both images
        src_centroid_x = sum(
            src_ref_positions[uuid][0] for uuid in overlapping_uuids
        ) / len(overlapping_uuids)
        src_centroid_y = sum(
            src_ref_positions[uuid][1] for uuid in overlapping_uuids
        ) / len(overlapping_uuids)

        tgt_centroid_x = sum(
            tgt_ref_positions[uuid][0] for uuid in overlapping_uuids
        ) / len(overlapping_uuids)
        tgt_centroid_y = sum(
            tgt_ref_positions[uuid][1] for uuid in overlapping_uuids
        ) / len(overlapping_uuids)

        # Calculate offset of target mole from source centroid
        offset_x = target_mole["x"] - src_centroid_x
        offset_y = target_mole["y"] - src_centroid_y

        # Apply offset to target centroid
        guessed_x = int(tgt_centroid_x + offset_x)
        guessed_y = int(tgt_centroid_y + offset_y)

        return guessed_x, guessed_y

    # Fallback: use single reference mole
    uuid = overlapping_uuids[0]
    src_pos = src_ref_positions[uuid]
    tgt_pos = tgt_ref_positions[uuid]

    # Apply the displacement from reference mole
    offset_x = target_mole["x"] - src_pos[0]
    offset_y = target_mole["y"] - src_pos[1]

    guessed_x = int(tgt_pos[0] + offset_x)
    guessed_y = int(tgt_pos[1] + offset_y)

    return guessed_x, guessed_y


def generate_pooled_labels(
    all_moles: List[Tuple[pathlib.Path, List[Dict]]]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Generate unique labels for all moles across images."""
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


def create_labeled_image_with_grid(
    image_path: pathlib.Path,
    moles: List[Dict],
    canonical_labels: Dict[str, str],
    non_canonical_labels: Dict[str, str],
    guessed_location: Optional[Tuple[int, int]] = None,
    zoom_factor: float = 2.0,
    zoom_size: int = 800,
) -> np.ndarray:
    """Create a labeled image with moles marked, labeled, and purple grid overlay."""
    # Load the original image
    image = mel.lib.image.load_image(image_path)
    original_height, original_width = image.shape[:2]

    # Initialize crop coordinates
    crop_coords = None

    # If we have a guessed location, crop around it for zooming
    if guessed_location is not None:
        guess_x, guess_y = guessed_location

        # Calculate crop area around guessed location
        crop_size = zoom_size // 2
        start_x = max(0, guess_x - crop_size)
        start_y = max(0, guess_y - crop_size)
        end_x = min(original_width, guess_x + crop_size)
        end_y = min(original_height, guess_y + crop_size)

        crop_coords = (start_x, start_y, end_x, end_y)

        # Crop the image
        image = image[start_y:end_y, start_x:end_x]

        # Adjust mole positions for the crop
        adjusted_moles = []
        for mole in moles:
            new_x = mole["x"] - start_x
            new_y = mole["y"] - start_y

            # Only include moles that are within the cropped area
            if 0 <= new_x < (end_x - start_x) and 0 <= new_y < (
                end_y - start_y
            ):
                adjusted_mole = mole.copy()
                adjusted_mole["x"] = new_x
                adjusted_mole["y"] = new_y
                adjusted_moles.append(adjusted_mole)

        moles = adjusted_moles

        # Adjust guessed location coordinates too
        guessed_location = (guess_x - start_x, guess_y - start_y)

    labeled_image = image.copy()

    # Apply mask if available to green out the background
    mask = mel.rotomap.mask.load_or_none(image_path)
    if mask is not None:
        # If we cropped the image, crop the mask too
        if crop_coords is not None:
            start_x, start_y, end_x, end_y = crop_coords
            mask = mask[start_y:end_y, start_x:end_x]

        # Create a green background where mask is 0
        green_background = np.zeros_like(labeled_image)
        green_background[:, :, 1] = 255  # Set green channel to max

        # Apply mask - keep original image where mask is 1, use green where mask is 0
        mask_3channel = np.stack([mask, mask, mask], axis=2)
        labeled_image = np.where(
            mask_3channel > 0, labeled_image, green_background
        )

    # Draw purple grid
    height, width = labeled_image.shape[:2]
    grid_color = (128, 0, 128)  # Purple
    grid_thickness = max(1, width // 400)  # Scale thickness with image size

    # Draw grid lines - 10x10 grid
    grid_rows = 10
    grid_cols = 10

    # Vertical lines
    for i in range(grid_cols + 1):
        x = int(i * width / grid_cols)
        cv2.line(
            labeled_image, (x, 0), (x, height), grid_color, grid_thickness
        )

    # Horizontal lines
    for i in range(grid_rows + 1):
        y = int(i * height / grid_rows)
        cv2.line(labeled_image, (0, y), (width, y), grid_color, grid_thickness)

    # Draw grid labels at intersections
    font_scale = max(0.3, min(0.7, width / 1000))
    font_thickness = max(1, int(font_scale * 2))

    for row in range(grid_rows):
        for col in range(grid_cols):
            # Calculate intersection position
            x = int((col + 0.5) * width / grid_cols)
            y = int((row + 0.5) * height / grid_rows)

            # Create label (AA, AB, AC, etc.)
            row_letter = string.ascii_uppercase[row % 26]
            col_letter = string.ascii_uppercase[col % 26]
            grid_label = row_letter + col_letter

            # Get text size to center it
            (text_width, text_height), baseline = cv2.getTextSize(
                grid_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thickness,
            )

            # Draw label centered at intersection
            cv2.putText(
                labeled_image,
                grid_label,
                (x - text_width // 2, y + text_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                grid_color,
                font_thickness,
            )

    # Track label positions to avoid overlaps
    label_positions = set()

    # Calculate circle size based on image size
    circle_radius = int(max(5, min(15, width / 40)))

    # Draw circles and labels for each mole
    for mole in moles:
        uuid = mole["uuid"]
        x, y = int(mole["x"]), int(mole["y"])

        # Skip if outside the image bounds
        if x < 0 or y < 0 or x >= width or y >= height:
            continue

        # Determine label and color based on whether mole is canonical
        if mole["is_uuid_canonical"]:
            label = canonical_labels[uuid]
            color = (0, 0, 255)  # Red for canonical moles
        else:
            label = non_canonical_labels.get(uuid, "?")
            color = (255, 0, 0)  # Blue for non-canonical moles

        # Draw a circle around the mole
        cv2.circle(labeled_image, (x, y), circle_radius, color, 2)

        # Calculate label position
        label_x = x + circle_radius + 5
        label_y = y

        # Make sure label is within image bounds
        if label_x >= width:
            label_x = x - circle_radius - 20

        # Check for overlaps with existing labels
        label_key = (label_x, label_y)
        if label_key not in label_positions:
            label_positions.add(label_key)

            # Calculate font scale based on image size
            font_scale = max(0.4, min(0.9, width / 800))

            # Draw the label next to the mole
            cv2.putText(
                labeled_image,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                max(1, int(1.5)),
            )

    # Draw guessed location if provided
    if guessed_location is not None:
        guess_x, guess_y = guessed_location
        if 0 <= guess_x < width and 0 <= guess_y < height:
            # Draw a larger yellow circle for the guessed location
            cv2.circle(
                labeled_image,
                (guess_x, guess_y),
                circle_radius + 5,
                (0, 255, 255),
                3,
            )
            # Draw crosshairs
            cv2.line(
                labeled_image,
                (guess_x - 15, guess_y),
                (guess_x + 15, guess_y),
                (0, 255, 255),
                2,
            )
            cv2.line(
                labeled_image,
                (guess_x, guess_y - 15),
                (guess_x, guess_y + 15),
                (0, 255, 255),
                2,
            )

    return labeled_image


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
