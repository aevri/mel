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
    parser.add_argument(
        "--output-width-height",
        type=str,
        help="Output image dimensions as 'width,height'. Images will be scaled "
        "to fit these dimensions while maintaining aspect ratio.",
    )
    parser.add_argument(
        "--zoom",
        type=str,
        help="UUID,SCALE where UUID is the mole UUID to center on and SCALE is "
        "the zoom factor. SCALE>1 means zoom out (e.g. 2 = half size).",
    )


def process_args(args):
    image_paths = [pathlib.Path(p) for p in args.IMAGES]

    # Check that all images exist
    for path in image_paths:
        if not path.exists():
            print(f"Error: Input image does not exist: {path}")
            return 1

    # Parse output dimensions if provided
    output_width = None
    output_height = None
    if args.output_width_height:
        try:
            width_height = args.output_width_height.split(",")
            if len(width_height) != 2:
                print(
                    "Error: --output-width-height must be in format 'width,height'"
                )
                return 1
            output_width = int(width_height[0])
            output_height = int(width_height[1])
            print(f"Output dimensions set to {output_width}x{output_height}")
        except ValueError:
            print("Error: --output-width-height must contain valid integers")
            return 1

    # Parse zoom parameters if provided
    zoom_uuid = None
    zoom_scale = None
    if args.zoom:
        try:
            zoom_params = args.zoom.split(",")
            if len(zoom_params) != 2:
                print("Error: --zoom must be in format 'UUID,SCALE'")
                return 1
            zoom_uuid = zoom_params[0]
            zoom_scale = float(zoom_params[1])
            print(f"Zoom set to UUID={zoom_uuid}, scale={zoom_scale}")
        except ValueError:
            print("Error: --zoom SCALE must be a valid number")
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
            # Create labeled image with optional resizing and zooming
            labeled_image = create_labeled_image(
                path,
                moles,
                canonical_moles,
                non_canonical_moles,
                output_width,
                output_height,
                zoom_uuid,
                zoom_scale,
            )

            # Determine output path
            output_path = get_output_path(path, args.output_dir)

            # Save the labeled image
            cv2.imwrite(str(output_path), labeled_image)
            print(f"Saved labeled image to {output_path}")

            # Save JSON file with mole data - use original mole positions, not scaled ones
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
    output_width: int = None,
    output_height: int = None,
    zoom_uuid: str = None,
    zoom_scale: float = None,
) -> np.ndarray:
    """Create a labeled image with moles marked and labeled.

    Args:
        image_path: Path to the original image
        moles: List of mole dictionaries
        canonical_labels: Dictionary mapping canonical UUIDs to letter labels
        non_canonical_labels: Dictionary mapping non-canonical UUIDs to numeric labels
        output_width: Optional width for the output image
        output_height: Optional height for the output image
        zoom_uuid: Optional UUID of the mole to center the image on
        zoom_scale: Optional scale factor for zooming (>1 means zoom out)

    Returns:
        A new image with moles labeled
    """
    # Load the original image
    image = mel.lib.image.load_image(image_path)

    # Get original dimensions
    original_height, original_width = image.shape[:2]

    # Find the center mole if zoom_uuid is specified
    center_x, center_y = None, None
    if zoom_uuid is not None:
        for mole in moles:
            if mole["uuid"] == zoom_uuid:
                center_x, center_y = int(mole["x"]), int(mole["y"])
                break

        if center_x is None:
            print(
                f"Warning: Mole with UUID {zoom_uuid} not found in image {image_path}"
            )
            # Default to center of image if mole not found
            center_x, center_y = original_width // 2, original_height // 2

    # Apply fixed zoom scale if specified
    if zoom_scale is not None:
        scale = 1.0 / zoom_scale  # Convert zoom scale to resize scale
    else:
        scale = 1.0

    # Calculate output dimensions for cropping (if both zoom and output dimensions specified)
    crop_to_output = False
    if output_width is not None and output_height is not None:
        if zoom_uuid is not None:
            # If both zoom and output dimensions specified, we'll crop to output dimensions
            crop_to_output = True
        else:
            # If only output dimensions specified (no zoom), calculate scale to fit
            scale_x = output_width / original_width
            scale_y = output_height / original_height
            if scale_x < scale_y:
                scale = scale_x  # Maintain aspect ratio
            else:
                scale = scale_y  # Maintain aspect ratio

    # Resize image before annotation if scale isn't 1.0
    if scale != 1.0:
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        image = cv2.resize(image, (new_width, new_height))

        # Also scale the center point if we're zooming on a mole
        if center_x is not None:
            center_x = int(center_x * scale)
            center_y = int(center_y * scale)

    # Create base labeled image
    labeled_image = image.copy()

    # Apply mask if available to green out the background
    mask = mel.rotomap.mask.load_or_none(image_path)
    if mask is not None:
        # Scale mask if necessary
        if scale != 1.0:
            mask = cv2.resize(
                mask,
                (labeled_image.shape[1], labeled_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Create a green background where mask is 0
        green_background = np.zeros_like(labeled_image)
        green_background[:, :, 1] = 255  # Set green channel to max

        # Apply mask - keep original image where mask is 1, use green where mask is 0
        mask_3channel = np.stack([mask, mask, mask], axis=2)
        labeled_image = np.where(
            mask_3channel > 0, labeled_image, green_background
        )

    # Prepare for centering and/or cropping
    if (
        center_x is not None
        and crop_to_output
        and output_width is not None
        and output_height is not None
    ):
        # Calculate the crop rectangle
        current_height, current_width = labeled_image.shape[:2]

        # Calculate the top-left corner of the crop area
        crop_start_x = max(0, center_x - output_width // 2)
        crop_start_y = max(0, center_y - output_height // 2)

        # Make sure the crop area doesn't go beyond the image boundaries
        if crop_start_x + output_width > current_width:
            crop_start_x = current_width - output_width
        if crop_start_y + output_height > current_height:
            crop_start_y = current_height - output_height

        # Ensure we don't have negative values (can happen with small images)
        crop_start_x = max(0, crop_start_x)
        crop_start_y = max(0, crop_start_y)
    else:
        # No centering/cropping
        crop_start_x = 0
        crop_start_y = 0

    # Adjust mole positions based on scale and crop
    scaled_moles = []
    for mole in moles:
        scaled_mole = mole.copy()
        # First scale the coordinates
        scaled_x = int(mole["x"] * scale)
        scaled_y = int(mole["y"] * scale)

        # Adjust for cropping
        scaled_x -= crop_start_x
        scaled_y -= crop_start_y

        # Only include moles that will be visible in the cropped area
        if (
            center_x is not None
            and crop_to_output
            and output_width is not None
            and output_height is not None
        ):
            if 0 <= scaled_x < output_width and 0 <= scaled_y < output_height:
                scaled_mole["x"] = scaled_x
                scaled_mole["y"] = scaled_y
                scaled_moles.append(scaled_mole)
        else:
            scaled_mole["x"] = scaled_x
            scaled_mole["y"] = scaled_y
            scaled_moles.append(scaled_mole)

    # Handle cropping if needed
    if (
        center_x is not None
        and crop_to_output
        and output_width is not None
        and output_height is not None
    ):
        current_height, current_width = labeled_image.shape[:2]

        # Make sure we don't try to crop more than the image size
        end_x = min(crop_start_x + output_width, current_width)
        end_y = min(crop_start_y + output_height, current_height)
        actual_width = end_x - crop_start_x
        actual_height = end_y - crop_start_y

        # Crop the image
        cropped_image = labeled_image[crop_start_y:end_y, crop_start_x:end_x]

        # If the cropped area is smaller than requested output size, pad it
        if actual_width < output_width or actual_height < output_height:
            padded_image = np.zeros(
                (output_height, output_width, 3), dtype=np.uint8
            )
            padded_image[:actual_height, :actual_width] = cropped_image
            labeled_image = padded_image
        else:
            labeled_image = cropped_image
    elif (
        output_width is not None
        and output_height is not None
        and not crop_to_output
    ):
        # Just resize to fit output dimensions if no centering/cropping
        current_height, current_width = labeled_image.shape[:2]
        if current_width != output_width or current_height != output_height:
            labeled_image = cv2.resize(
                labeled_image, (output_width, output_height)
            )

            # Need to adjust mole positions again for this final resize
            final_scale_x = output_width / current_width
            final_scale_y = output_height / current_height

            for mole in scaled_moles:
                mole["x"] = int(mole["x"] * final_scale_x)
                mole["y"] = int(mole["y"] * final_scale_y)

    # Track label positions to avoid overlaps
    label_positions = set()

    # Calculate circle size based on image size
    circle_radius = int(max(5, min(15, labeled_image.shape[1] / 40)))

    # Draw circles and labels for each mole
    for mole in scaled_moles:
        uuid = mole["uuid"]
        x, y = int(mole["x"]), int(mole["y"])

        # Skip if outside the image bounds
        if (
            x < 0
            or y < 0
            or x >= labeled_image.shape[1]
            or y >= labeled_image.shape[0]
        ):
            continue

        # Determine label and color based on whether mole is canonical
        if mole["is_uuid_canonical"]:
            label = canonical_labels[uuid]
            color = (0, 0, 255)  # Red for canonical moles
        else:
            label = non_canonical_labels.get(uuid, "?")
            color = (255, 0, 0)  # Blue for non-canonical moles

        # Draw a circle around the mole
        cv2.circle(
            labeled_image,
            (x, y),
            circle_radius,
            color,
            max(1, int(2 * scale)),
        )

        # Calculate label position
        label_x = x + circle_radius + 5
        label_y = y

        # Make sure label is within image bounds
        if label_x >= labeled_image.shape[1]:
            label_x = x - circle_radius - 20

        # Check for overlaps with existing labels
        label_key = (label_x, label_y)
        if label_key in label_positions:
            continue  # Skip this label if position already occupied

        label_positions.add(label_key)

        # Calculate font scale based on image size
        font_scale = max(0.4, min(0.9, labeled_image.shape[1] / 800))

        # Draw the label next to the mole
        cv2.putText(
            labeled_image,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, int(1.5 * scale)),
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
