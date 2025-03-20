"""Annotate an image with mole locations from a JSON file."""

import json
import pathlib
from typing import Dict, List, Tuple

import cv2
import numpy as np

import mel.lib.image
import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        "IMAGE",
        help="Path to the image to annotate.",
    )
    parser.add_argument(
        "MOLES_JSON",
        help="Path to the JSON file containing mole coordinates.",
    )
    parser.add_argument(
        "OUTPUT",
        help="Path to save the annotated image.",
    )


def process_args(args):
    image_path = pathlib.Path(args.IMAGE)
    if not image_path.exists():
        print(f"Error: Input image does not exist: {image_path}")
        return 1

    json_path = pathlib.Path(args.MOLES_JSON)
    if not json_path.exists():
        print(f"Error: JSON file does not exist: {json_path}")
        return 1

    # Try to load moles from JSON
    moles = []
    grid_ref_moles = []

    # First, try to load as standard format with x,y coordinates
    try:
        moles = mel.rotomap.moles.load_json(json_path)
        print(f"Loaded {len(moles)} moles in standard format from {json_path}")
    except Exception:
        # If that fails, try to load as grid_ref format
        try:
            with open(json_path, "r") as f:
                grid_ref_moles = json.loads(f.read())
            print(
                f"Loaded {len(grid_ref_moles)} moles with grid references from {json_path}"
            )
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return 1

    # Create annotated image with grid
    grid_annotated_image, grid_points = create_grid_annotated_image(image_path)

    # Convert grid references to coordinates if needed
    if not moles and grid_ref_moles:
        for mole in grid_ref_moles:
            if "grid_ref" in mole:
                try:
                    x, y = grid_ref_to_coordinates(
                        mole["grid_ref"], grid_points
                    )
                    mole["x"] = x
                    mole["y"] = y
                except KeyError:
                    print(
                        f"Warning: Invalid grid reference: {mole['grid_ref']}"
                    )
        moles = grid_ref_moles

    if not moles:
        print("No valid mole data found in JSON file")
        return 1

    # Create the final annotated image
    annotated_image = create_annotated_image(image_path, moles, grid_points)

    # Save annotated image
    output_path = pathlib.Path(args.OUTPUT)
    cv2.imwrite(str(output_path), annotated_image)
    print(f"Saved annotated image to {output_path}")

    return 0


def create_grid_annotated_image(
    image_path: pathlib.Path,
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """Create an image with a 7x7 grid of lettered points that includes edges
    and corners.

    Args:
        image_path: Path to the original image

    Returns:
        Tuple containing:
        - Annotated image as a numpy array
        - Dictionary mapping grid labels to (x, y) coordinates
    """
    # Load the original image
    image = mel.lib.image.load_image(image_path)
    height, width = image.shape[:2]

    # Create a copy for annotation
    annotated = image.copy()

    # Define the grid (7x7 to include edges)
    rows, cols = 7, 7

    # Generate grid labels (a-z lowercase, then A-Z uppercase)
    import string

    labels = list(string.ascii_lowercase)  # a-z (26 letters)
    labels.extend(list(string.ascii_uppercase[:23]))  # A-W (need 49 total)
    labels = labels[:49]  # Limit to needed number of labels

    # Create dictionary to store the grid point coordinates
    grid_points = {}

    # Calculate positions including edges
    x_positions = [0]  # Left edge
    for i in range(1, cols - 1):
        x_positions.append(width * i // (cols - 1))
    x_positions.append(width - 1)  # Right edge

    y_positions = [0]  # Top edge
    for i in range(1, rows - 1):
        y_positions.append(height * i // (rows - 1))
    y_positions.append(height - 1)  # Bottom edge

    idx = 0
    for y in y_positions:
        for x in x_positions:
            label = labels[idx]

            # Adjust drawing position slightly for edge points
            draw_x = max(5, min(width - 5, x))
            draw_y = max(5, min(height - 5, y))

            # Draw a small dot at the grid point
            cv2.circle(annotated, (draw_x, draw_y), 5, (0, 255, 0), -1)

            # Position label - adjust for edges
            label_x = draw_x - 10
            label_y = draw_y - 10
            if x < 20:  # Left edge
                label_x = draw_x + 5
            if y < 20:  # Top edge
                label_y = draw_y + 15

            # Add the letter label
            cv2.putText(
                annotated,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Store the actual coordinates (not the adjusted drawing coordinates)
            grid_points[label] = (x, y)
            idx += 1

    return annotated, grid_points


def create_annotated_image(
    image_path: pathlib.Path,
    moles: List[Dict],
    grid_points: Dict[str, Tuple[int, int]],
) -> np.ndarray:
    """Create an annotated image with markers for detected moles.

    Args:
        image_path: Path to the original image
        moles: List of moles with x,y coordinates
        grid_points: Dictionary mapping grid labels to coordinates

    Returns:
        Annotated image as a numpy array
    """
    # Start with a grid-annotated image
    annotated, _ = create_grid_annotated_image(image_path)

    # Add numbered markers for each mole
    for i, mole in enumerate(moles):
        # Get coordinates - either directly from x,y or from grid_ref
        if "x" in mole and "y" in mole:
            x, y = int(mole["x"]), int(mole["y"])
        elif "grid_ref" in mole and grid_points:
            try:
                x, y = grid_ref_to_coordinates(mole["grid_ref"], grid_points)
            except KeyError:
                print(
                    f"Warning: Skipping mole with invalid grid reference: {mole.get('grid_ref')}"
                )
                continue
        else:
            continue

        # Draw a circle around the mole
        cv2.circle(annotated, (x, y), 10, (0, 0, 255), 2)

        # Add a label for the mole
        if "id" in mole:
            # Use ID if available
            label = str(mole["id"])
        elif "uuid" in mole:
            # Use shortened UUID if available (first 8 chars)
            label = str(mole["uuid"])[:8]
        elif "grid_ref" in mole:
            # Use grid reference as label
            label = mole["grid_ref"]
        else:
            # Default to index
            label = str(i + 1)

        cv2.putText(
            annotated,
            label,
            (x + 15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    return annotated


def grid_ref_to_coordinates(
    grid_ref: str, grid_points: Dict[str, Tuple[int, int]]
) -> Tuple[int, int]:
    """Convert a grid reference to pixel coordinates.

    Args:
        grid_ref: A string grid reference (e.g., "C", "FG", "AA", or "BAC")
        grid_points: Dictionary mapping grid labels to (x, y) coordinates

    Returns:
        Tuple of (x, y) pixel coordinates
    """
    points = [grid_points[r] for r in grid_ref]
    x_sum = sum(p[0] for p in points)
    y_sum = sum(p[1] for p in points)
    return (x_sum // len(points), y_sum // len(points))


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
