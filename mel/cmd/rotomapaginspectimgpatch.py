"""Analyze an image patch with a VLM and compare with ground truth."""

import argparse
import base64
import json
import math
import os
import pathlib
import time
from typing import Dict, List, Optional, Tuple

import anthropic
import cv2
import numpy as np

import mel.lib.image
import mel.rotomap.moles

# Claude model pricing information (per 1M tokens as of March 2025)
# Check for updates: https://docs.anthropic.com/en/docs/about-claude/models/all-models#model-names
MODEL_PRICING = {
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-3-7-sonnet-20250219": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet-20240620": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.8, "output": 4.0},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}

# Prompt templates
ANALYSIS_PROMPT = """This image is a patch from a skin imaging system that tracks moles. The image has a 7x7 grid of lettered dots overlaid on it to help with location references. The grid covers the entire image, including the edges and corners. Grid points are labeled with lowercase letters (a-z) first, and then uppercase letters (A-W). Please examine the image carefully and describe all the moles you can identify.

A mole typically appears as a small, dark spot on the skin. It can be black, brown, or tan in color and circular or oval in shape.

For each mole:
1. Assign it a numerical id (starting from 1)
2. Describe its appearance (color, size, shape)
3. Describe its location relative to the grid points (e.g., "under point c", "halfway between points f and k" or "a third of the way between A and B")
4. Note any distinctive features or landmarks near the mole that could help with identification

Focus on being thorough and accurate. Distinguish between actual moles and potential artifacts, shadows, or reflections that may appear similar. The numbering will help us track each mole consistently.
"""

COORDINATES_PROMPT = """Thank you for that analysis. Now, based on your observations, please provide the location for each numbered mole using the grid reference system in the following format:
```json
[
  {"id": 1, "grid_ref": "c"},
  {"id": 2, "grid_ref": "ffk"},
  {"id": 3, "grid_ref": "M"}
]
```

Important guidelines:
1. Only include actual moles, not artifacts, shadows, or reflections.
2. For moles positioned directly under or very close to a grid point, use that point's letter (e.g., "c" or "M").
3. For moles located between grid points, use more letters to indicate the nearest points (e.g., "fk" or "uA").
4. Note that you may use an arbitrary number of letters, and the position is the mean of the corresponding grid points. This means that "a third of the way between A and B" is represented as "AAB" or "BAA".
5. Keep the same numbering you used in your analysis.
6. Provide coordinates in a valid JSON array of objects.
7. Don't include any other information in the JSON besides id and grid_ref values.

Please respond with ONLY the JSON array and no additional explanation or text.
"""


def setup_parser(parser):
    parser.add_argument(
        "IMAGE",
        help="Path to the image patch created by ag-get-image-patch.",
    )
    parser.add_argument(
        "--api-key",
        help="Claude API key. If not provided, will use ANTHROPIC_API_KEY environment variable.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=20.0,
        help="Maximum distance (in pixels) to consider a mole match (default: 20.0).",
    )
    # For current model options, see:
    #
    #   curl https://api.anthropic.com/v1/models \
    #       --header "x-api-key: $ANTHROPIC_API_KEY" \
    #       --header "anthropic-version: 2023-06-01"
    #
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
        "--save-annotated",
        help="Save the annotated image to the specified path (for debugging).",
    )


def process_args(args):
    image_path = pathlib.Path(args.IMAGE)
    if not image_path.exists():
        print(f"Error: Input image does not exist: {image_path}")
        return 1

    json_path = pathlib.Path(f"{image_path}.json")
    if not json_path.exists():
        print(f"Error: JSON file does not exist: {json_path}")
        return 1

    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "Error: Claude API key not provided. Please provide the --api-key parameter "
            "or set the ANTHROPIC_API_KEY environment variable."
        )
        return 1

    # Load ground truth moles
    ground_truth_moles = mel.rotomap.moles.load_json(json_path)
    print(f"Ground truth moles: {len(ground_truth_moles)}")
    print("\nUnmatched ground truth moles (UUID):")
    for mole in ground_truth_moles:
        print(f"  {mole['uuid']} at ({mole['x']}, {mole['y']})")
    print()

    # Create a grid-annotated version of the original image for reference
    grid_annotated_image, grid_points = create_grid_annotated_image(image_path)

    # Save grid image if requested
    if args.save_annotated:
        grid_path = f"{args.save_annotated}.grid.jpg"
        cv2.imwrite(grid_path, grid_annotated_image)
        print(f"Saved grid reference image to {grid_path}")

    # First round: Initial analysis with Claude
    start_time = time.time()
    print("Analyzing image with " + args.model + " (first round)")
    print("  Using a lettered grid for coordinate references")
    detected_moles, messages = analyze_image_with_claude(
        image_path, api_key, args.model
    )

    if not detected_moles:
        print("No moles detected by Claude.")
        return 1

    print()

    # Print detected moles with both grid references and coordinates
    print("Detected moles:")
    for mole in detected_moles:
        grid_ref = mole["grid_ref"]
        x, y = mole["x"], mole["y"]
        print(f"  Mole {mole['id']}: Grid ref {grid_ref} at ({x}, {y})")
    print()

    if args.save_annotated:
        annotated_image = create_annotated_image(
            image_path, detected_moles, grid_points
        )
        annotated_path = f"{args.save_annotated}.1.jpg"
        cv2.imwrite(annotated_path, annotated_image)
        print(f"Saved annotated image to {annotated_path}")

    # Refine the detected moles, one at a time.
    print("\nRefining mole coordinates...")
    refined_moles = refine_mole_coordinates(
        image_path, detected_moles, grid_points, api_key, args.model, messages
    )

    print("\nRefined moles:")
    for mole in refined_moles:
        grid_ref = mole.get("grid_ref", "")
        x, y = mole["x"], mole["y"]
        print(f"  Mole {mole['id']}: Grid ref {grid_ref} at ({x}, {y})")

    # Update detected_moles with refined coordinates
    detected_moles = refined_moles

    if args.save_annotated:
        annotated_image = create_annotated_image(
            image_path, detected_moles, grid_points
        )
        annotated_path = f"{args.save_annotated}.2.jpg"
        cv2.imwrite(annotated_path, annotated_image)
        print(f"Saved annotated image to {annotated_path}")

    elapsed_time = time.time() - start_time

    # Compare detected moles with ground truth
    matches, unmatched_truth, unmatched_detected = compare_moles(
        ground_truth_moles, detected_moles, args.threshold
    )

    print(f"Matched moles: {len(matches)}")
    print(f"Unmatched ground truth: {len(unmatched_truth)}")
    print(f"Unmatched detected: {len(unmatched_detected)}")

    if matches:
        print("\nMatched moles (UUID, distance in pixels):")
        for gt_uuid, distance in matches:
            print(f"  {gt_uuid}: {distance:.2f}px")

    if unmatched_truth:
        print("\nUnmatched ground truth moles (UUID):")
        for mole in unmatched_truth:
            print(f"  {mole['uuid']} at ({mole['x']}, {mole['y']})")

    if unmatched_detected:
        print("\nUnmatched detected moles (coordinates):")
        for mole in unmatched_detected:
            grid_ref = mole.get("grid_ref", "")
            id_str = f"Mole {mole['id']}: " if "id" in mole else ""
            grid_str = f"Grid ref {grid_ref} " if grid_ref else ""
            print(f"  {id_str}{grid_str}at ({mole['x']}, {mole['y']})")

    print()
    print(f"API request completed in {elapsed_time:.2f} seconds")

    return 0


def analyze_image_with_claude(
    image_path: pathlib.Path,
    api_key: str,
    model: str = "claude-3-opus-20240229",
) -> Tuple[List[Dict], List[Dict]]:
    """Analyze an image with Claude API to detect moles using the Anthropic
    library.

    Args:
        image_path: Path to the image file
        api_key: Claude API key
        model: Claude model to use (default: claude-3-opus-20240229)

    Returns:
        - List of detected moles with coordinates (x, y)
        - Full conversation messages list that can be used in subsequent calls
    """
    grid_annotated_image, grid_points = create_grid_annotated_image(image_path)

    image_content = encode_image_for_api(grid_annotated_image)

    # Initialize the Anthropic client
    client = anthropic.Anthropic(api_key=api_key)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ANALYSIS_PROMPT},
                image_content,
            ],
        }
    ]

    # First turn: Get qualitative analysis
    print("  Step 1: Asking Claude to analyze the image with grid points...")
    analysis_response = client.messages.create(
        model=model,
        max_tokens=1500,
        messages=messages,
    )

    # Extract the analysis text and build conversation history
    if not analysis_response.content:
        raise ValueError("No content in Claude analysis response.")

    if len(analysis_response.content) != 1:
        raise ValueError(
            "Unexpected number of content blocks in analysis response",
            analysis_response.content,
        )

    if analysis_response.content[0].type != "text":
        raise ValueError(
            "Unexpected content type in analysis response",
            analysis_response.content[0].type,
        )

    mole_analysis = analysis_response.content[0].text

    print("\nClaude's analysis:")
    print("-" * 60)
    print(mole_analysis)
    print("-" * 60)

    messages.append(
        {"role": "assistant", "content": analysis_response.content}
    )

    messages.append({"role": "user", "content": COORDINATES_PROMPT})

    # Second turn: Ask for grid references based on analysis
    response = client.messages.create(
        model=model, max_tokens=1000, messages=messages
    )
    messages.append({"role": "assistant", "content": response.content})

    detected_moles = parse_claude_gridref_response(response, grid_points)

    return detected_moles, messages


def encode_image_for_api(image_data: np.ndarray) -> str:
    """Encode an image for use with the Anthropic API.

    Args:
        image_data: Image data as a numpy array

    Returns:
        Base64 encoded image string
    """
    success, img_encoded = cv2.imencode(".jpg", image_data)
    if not success:
        raise ValueError("Failed to encode image")
    image_bytes = img_encoded.tobytes()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    image_content = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64_image,
        },
    }

    return image_content


def parse_claude_gridref_response(response, grid_points) -> List[Dict]:
    """Parse a Claude response for grid references to mole coordinates.

    Args:
        response: Claude response message object

    Returns:
        List of detected moles. Each mole is a dictionary with "id" and "grid_ref" keys.
    """
    if not response.content:
        raise ValueError("No content in Claude response")

    if len(response.content) > 1:
        raise ValueError(
            "Unexpected number of content blocks in response", response.content
        )

    block = response.content[0]

    if block.type != "text":
        raise ValueError("Unexpected content type in response", block.type)

    # We are likely to get a markdown block with json in it.
    # Extract whatever might be between '[]'.
    json_block = block.text[block.text.find("[") : block.text.rfind("]") + 1]

    try:
        moles = json.loads(json_block)
    except json.JSONDecodeError as e:
        raise ValueError("Failed to parse JSON in response", block.text) from e

    for m in moles:
        if "id" not in m:
            raise ValueError("Missing 'id' in mole data", m)

        if "grid_ref" not in m:
            raise ValueError("Missing 'grid_ref' in mole data", m)

        try:
            m["x"], m["y"] = grid_ref_to_coordinates(
                m["grid_ref"], grid_points
            )
        except KeyError as e:
            raise ValueError("Invalid grid reference in mole data", m) from e

    return moles


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
    grid_points: Optional[Dict[str, Tuple[int, int]]] = None,
) -> np.ndarray:
    """Create an annotated image with markers for detected moles.

    Args:
        image_path: Path to the original image
        moles: List of detected moles with x,y coordinates or id/grid_ref
        grid_points: Optional dictionary mapping grid labels to coordinates

    Returns:
        Annotated image as a numpy array
    """
    # Load the original image or use the grid-annotated image
    if grid_points:
        # Start with a grid-annotated image
        annotated, _ = create_grid_annotated_image(image_path)
    else:
        # Use the original image
        image = mel.lib.image.load_image(image_path)
        annotated = image.copy()

    # Add numbered markers for each mole
    for i, mole in enumerate(moles):
        # Handle both coordinate formats
        if "x" in mole and "y" in mole:
            x, y = int(mole["x"]), int(mole["y"])
        elif "id" in mole and "grid_ref" in mole:
            # Skip if we don't have grid points and this is a grid reference
            if not grid_points:
                continue
            # Convert grid reference to coordinates
            x, y = grid_ref_to_coordinates(mole["grid_ref"], grid_points)
        else:
            continue

        # Draw a circle around the mole
        cv2.circle(annotated, (x, y), 10, (0, 0, 255), 2)

        # Add a number label (use id if available, otherwise i+1)
        number = str(mole.get("id", i + 1))
        cv2.putText(
            annotated,
            number,
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


def refine_mole_coordinates(
    image_path: pathlib.Path,
    moles: List[Dict],
    grid_points: Dict[str, Tuple[int, int]],
    api_key: str,
    model: str,
    previous_messages: List[Dict],
) -> List[Dict]:
    """Refine the coordinates of detected moles using a grid of 9 candidate
    positions.

    For each mole, we create up to 9 candidate images with different offsets from the
    original coordinates. We send the images to Claude in a single turn and ask
    it to analyze which image best circles the mole. Based on the response, we
    update the mole coordinates.

    Args:
        image_path: Path to the original image
        moles: List of detected moles with x,y coordinates
        grid_points: Dictionary mapping grid labels to coordinates
        api_key: Claude API key
        model: Claude model to use
        previous_messages: Previous conversation with Claude

    Returns:
        List of moles with refined coordinates
    """
    # Load the original image
    original_image, _ = create_grid_annotated_image(image_path)

    # Create a copy of the moles list to update
    refined_moles = moles.copy()

    # Process each mole
    for i, mole in enumerate(moles):
        mole_id = mole["id"]
        print(f"  Refining Mole {mole_id}...")

        # Generate 9 candidate images with different offsets
        candidates = generate_offset_candidates(mole, original_image)
        candidate_images = []

        # Create annotated images for each candidate
        for j, (offset_name, offset_x, offset_y) in enumerate(candidates):
            # Create a temporary mole with the offset coordinates
            temp_mole = mole.copy()
            temp_mole["x"] = int(mole["x"] + offset_x)
            temp_mole["y"] = int(mole["y"] + offset_y)

            # Create an annotated image
            annotated = original_image.copy()
            cv2.circle(
                annotated, (temp_mole["x"], temp_mole["y"]), 10, (0, 0, 255), 2
            )

            # Add image number and offset name
            position_text = f"Image {j+1}: {offset_name}"
            cv2.putText(
                annotated,
                position_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            # Save the annotated image for debugging
            annotated_path = f"annotated.{i}.{j}.jpg"
            cv2.imwrite(annotated_path, annotated)
            print(f"Saved annotated image to {annotated_path}")

            # Add to candidate images list
            candidate_images.append((f"Image {j+1}", annotated))

        # Prepare the refinement prompt and candidate images for Claude
        refinement_result = analyze_candidate_images(
            candidate_images, mole_id, api_key, model, previous_messages
        )

        # Extract the best candidate index (1-based)
        best_index = parse_refinement_result(refinement_result)

        if best_index is not None and 1 <= best_index <= len(candidates):
            # Update mole coordinates with the selected offset
            _, offset_x, offset_y = candidates[best_index - 1]
            refined_moles[i]["x"] = int(mole["x"] + offset_x)
            refined_moles[i]["y"] = int(mole["y"] + offset_y)
            print(
                f"    Selected candidate {best_index}, offset: {candidates[best_index - 1][0]}"
            )

    return refined_moles


def generate_offset_candidates(
    mole: Dict, original_image: np.ndarray
) -> List[Tuple[str, int, int]]:
    """Generate 9 candidate positions for a mole with different offsets.

    Args:
        mole: Mole dictionary with x,y coordinates
        original_image: Original image as numpy array

    Returns:
        List of tuples (offset_name, offset_x, offset_y)
    """
    # Get image dimensions
    height, width = original_image.shape[:2]

    # Calculate pixel offset based on image size (approximately 5% of width/height)
    offset_pixels = min(int(min(width, height) * 0.05), 15)

    # Define 9 offset positions
    candidates = [
        ("Center (no offset)", 0, 0),
        ("Right", offset_pixels, 0),
        ("Upper Right", offset_pixels, -offset_pixels),
        ("Up", 0, -offset_pixels),
        ("Upper Left", -offset_pixels, -offset_pixels),
        ("Left", -offset_pixels, 0),
        ("Lower Left", -offset_pixels, offset_pixels),
        ("Down", 0, offset_pixels),
        ("Lower Right", offset_pixels, offset_pixels),
    ]

    # Ensure coordinates stay within image bounds
    valid_candidates = []
    for name, dx, dy in candidates:
        new_x = max(0, min(width - 1, mole["x"] + dx))
        new_y = max(0, min(height - 1, mole["y"] + dy))
        # Adjust dx and dy based on boundary constraints
        actual_dx = new_x - mole["x"]
        actual_dy = new_y - mole["y"]
        valid_candidates.append((name, actual_dx, actual_dy))

    return valid_candidates


def analyze_candidate_images(
    candidate_images: List[Tuple[str, np.ndarray]],
    mole_id: int,
    api_key: str,
    model: str,
    previous_messages: Optional[List[Dict]] = None,
) -> str:
    """Send candidate images to Claude for analysis and selection of the best
    one.

    This is done in two turns:
    1. First turn: Ask Claude to analyze all 9 images in detail
    2. Second turn: Ask Claude to select the best image number

    Args:
        candidate_images: List of (image_name, image_data) tuples
        mole_id: ID of the mole being refined
        api_key: Claude API key
        model: Claude model to use
        previous_messages: Optional previous conversation with Claude

    Returns:
        Claude's final selection response text
    """
    client = anthropic.Anthropic(api_key=api_key)

    # --- FIRST TURN: Detailed analysis of all images ---

    # Create content blocks for analysis
    analysis_content_blocks = []
    analysis_content_blocks.append(
        {
            "type": "text",
            "text": f"""Now we'll try to refine the targetting of mole {mole_id}. I'll show you multiple image options, with a red circle drawn in slightly offset positions on each image.

Please analyze all the images with respect to the distance from the circle to the target. Please focus on:
1. The index of the image, starting from 1.
2. Whether the circle appears to be properly centered on a mole.
3. Whether this image seems to be closer to correct than the other images.

Provide a brief analysis for each image, and then finally comment on which seems to be the closest to correct.""",
        }
    )

    # Add each image
    for image_name, image_data in candidate_images:
        analysis_content_blocks.append(encode_image_for_api(image_data))

    # Build message list for analysis, including previous conversation if provided
    analysis_messages = []
    if previous_messages:
        # Include relevant previous context but limit to essential messages
        # This typically includes the initial analysis and coordinates response
        relevant_previous = previous_messages[
            :4
        ]  # Usually user request, assistant analysis, user request for coords, assistant coords
        analysis_messages.extend(relevant_previous)

    # Add the new analysis request
    analysis_messages.append(
        {"role": "user", "content": analysis_content_blocks}
    )

    # Get analysis response
    print(f"    Requesting detailed analysis of all 9 candidate images...")
    analysis_response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=analysis_messages,
    )

    if not analysis_response.content:
        raise ValueError("No content in Claude analysis response")

    if (
        len(analysis_response.content) < 1
        or analysis_response.content[0].type != "text"
    ):
        raise ValueError(
            "Unexpected content format in Claude analysis",
            analysis_response.content,
        )

    analysis_text = analysis_response.content[0].text

    # Print the analysis
    print("\n    Claude's analysis of candidate images:")
    print("    " + "-" * 60)
    # Format the analysis text with proper indentation
    for line in analysis_text.split("\n"):
        print(f"    {line}")
    print("    " + "-" * 60 + "\n")

    # --- SECOND TURN: Selection of the best image ---

    # Create selection messages, continuing from the analysis
    selection_messages = analysis_messages.copy()
    selection_messages.append(
        {"role": "assistant", "content": analysis_response.content}
    )
    selection_messages.append(
        {
            "role": "user",
            "content": f"""Based on your analysis, which single image has the red circle that comes closest to the target mole?

Your response should be a single number (e.g. 1-9) indicating your final selection. If multiple images seem equally good, choose the one that appears to be most precisely centered on the mole.""",
        }
    )

    # Get selection response
    print(f"    Requesting final image selection...")
    selection_response = client.messages.create(
        model=model,
        max_tokens=250,
        messages=selection_messages,
    )

    if not selection_response.content:
        raise ValueError("No content in Claude selection response")

    if (
        len(selection_response.content) < 1
        or selection_response.content[0].type != "text"
    ):
        raise ValueError(
            "Unexpected content format in Claude selection",
            selection_response.content,
        )

    selection_text = selection_response.content[0].text

    # Print the selection
    print(f"    Claude's selection: {selection_text.strip()}")

    # Return the selection response text
    return selection_text


def parse_refinement_result(result_text: str) -> Optional[int]:
    """Parse the refinement result from Claude to extract the best image
    number.

    Args:
        result_text: Claude's response text

    Returns:
        The selected image number (1-9) or None if parsing failed
    """
    # First try to find a single digit between 1-9
    import re

    matches = re.findall(r"\b([1-9])\b", result_text)

    if matches:
        try:
            return int(matches[0])
        except (ValueError, IndexError):
            pass

    # Try more complex parsing if simple digit extraction fails
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
    }

    for word, num in number_words.items():
        if word in result_text.lower():
            return num

    raise ValueError("Failed to parse Claude's response", result_text)


def compare_moles(
    ground_truth: List[Dict], detected: List[Dict], threshold: float
) -> Tuple[List[Tuple[str, float]], List[Dict], List[Dict]]:
    """Compare detected moles with ground truth moles.

    Args:
        ground_truth: List of ground truth moles with x, y coordinates and uuid
        detected: List of detected moles with x, y coordinates
        threshold: Maximum distance (pixels) to consider a match

    Returns:
        Tuple containing:
        - List of (uuid, distance) tuples for matched moles
        - List of unmatched ground truth moles
        - List of unmatched detected moles
    """
    matches = []
    unmatched_truth = ground_truth.copy()
    unmatched_detected = detected.copy()

    # For each ground truth mole, find the closest detected mole
    # Start with the closest pairs first to handle overlapping moles correctly
    match_candidates = []

    for gt_idx, gt_mole in enumerate(ground_truth):
        for det_idx, det_mole in enumerate(detected):
            distance = math.sqrt(
                (gt_mole["x"] - det_mole["x"]) ** 2
                + (gt_mole["y"] - det_mole["y"]) ** 2
            )
            if distance <= threshold:
                match_candidates.append((gt_idx, det_idx, distance))

    # Sort by distance
    match_candidates.sort(key=lambda x: x[2])

    # Process matches in order of increasing distance
    gt_matched = set()
    det_matched = set()

    for gt_idx, det_idx, distance in match_candidates:
        # Skip if either mole is already matched
        if gt_idx in gt_matched or det_idx in det_matched:
            continue

        # Record the match
        gt_matched.add(gt_idx)
        det_matched.add(det_idx)

        # Add to matches list
        matches.append((ground_truth[gt_idx]["uuid"], distance))

    # Collect unmatched moles
    unmatched_truth = [
        m for i, m in enumerate(ground_truth) if i not in gt_matched
    ]
    unmatched_detected = [
        m for i, m in enumerate(detected) if i not in det_matched
    ]

    return matches, unmatched_truth, unmatched_detected


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
