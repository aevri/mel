"""Functionality for working with AI agents and VLMs for skin analysis."""

import base64
import re
from typing import Dict, List, Optional, Tuple

import anthropic
import cv2
import numpy as np


def encode_image_for_api(image_data: np.ndarray) -> Dict:
    """Encode an image for use with the Anthropic API.

    Args:
        image_data: Image data as a numpy array

    Returns:
        Content block for the API with base64 encoded image
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
            "text": f"""I'm showing you 9 different images of the same skin area, with a red circle drawn in slightly different positions on each image. These are labeled as Image 1 through Image 9.

Please analyze all 9 images and describe what you see in each one. Focus on:
1. The position of the red circle in each image
2. How well the circle captures any mole or skin feature
3. The differences between each image's circle placement
4. Whether the circle appears to be properly centered on a mole

Provide a brief analysis for each image (1-9), comparing them to help determine which one has the most accurate circle placement.""",
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
            "content": f"""Based on your analysis, which single image (1-9) has the red circle that most accurately encircles the mole?

Your response should be a single number (1-9) indicating your final selection. If multiple images seem equally good, choose the one that appears to be most precisely centered on the mole.""",
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


def parse_selection_result(result_text: str) -> Optional[int]:
    """Parse the selection result from Claude to extract the best image number.

    Args:
        result_text: Claude's response text

    Returns:
        The selected image number (1-9) or None if parsing failed
    """
    # First try to find a single digit between 1-9
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

    # If all parsing attempts fail, return None
    print(
        f"Warning: Could not parse a valid image selection from Claude's response: {result_text}"
    )
    return 5  # Default to center image if parsing fails


def refine_mole_coordinates(
    image_path: str,
    moles: List[Dict],
    api_key: str,
    model: str = "claude-3-7-sonnet-20250219",
    previous_messages: Optional[List[Dict]] = None,
    debug_output_prefix: Optional[str] = None,
) -> List[Dict]:
    """Refine the coordinates of detected moles using a grid of 9 candidate
    positions.

    For each mole, we create 9 candidate images with different offsets from the
    original coordinates. We send all 9 images to Claude in a single turn and ask
    it to analyze which image best circles the mole. Based on the response, we
    update the mole coordinates.

    Args:
        image_path: Path to the original image
        moles: List of detected moles with x,y coordinates
        api_key: Claude API key
        model: Claude model to use (default: claude-3-7-sonnet-20250219)
        previous_messages: Optional previous conversation with Claude
        debug_output_prefix: Optional prefix for saving debug images

    Returns:
        List of moles with refined coordinates
    """
    # Load the original image
    if isinstance(image_path, str):
        from pathlib import Path

        image_path = Path(image_path)

    import mel.lib.image

    original_image = mel.lib.image.load_image(image_path)

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

            # Save the annotated image for debugging if prefix is provided
            if debug_output_prefix:
                annotated_path = f"{debug_output_prefix}.mole{i}.pos{j}.jpg"
                cv2.imwrite(annotated_path, annotated)
                print(f"    Saved annotated image to {annotated_path}")

            # Add to candidate images list
            candidate_images.append((f"Image {j+1}", annotated))

        # Prepare the refinement prompt and candidate images for Claude
        refinement_result = analyze_candidate_images(
            candidate_images, mole_id, api_key, model, previous_messages
        )

        # Extract the best candidate index (1-based)
        best_index = parse_selection_result(refinement_result)

        if best_index is not None and 1 <= best_index <= len(candidates):
            # Update mole coordinates with the selected offset
            _, offset_x, offset_y = candidates[best_index - 1]
            refined_moles[i]["x"] = int(mole["x"] + offset_x)
            refined_moles[i]["y"] = int(mole["y"] + offset_y)
            print(
                f"    Selected candidate {best_index}, offset: {candidates[best_index - 1][0]}"
            )

    return refined_moles


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
