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
import numpy as np

import mel.rotomap.moles


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

    # Request analysis from Claude
    start_time = time.time()
    detected_moles, cost, api_error = analyze_image_with_claude(
        image_path, api_key
    )
    elapsed_time = time.time() - start_time

    if api_error:
        print(f"Error from Claude API: {api_error}")
        return 1

    if not detected_moles:
        print("No moles detected by Claude.")
        return 1

    # Compare detected moles with ground truth
    matches, unmatched_truth, unmatched_detected = compare_moles(
        ground_truth_moles, detected_moles, args.threshold
    )

    # Print results
    print(f"\nResults for {image_path}:")
    print(f"Ground truth moles: {len(ground_truth_moles)}")
    print(f"Claude detected moles: {len(detected_moles)}")
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
            print(f"  ({mole['x']}, {mole['y']})")

    print(f"\nAPI request completed in {elapsed_time:.2f} seconds")
    print(f"Estimated API cost: ${cost:.6f}")

    return 0


def analyze_image_with_claude(
    image_path: pathlib.Path, api_key: str
) -> Tuple[List[Dict], float, Optional[str]]:
    """Analyze an image with Claude API to detect moles using the Anthropic
    library.

    Args:
        image_path: Path to the image file
        api_key: Claude API key

    Returns:
        Tuple containing:
        - List of detected moles with x, y coordinates
        - Estimated cost of the API call
        - Error message if the request failed, None otherwise
    """
    # Read and encode the image
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Base64 encode the image for the API
    base64_image = base64.b64encode(image_data).decode("utf-8")

    # Create prompt for Claude
    prompt = """This image is a patch from a skin imaging system that tracks moles. Please identify all moles in this image.

A mole typically appears as a small, dark spot on the skin. It can be black, brown, or tan in color and circular or oval in shape.

For each mole you identify, provide its pixel coordinates (x, y) in the format:
```json
[
  {"x": 123, "y": 456},
  {"x": 789, "y": 101}
]
```

Important guidelines:
1. Only include actual moles, not artifacts, shadows, or reflections
2. Use integer pixel coordinates
3. Provide coordinates in a valid JSON array of objects
4. Don't include any other information in the JSON besides x and y values

Please respond with ONLY the JSON array and no additional explanation or text.
"""

    # Set up the message content with the prompt and image
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64_image,
            },
        },
    ]

    # Initialize the Anthropic client
    client = anthropic.Anthropic(api_key=api_key)

    try:
        # Send the message to Claude
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": content}],
        )

        # Calculate approximate cost
        # Claude-3 Opus costs per 1M input tokens: $15, output tokens: $75
        # Image processing is roughly equivalent to 1,000 tokens
        input_tokens = 1000  # Approximate for image + short prompt
        output_tokens = (
            response.usage.output_tokens if hasattr(response, "usage") else 100
        )

        # Calculate cost in dollars
        input_cost = (input_tokens / 1_000_000) * 15
        output_cost = (output_tokens / 1_000_000) * 75
        total_cost = input_cost + output_cost

        # Extract moles from response
        detected_moles = []

        # Parse the response content
        if response.content:
            for block in response.content:
                if block.type == "text":
                    text = block.text
                    # Extract JSON content
                    try:
                        # Try to find JSON array in the text
                        json_start = text.find("[")
                        json_end = text.rfind("]") + 1

                        if json_start >= 0 and json_end > json_start:
                            json_text = text[json_start:json_end]
                            detected_moles = json.loads(json_text)
                            break
                        else:
                            return (
                                [],
                                total_cost,
                                "No JSON data found in Claude's response",
                            )
                    except json.JSONDecodeError:
                        return (
                            [],
                            total_cost,
                            "Invalid JSON in Claude's response",
                        )

        return detected_moles, total_cost, None

    except anthropic.APIError as e:
        return [], 0.0, f"Anthropic API error: {str(e)}"
    except Exception as e:
        return [], 0.0, f"Unexpected error: {str(e)}"


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
