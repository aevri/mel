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
ANALYSIS_PROMPT = """This image is a patch from a skin imaging system that tracks moles. Please examine the image carefully and describe all the moles you can identify.

A mole typically appears as a small, dark spot on the skin. It can be black, brown, or tan in color and circular or oval in shape.

For each mole:
1. Describe its appearance (color, size, shape)
2. Describe its location on the skin
3. Note any distinctive features or landmarks near the mole that could help with identification
4. Estimate its approximate pixel coordinates (x, y) if possible

Focus on being thorough and accurate. Distinguish between actual moles and potential artifacts, shadows, or reflections that may appear similar.
"""

COORDINATES_PROMPT = """Thank you for that analysis. Now, based on your observations, please provide the precise pixel coordinates (x, y) for each mole you identified in the format:
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

REFINEMENT_ANALYSIS_PROMPT_TEMPLATE = """This image is a patch from a skin imaging system with annotations for moles that were detected in a previous step. The moles have been numbered and circled in red.

Here are the moles I detected previously:
{numbered_moles}

I need you to carefully analyze this image again. For each numbered circle:
1. Is it actually a mole, or could it be something else (shadow, artifact, etc.)?
2. Is the circle accurately centered on the mole, or should the position be adjusted?

Also:
3. Are there any moles I missed entirely in my first analysis?
4. If you have access to your previous analysis from the first image, what did you learn that could help with this refined analysis?

Please provide a thoughtful analysis of each potential mole and explain your reasoning clearly.
"""

REFINEMENT_COORDINATES_PROMPT = """Thank you for that thoughtful analysis. Now, based on your observations, please provide your final refined list of mole coordinates in the format:
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
        "--no-refine",
        action="store_true",
        help="Disable the refinement step (second round with annotated image).",
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

    # First round: Initial analysis with Claude
    start_time = time.time()
    print("Analyzing image with " + args.model + " (first round)")
    (
        detected_moles,
        first_cost,
        api_error,
        analysis_text,
        conversation_history,
        _,  # We don't need the refinement analysis in the first round
    ) = analyze_image_with_claude(image_path, api_key, args.model)

    # Print the analysis text if available
    if analysis_text:
        print("\nClaude's analysis:")
        print("-" * 60)
        print(analysis_text)
        print("-" * 60)
    first_round_time = time.time() - start_time

    if api_error:
        print(f"Error from Claude API: {api_error}")
        return 1

    if not detected_moles:
        print("No moles detected by Claude in the first round.")
        return 1

    total_cost = first_cost

    print()

    print("Detected moles (coordinates):")
    for mole in detected_moles:
        print(f"  ({mole['x']}, {mole['y']})")
    print()

    # Second round: Refinement with annotated image
    if not args.no_refine:
        # Create annotated image with the first-round detections
        annotated_image = create_annotated_image(image_path, detected_moles)

        # Save annotated image if requested
        if args.save_annotated:
            annotated_path = f"{args.save_annotated}.1.jpg"
            cv2.imwrite(annotated_path, annotated_image)
            print(f"Saved annotated image to {annotated_path}")

        # Second round with annotated image
        second_round_start = time.time()
        print("Refining analysis with " + args.model + " (second round)")
        print("  Using conversation history from first round for context")
        (
            refined_moles,
            second_cost,
            api_error,
            _,
            refined_conversation,
            refinement_analysis,
        ) = analyze_image_with_claude(
            image_path,
            api_key,
            args.model,
            annotated_image=annotated_image,
            first_round_moles=detected_moles,
            previous_conversation=conversation_history,
        )
        second_round_time = time.time() - second_round_start

        if api_error:
            print(f"Error from Claude API in refinement round: {api_error}")
            print("Using results from first round only.")
        elif not refined_moles:
            print(
                "No moles detected in refinement round. Using results from first round."
            )
        else:
            # Print the refinement analysis if available
            if refinement_analysis:
                print("\nClaude's refinement analysis:")
                print("-" * 60)
                print(refinement_analysis)
                print("-" * 60)

            # Use the refined results
            print(f"First round detected {len(detected_moles)} moles")
            print(f"Second round detected {len(refined_moles)} moles")
            detected_moles = refined_moles
            total_cost += second_cost

            # Save annotated image if requested
            if args.save_annotated:
                annotated_image = create_annotated_image(
                    image_path, detected_moles
                )
                annotated_path = f"{args.save_annotated}.2.jpg"
                cv2.imwrite(annotated_path, annotated_image)
                print(f"Saved annotated image to {annotated_path}")

    elapsed_time = time.time() - start_time

    # Compare detected moles with ground truth
    matches, unmatched_truth, unmatched_detected = compare_moles(
        ground_truth_moles, detected_moles, args.threshold
    )

    # Print results
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

    # Report timing and cost information
    if not args.no_refine:
        print(
            f"\nFirst round completed in {first_round_time:.2f} seconds (cost: ${first_cost:.6f})"
        )
        if "second_round_time" in locals():
            print(
                f"Second round completed in {second_round_time:.2f} seconds (cost: ${second_cost:.6f})"
            )
        print(f"Total processing time: {elapsed_time:.2f} seconds")
        print(f"Total estimated API cost: ${total_cost:.6f}")
    else:
        print(f"\nAPI request completed in {elapsed_time:.2f} seconds")
        print(f"Estimated API cost: ${total_cost:.6f}")

    return 0


def analyze_image_with_claude(
    image_path: pathlib.Path,
    api_key: str,
    model: str = "claude-3-opus-20240229",
    annotated_image: Optional[np.ndarray] = None,
    first_round_moles: Optional[List[Dict]] = None,
    previous_conversation: Optional[List[Dict]] = None,
) -> Tuple[
    List[Dict],
    float,
    Optional[str],
    Optional[str],
    Optional[List],
    Optional[str],
]:
    """Analyze an image with Claude API to detect moles using the Anthropic
    library.

    Args:
        image_path: Path to the image file
        api_key: Claude API key
        model: Claude model to use (default: claude-3-opus-20240229)
        annotated_image: Optional annotated image for refinement round
        first_round_moles: Optional list of moles detected in the first round
        previous_conversation: Optional conversation history from the first round

    Returns:
        Tuple containing:
        - List of detected moles with x, y coordinates
        - Estimated cost of the API call
        - Error message if the request failed, None otherwise
        - Analysis text from the first turn (None if refinement round)
        - Full conversation messages list that can be used in subsequent calls
        - Refinement analysis text if in refinement mode (None in first round)
    """
    # Determine if this is the first or refinement round
    is_refinement = (
        annotated_image is not None and first_round_moles is not None
    )

    # Get image data (either from file or from the annotated numpy array)
    if is_refinement:
        # Encode the annotated image
        success, img_encoded = cv2.imencode(".jpg", annotated_image)
        if not success:
            return [], 0.0, "Failed to encode annotated image", None
        image_data = img_encoded.tobytes()
    else:
        # Read the original image
        with open(image_path, "rb") as f:
            image_data = f.read()

    # Base64 encode the image for the API
    base64_image = base64.b64encode(image_data).decode("utf-8")

    # Initialize the Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    total_input_tokens = 0
    total_output_tokens = 0
    mole_analysis = ""

    try:
        if is_refinement:
            # Prepare numbered moles list for the prompt
            numbered_moles = "\n".join(
                [
                    f"{i+1}. ({m['x']}, {m['y']})"
                    for i, m in enumerate(first_round_moles)
                ]
            )

            # Setup image content
            image_content = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_image,
                },
            }

            # Initialize messages or use previous conversation
            messages = []
            refinement_analysis = ""

            if previous_conversation:
                # If we have previous conversation, use it
                messages = previous_conversation.copy()
            else:
                # First turn of refinement: Ask for analysis of the annotated image
                analysis_prompt = REFINEMENT_ANALYSIS_PROMPT_TEMPLATE.format(
                    numbered_moles=numbered_moles
                )

                # First turn: analyze the annotated image
                print(
                    "  Step 1: Asking Claude to analyze the annotated image..."
                )
                analysis_content = [
                    {"type": "text", "text": analysis_prompt},
                    image_content,
                ]

                analysis_response = client.messages.create(
                    model=model,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": analysis_content}],
                )

                # Track token usage
                input_tokens = 1000  # Approximate for image + prompt
                output_tokens = (
                    analysis_response.usage.output_tokens
                    if hasattr(analysis_response, "usage")
                    else 500
                )
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

                # Extract analysis text and add to conversation
                if analysis_response.content:
                    for block in analysis_response.content:
                        if block.type == "text":
                            refinement_analysis = block.text
                            print(
                                "  Refinement analysis received. Asking for coordinates..."
                            )
                            break

                # Build the message list for the next turn
                messages = [
                    {"role": "user", "content": analysis_content},
                    {"role": "assistant", "content": refinement_analysis},
                ]

            # Second turn: Ask for coordinates based on analysis
            # Add request for coordinates to the conversation
            messages.append(
                {"role": "user", "content": REFINEMENT_COORDINATES_PROMPT}
            )

            # Get the final coordinates from Claude
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                messages=messages,
            )

            # Add the response to the conversation history
            messages.append({"role": "assistant", "content": response.content})

            # Track token usage
            input_tokens = (
                len(refinement_analysis) // 4 if refinement_analysis else 500
            )  # Rough estimate of tokens
            output_tokens = (
                response.usage.output_tokens
                if hasattr(response, "usage")
                else 100
            )
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

        else:
            # Multi-turn approach for first analysis
            # Initialize conversation history
            messages = []

            # Setup image content object
            image_content = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_image,
                },
            }

            # First turn: Ask for qualitative analysis
            content = [
                {"type": "text", "text": ANALYSIS_PROMPT},
                image_content,
            ]

            # First turn: Get qualitative analysis
            print("  Step 1: Asking Claude to analyze the image...")
            analysis_response = client.messages.create(
                model=model,
                max_tokens=1500,
                messages=[{"role": "user", "content": content}],
            )

            # Track token usage
            input_tokens = 1000  # Approximate for image + prompt
            output_tokens = (
                analysis_response.usage.output_tokens
                if hasattr(analysis_response, "usage")
                else 500
            )
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            # Extract the analysis text and build conversation history
            if analysis_response.content:
                for block in analysis_response.content:
                    if block.type == "text":
                        mole_analysis = block.text
                        print("  Analysis received. Asking for coordinates...")
                        break

            # Build the conversation history
            messages = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": analysis_response.content},
            ]

            # Add coordinates request to conversation
            messages.append({"role": "user", "content": COORDINATES_PROMPT})

            # Second turn: Ask for coordinates based on analysis
            response = client.messages.create(
                model=model, max_tokens=1000, messages=messages
            )

            # Add response to conversation history
            messages.append({"role": "assistant", "content": response.content})

            # Track token usage for second turn
            input_tokens = len(mole_analysis) // 4  # Rough estimate of tokens
            output_tokens = (
                response.usage.output_tokens
                if hasattr(response, "usage")
                else 100
            )
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

        # Get pricing for the model from global dictionary
        # Default to Opus pricing if model is not found
        model_pricing = MODEL_PRICING.get(
            model, {"input": 15.0, "output": 75.0}  # Default to Opus pricing
        )

        # Calculate cost in dollars (use accumulated tokens for multi-turn)
        if not is_refinement:
            # Use the accumulated tokens from both turns
            input_cost = (total_input_tokens / 1_000_000) * model_pricing[
                "input"
            ]
            output_cost = (total_output_tokens / 1_000_000) * model_pricing[
                "output"
            ]
        else:
            # For refinement, we only have one turn
            input_cost = (total_input_tokens / 1_000_000) * model_pricing[
                "input"
            ]
            output_cost = (total_output_tokens / 1_000_000) * model_pricing[
                "output"
            ]

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
                                mole_analysis if not is_refinement else None,
                                messages if "messages" in locals() else None,
                                refinement_analysis if is_refinement else None,
                            )
                    except json.JSONDecodeError:
                        return (
                            [],
                            total_cost,
                            "Invalid JSON in Claude's response",
                            mole_analysis if not is_refinement else None,
                            messages if "messages" in locals() else None,
                            refinement_analysis if is_refinement else None,
                        )

        return (
            detected_moles,
            total_cost,
            None,
            mole_analysis if not is_refinement else None,
            messages if "messages" in locals() else None,
            (
                refinement_analysis
                if is_refinement and "refinement_analysis" in locals()
                else None
            ),
        )

    except anthropic.APIError as e:
        return [], 0.0, f"Anthropic API error: {str(e)}", None, None, None
    except Exception as e:
        return [], 0.0, f"Unexpected error: {str(e)}", None, None, None


def create_annotated_image(
    image_path: pathlib.Path, moles: List[Dict]
) -> np.ndarray:
    """Create an annotated image with markers for detected moles.

    Args:
        image_path: Path to the original image
        moles: List of detected moles with x,y coordinates

    Returns:
        Annotated image as a numpy array
    """
    # Load the original image
    image = mel.lib.image.load_image(image_path)

    # Create a copy for annotation
    annotated = image.copy()

    # Add numbered markers for each mole
    for i, mole in enumerate(moles):
        x, y = int(mole["x"]), int(mole["y"])

        # Draw a circle around the mole
        cv2.circle(annotated, (x, y), 10, (0, 0, 255), 2)

        # Add a number label
        cv2.putText(
            annotated,
            str(i + 1),
            (x + 15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    return annotated


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
