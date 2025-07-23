"""Automark moles using DINOv2 features from reference images."""

import argparse
import pathlib

import cv2
import torch
import torchvision.transforms as transforms

import mel.lib.dinov2
import mel.lib.image
import mel.rotomap.moles


def _existing_file_path(string):
    """Argparse type for validating that a file exists."""
    path = pathlib.Path(string)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File does not exist: {string}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Path is not a file: {string}")
    return path


def setup_parser(parser):
    parser.add_argument(
        "--reference",
        nargs="+",
        type=_existing_file_path,
        required=True,
        help="Paths to reference images with canonical mole locations.",
    )
    parser.add_argument(
        "--target",
        nargs="+",
        type=_existing_file_path,
        required=True,
        help="Paths to target images to mark moles in.",
    )
    parser.add_argument(
        "--debug-images",
        action="store_true",
        help="Save debug images showing patches and search areas.",
    )
    parser.add_argument(
        "--dino-size",
        type=str,
        choices=["small", "base", "large", "giant"],
        default="base",
        help="DINOv2 model size variant (default: base).",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Minimum similarity threshold for accepting matches (default: 0.7).",
    )
    parser.add_argument(
        "--extra-stem",
        help="Add an extra bit to the filename stem, e.g. '0.jpg.EXTRA.json'.",
    )


def process_single_target_image(
    target_path: pathlib.Path,
    aggregated_features: dict,
    model,
    transform,
    feature_dim: int,
    context_size: int,
    similarity_threshold: float,
    extra_stem: str,
    debug_images: bool,
) -> int:
    """Process a single target image to find and mark moles.

    Args:
        target_path: Path to the target image
        aggregated_features: Dictionary of aggregated features by UUID
        model: DINOv2 model
        transform: Image transformation pipeline
        feature_dim: Feature dimension of the model
        context_size: Context window size for feature extraction
        similarity_threshold: Minimum similarity threshold for matches
        extra_stem: Extra stem for filename
        debug_images: Whether to save debug images

    Returns:
        Number of matches found in this image
    """
    print(f"Processing target image: {target_path.name}")

    try:
        # Load target image and existing moles
        target_image = mel.lib.image.load_image(target_path)
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        target_moles = mel.rotomap.moles.load_image_moles(
            target_path, extra_stem=extra_stem
        )

        # Create lookup for existing moles by UUID
        existing_moles_by_uuid = {m["uuid"]: m for m in target_moles}
        canonical_uuids = {
            m["uuid"] for m in target_moles if m[mel.rotomap.moles.KEY_IS_CONFIRMED]
        }

        matches_in_this_image = 0

        # Try to match each reference mole
        for ref_uuid, ref_features in aggregated_features.items():
            # Skip if this mole is already marked canonical in target
            if ref_uuid in canonical_uuids:
                print(f"  Skipping mole {ref_uuid[:8]}: already canonical in target")
                continue

            # Use existing location as starting point, or center of image if not present
            if ref_uuid in existing_moles_by_uuid:
                start_x = existing_moles_by_uuid[ref_uuid]["x"]
                start_y = existing_moles_by_uuid[ref_uuid]["y"]
                print(
                    f"  Searching for mole {ref_uuid[:8]} starting from existing location ({start_x}, {start_y})"
                )
            else:
                # Start search from center of image if mole doesn't exist yet
                start_x = target_image.shape[1] // 2
                start_y = target_image.shape[0] // 2
                print(
                    f"  Searching for new mole {ref_uuid[:8]} starting from image center ({start_x}, {start_y})"
                )

            try:
                # Find best match location
                best_x, best_y, similarity = mel.lib.dinov2.find_best_contextual_match(
                    ref_features,
                    target_image,
                    start_x,
                    start_y,
                    context_size,
                    model,
                    transform,
                    feature_dim,
                    debug_images,
                    ref_uuid[:8] if debug_images else None,
                )

                # Check if similarity meets threshold
                if similarity >= similarity_threshold:
                    matches_in_this_image += 1

                    if ref_uuid in existing_moles_by_uuid:
                        # Update existing mole location
                        old_x = existing_moles_by_uuid[ref_uuid]["x"]
                        old_y = existing_moles_by_uuid[ref_uuid]["y"]
                        distance_moved = (
                            (best_x - old_x) ** 2 + (best_y - old_y) ** 2
                        ) ** 0.5

                        existing_moles_by_uuid[ref_uuid]["x"] = best_x
                        existing_moles_by_uuid[ref_uuid]["y"] = best_y

                        print(
                            f"    Updated mole {ref_uuid[:8]}: ({old_x}, {old_y}) -> ({best_x}, {best_y}) "
                            f"(moved {distance_moved:.1f}px, similarity: {similarity:.3f})"
                        )
                    else:
                        # Add new mole
                        new_mole = {
                            "uuid": ref_uuid,
                            "x": best_x,
                            "y": best_y,
                            mel.rotomap.moles.KEY_IS_CONFIRMED: False,  # Non-canonical
                        }
                        target_moles.append(new_mole)
                        existing_moles_by_uuid[ref_uuid] = new_mole

                        print(
                            f"    Added new mole {ref_uuid[:8]} at ({best_x}, {best_y}) "
                            f"(similarity: {similarity:.3f})"
                        )
                else:
                    print(
                        f"    Mole {ref_uuid[:8]}: similarity {similarity:.3f} below threshold {similarity_threshold}"
                    )

            except Exception as e:
                print(f"    Error matching mole {ref_uuid[:8]}: {e}")
                continue

        # Save updated moles
        if matches_in_this_image > 0:
            mel.rotomap.moles.save_image_moles(
                target_moles, target_path, extra_stem=extra_stem
            )
            print(f"  Saved {matches_in_this_image} matches to {target_path.name}")
        else:
            print(f"  No matches found above threshold in {target_path.name}")

        return matches_in_this_image

    except Exception as e:
        print(f"Error processing target image {target_path}: {e}")
        return 0


def process_args(args) -> int:
    """Main processing function that orchestrates the automark3 workflow.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Extract arguments
    reference_paths = args.reference
    target_paths = args.target
    debug_images = args.debug_images
    dino_size = args.dino_size
    similarity_threshold = args.similarity_threshold
    extra_stem = args.extra_stem

    context_size = 910  # Large context window for rich semantic features

    # Step 1: Load DINOv2 model
    print(f"Loading DINOv2 model ({dino_size})...")
    try:
        model, feature_dim = mel.lib.dinov2.load_dinov2_model(dino_size)
        print(f"DINOv2 model loaded successfully with {feature_dim} feature dimensions")
    except RuntimeError as e:
        print(f"Error loading DINOv2 model: {e}")
        return 1

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Step 2: Load and aggregate reference features
    try:
        aggregated_features, aggregation_metadata, reference_moles_by_uuid = (
            mel.lib.dinov2.load_and_aggregate_reference_features(
                reference_paths, model, transform, feature_dim, context_size
            )
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1

    # Step 3: Process each target image
    print(f"\nProcessing {len(target_paths)} target images...")
    total_matches_found = 0

    for target_path in target_paths:
        matches_found = process_single_target_image(
            target_path,
            aggregated_features,
            model,
            transform,
            feature_dim,
            context_size,
            similarity_threshold,
            extra_stem,
            debug_images,
        )
        total_matches_found += matches_found

    print(
        f"\nAutomark3 completed: {total_matches_found} total matches found across all target images"
    )
    return 0


# -----------------------------------------------------------------------------
# Copyright (C) 2025 Angelos Evripiotis.
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
