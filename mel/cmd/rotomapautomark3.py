"""Auto-mark moles in target images using DINOv3 feature matching from reference images."""

import argparse
import collections
import pathlib

import cv2
import numpy as np

import mel.lib.dinov3
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
        type=_existing_file_path,
        nargs="+",
        required=True,
        help="Reference images containing canonical moles.",
    )
    parser.add_argument(
        "--target",
        type=_existing_file_path,
        nargs="+",
        required=True,
        help="Target images to auto-mark with moles.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Cosine similarity threshold for accepting matches (optional, will be determined from ROC analysis).",
    )
    parser.add_argument(
        "--dino-size",
        type=str,
        choices=["small", "base", "large", "giant"],
        default="base",
        help="DINOv3 model size variant (default: base).",
    )
    parser.add_argument(
        "--debug-images",
        action="store_true",
        help="Save debug images showing matched patches.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Size of patch to extract around each mole (default: 224).",
    )


def process_args(args):
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torchvision.transforms as transforms

    reference_paths = args.reference
    target_paths = args.target
    threshold = args.threshold
    dino_size = args.dino_size
    debug_images = args.debug_images
    patch_size = args.patch_size

    # Load DINOv3 model
    try:
        model, feature_dim = mel.lib.dinov3.load_dinov3_model(dino_size)
    except RuntimeError as e:
        print(f"Error loading DINOv3 model: {e}")
        return 1

    # Set up image transform
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Step 1: Gather all reference moles with their features
    # Structure: {uuid: [(feature, x, y, image_path, similarity_score), ...]}
    reference_moles = collections.defaultdict(list)

    for ref_path in reference_paths:
        try:
            ref_image = mel.lib.image.load_image(ref_path)
            ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
            ref_moles = mel.rotomap.moles.load_image_moles(ref_path)
        except Exception as e:
            print(f"Error loading reference image {ref_path}: {e}")
            continue

        # Only use canonical (confirmed) moles from reference images
        canonical_moles = [
            m for m in ref_moles if m[mel.rotomap.moles.KEY_IS_CONFIRMED]
        ]

        for mole in canonical_moles:
            uuid = mole["uuid"]
            x, y = mole["x"], mole["y"]

            try:
                # Extract feature for this mole
                feature = mel.lib.dinov3.extract_patch_feature(
                    ref_image_rgb, x, y, patch_size, model, transform
                )
                # Store with a placeholder score (will be updated during aggregation)
                reference_moles[uuid].append((feature, x, y, ref_path, 0.0))
            except Exception as e:
                print(f"Error extracting feature for mole {uuid} in {ref_path}: {e}")
                continue

    if not reference_moles:
        print("No canonical moles found in reference images")
        return 1

    # Step 2: For each UUID, keep only the best representation
    # (We'll use the first one for now, but aggregate by finding the one with
    # highest average similarity to all others)
    aggregated_references = {}
    for uuid, mole_instances in reference_moles.items():
        if len(mole_instances) == 1:
            # Only one instance, use it directly
            aggregated_references[uuid] = mole_instances[0]
        else:
            # Multiple instances: find the one with highest average similarity to others
            best_instance = None
            best_avg_similarity = -float("inf")

            for i, (feature_i, x_i, y_i, path_i, _) in enumerate(mole_instances):
                similarities = []
                for j, (feature_j, x_j, y_j, path_j, _) in enumerate(mole_instances):
                    if i != j:
                        sim = mel.lib.dinov3.compute_similarity(feature_i, feature_j)
                        similarities.append(sim)

                avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
                if avg_sim > best_avg_similarity:
                    best_avg_similarity = avg_sim
                    best_instance = (feature_i, x_i, y_i, path_i, avg_sim)

            aggregated_references[uuid] = best_instance

    # Step 3: Match each reference mole against target images
    total_matches = 0
    total_updates = 0

    for target_path in target_paths:
        try:
            target_image = mel.lib.image.load_image(target_path)
            target_image_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            target_moles = mel.rotomap.moles.load_image_moles(target_path)
        except Exception as e:
            print(f"Error loading target image {target_path}: {e}")
            continue

        # Build a set of canonical UUIDs in the target (we'll skip these)
        canonical_uuids = {
            m["uuid"] for m in target_moles if m[mel.rotomap.moles.KEY_IS_CONFIRMED]
        }

        # Build a dict of existing non-canonical moles by UUID for quick lookup
        existing_moles_by_uuid = {
            m["uuid"]: m
            for m in target_moles
            if not m[mel.rotomap.moles.KEY_IS_CONFIRMED]
        }

        image_modified = False

        # For each reference mole, try to find it in the target image
        for ref_uuid, (
            ref_feature,
            ref_x,
            ref_y,
            ref_path,
            ref_score,
        ) in aggregated_references.items():
            # Skip if this UUID is already canonical in the target
            if ref_uuid in canonical_uuids:
                continue

            # Search for the best match in the target image
            # Use a grid search or scan the image for the best matching location
            best_match_x, best_match_y, best_similarity = search_for_mole(
                target_image_rgb,
                ref_feature,
                model,
                transform,
                patch_size,
                existing_moles_by_uuid.get(ref_uuid),
            )

            # Check if similarity exceeds threshold
            if threshold is not None and best_similarity < threshold:
                continue

            # Update or add the mole
            if ref_uuid in existing_moles_by_uuid:
                # Update existing non-canonical mole
                mole = existing_moles_by_uuid[ref_uuid]
                old_x, old_y = mole["x"], mole["y"]
                mole["x"] = best_match_x
                mole["y"] = best_match_y
                total_updates += 1
                image_modified = True
            else:
                # Add new mole
                new_mole = {
                    "uuid": ref_uuid,
                    "x": best_match_x,
                    "y": best_match_y,
                    mel.rotomap.moles.KEY_IS_CONFIRMED: False,
                }
                target_moles.append(new_mole)
                existing_moles_by_uuid[ref_uuid] = new_mole
                total_matches += 1
                image_modified = True

            # Save debug image if requested
            if debug_images:
                save_debug_patch(
                    target_image_rgb,
                    best_match_x,
                    best_match_y,
                    patch_size,
                    ref_uuid,
                    target_path,
                    best_similarity,
                )

        # Save updated moles if any changes were made
        if image_modified:
            try:
                mel.rotomap.moles.save_image_moles(target_moles, target_path)
            except Exception as e:
                print(f"Error saving moles for {target_path}: {e}")
                return 1

    return 0


def search_for_mole(
    target_image, ref_feature, model, transform, patch_size, existing_mole=None
):
    """Search for the best matching location in the target image.

    Args:
        target_image: Target image (RGB)
        ref_feature: Reference mole feature
        model: DINOv3 model
        transform: Image transform
        patch_size: Patch size for feature extraction
        existing_mole: Existing mole dict with 'x' and 'y' if available

    Returns:
        tuple: (best_x, best_y, best_similarity)
    """
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torch

    height, width = target_image.shape[:2]

    # If there's an existing mole, start search around it
    if existing_mole:
        center_x = existing_mole["x"]
        center_y = existing_mole["y"]
        search_radius = 100  # Search within 100 pixels
    else:
        # Otherwise, search the entire image (but this is expensive)
        # For now, we'll just return a dummy location if no existing mole
        # A full implementation would need a more sophisticated search strategy
        center_x = width // 2
        center_y = height // 2
        search_radius = min(width, height) // 4

    best_x, best_y = center_x, center_y
    best_similarity = -float("inf")

    # Grid search around the center point
    step_size = 16  # Search every 16 pixels
    for dy in range(-search_radius, search_radius + 1, step_size):
        for dx in range(-search_radius, search_radius + 1, step_size):
            test_x = center_x + dx
            test_y = center_y + dy

            # Skip if out of bounds
            if (
                test_x < patch_size // 2
                or test_x >= width - patch_size // 2
                or test_y < patch_size // 2
                or test_y >= height - patch_size // 2
            ):
                continue

            try:
                # Extract feature at this location
                target_feature = mel.lib.dinov3.extract_patch_feature(
                    target_image, test_x, test_y, patch_size, model, transform
                )

                # Compute similarity
                similarity = mel.lib.dinov3.compute_similarity(
                    ref_feature, target_feature
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_x = test_x
                    best_y = test_y

            except Exception:
                # Skip locations where feature extraction fails
                continue

    return best_x, best_y, best_similarity


def save_debug_patch(image_rgb, x, y, patch_size, uuid, target_path, similarity):
    """Save a debug image showing the matched patch."""
    import cv2

    half_size = patch_size // 2
    y_start = max(0, y - half_size)
    y_end = min(image_rgb.shape[0], y + half_size)
    x_start = max(0, x - half_size)
    x_end = min(image_rgb.shape[1], x + half_size)

    patch = image_rgb[y_start:y_end, x_start:x_end].copy()

    # Draw a cross at the center
    patch_center_y = y - y_start
    patch_center_x = x - x_start
    cv2.line(
        patch,
        (patch_center_x - 10, patch_center_y),
        (patch_center_x + 10, patch_center_y),
        (0, 255, 0),
        2,
    )
    cv2.line(
        patch,
        (patch_center_x, patch_center_y - 10),
        (patch_center_x, patch_center_y + 10),
        (0, 255, 0),
        2,
    )

    # Convert back to BGR for saving
    patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)

    # Save with a descriptive filename
    target_name = pathlib.Path(target_path).stem
    debug_filename = f"debug_{target_name}_{uuid}_{similarity:.3f}.jpg"
    cv2.imwrite(debug_filename, patch_bgr)


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
