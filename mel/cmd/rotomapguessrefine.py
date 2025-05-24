"""Refine non-canonical mole locations using DINOv2 feature matching."""

import argparse
import pathlib

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

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
        "SRC_JPG",
        type=_existing_file_path,
        help="Path to the source image with canonical mole locations.",
    )
    parser.add_argument(
        "TGT_JPG",
        type=_existing_file_path,
        help="Path to the target image with non-canonical moles to refine.",
    )
    parser.add_argument(
        "--search-radius",
        type=int,
        default=35,
        help="Search radius around non-canonical mole location (default: 35 pixels).",
    )


def load_dinov2_model():
    """Load the DINOv2 model for feature extraction."""
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(
            "Failed to load DINOv2 model. Please ensure you have internet access "
            "and the required dependencies. Error: " + str(e)
        ) from e


def extract_patch_features(
    image, center_x, center_y, patch_size, model, transform
):
    """Extract DINOv2 features from a 140x140 pixel patch centered at
    (center_x, center_y)."""
    half_size = patch_size // 2

    # Extract patch with bounds checking
    y_start = max(0, center_y - half_size)
    y_end = min(image.shape[0], center_y + half_size)
    x_start = max(0, center_x - half_size)
    x_end = min(image.shape[1], center_x + half_size)

    patch = image[y_start:y_end, x_start:x_end]

    # Pad if necessary to ensure patch is patch_size x patch_size
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        padded_patch = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
        padded_patch[: patch.shape[0], : patch.shape[1]] = patch
        patch = padded_patch
    elif patch.shape[0] > patch_size or patch.shape[1] > patch_size:
        patch = patch[:patch_size, :patch_size]

    # Convert to tensor and normalize
    patch_tensor = transform(patch).unsqueeze(0)

    with torch.no_grad():
        features = model(patch_tensor)

    return features.squeeze(0)  # Remove batch dimension


def find_best_match_location(
    src_features,
    tgt_image,
    center_x,
    center_y,
    search_radius,
    patch_size,
    model,
    transform,
):
    """Find the best matching location within search_radius of the initial
    guess."""
    best_similarity = -float("inf")
    best_x, best_y = center_x, center_y

    # Search in a grid around the initial location
    for dy in range(
        -search_radius, search_radius + 1, 2
    ):  # Step by 2 for efficiency
        for dx in range(-search_radius, search_radius + 1, 2):
            candidate_x = center_x + dx
            candidate_y = center_y + dy

            # Skip if candidate location is too close to image borders
            half_size = patch_size // 2
            if (
                candidate_x - half_size < 0
                or candidate_x + half_size >= tgt_image.shape[1]
                or candidate_y - half_size < 0
                or candidate_y + half_size >= tgt_image.shape[0]
            ):
                continue

            # Extract features from candidate location
            candidate_features = extract_patch_features(
                tgt_image,
                candidate_x,
                candidate_y,
                patch_size,
                model,
                transform,
            )

            # Calculate cosine similarity
            similarity = torch.cosine_similarity(
                src_features, candidate_features, dim=0
            ).item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_x, best_y = candidate_x, candidate_y

    return best_x, best_y, best_similarity


def process_args(args):
    src_path = args.SRC_JPG
    tgt_path = args.TGT_JPG
    search_radius = args.search_radius

    # Load images
    try:
        src_image = mel.lib.image.load_image(src_path)
        tgt_image = mel.lib.image.load_image(tgt_path)

        # Convert BGR to RGB for DINOv2
        src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        tgt_image = cv2.cvtColor(tgt_image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading images: {e}")
        return 1

    # Load moles
    try:
        src_moles = mel.rotomap.moles.load_image_moles(src_path)
        tgt_moles = mel.rotomap.moles.load_image_moles(tgt_path)
    except Exception as e:
        print(f"Error loading moles: {e}")
        return 1

    # Filter to get canonical moles from source and non-canonical from target
    src_canonical_moles = [
        m for m in src_moles if m[mel.rotomap.moles.KEY_IS_CONFIRMED]
    ]

    # Find non-canonical moles in target that have matching UUIDs with canonical moles in source
    src_canonical_uuids = {m["uuid"] for m in src_canonical_moles}
    tgt_non_canonical_to_refine = [
        m
        for m in tgt_moles
        if not m[mel.rotomap.moles.KEY_IS_CONFIRMED]
        and m["uuid"] in src_canonical_uuids
    ]

    if not tgt_non_canonical_to_refine:
        print(
            "No non-canonical moles in target that match canonical moles in source"
        )
        return 0

    print(
        f"Found {len(tgt_non_canonical_to_refine)} non-canonical moles to refine"
    )

    # Load DINOv2 model
    try:
        model = load_dinov2_model()
        print("DINOv2 model loaded successfully")
    except RuntimeError as e:
        print(f"Error loading DINOv2 model: {e}")
        return 1

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # Create lookup dict for src canonical moles
    src_mole_lookup = {m["uuid"]: m for m in src_canonical_moles}

    patch_size = 70
    refined_count = 0

    # Process each non-canonical mole in target
    for tgt_mole in tgt_non_canonical_to_refine:
        uuid = tgt_mole["uuid"]
        src_mole = src_mole_lookup[uuid]

        print(f"Refining mole {uuid}...")

        # Extract features from source mole location (canonical)
        try:
            src_features = extract_patch_features(
                src_image,
                src_mole["x"],
                src_mole["y"],
                patch_size,
                model,
                transform,
            )
        except Exception as e:
            print(f"Error extracting source features for mole {uuid}: {e}")
            continue

        # Find best matching location in target image around the non-canonical location
        try:
            best_x, best_y, similarity = find_best_match_location(
                src_features,
                tgt_image,
                tgt_mole["x"],
                tgt_mole["y"],
                search_radius,
                patch_size,
                model,
                transform,
            )

            # Update the mole location if we found a better match
            old_x, old_y = tgt_mole["x"], tgt_mole["y"]
            distance_moved = (
                (best_x - old_x) ** 2 + (best_y - old_y) ** 2
            ) ** 0.5

            if distance_moved > 1:  # Only update if we moved more than 1 pixel
                tgt_mole["x"] = best_x
                tgt_mole["y"] = best_y
                refined_count += 1
                print(
                    f"  Refined from ({old_x}, {old_y}) to ({best_x}, {best_y}) "
                    f"(moved {distance_moved:.1f} pixels, similarity: {similarity:.3f})"
                )
            else:
                print(f"  No refinement needed (similarity: {similarity:.3f})")

        except Exception as e:
            print(f"Error refining mole {uuid}: {e}")
            continue

    # Save refined moles if any were updated
    if refined_count > 0:
        try:
            mel.rotomap.moles.save_image_moles(tgt_moles, tgt_path)
            print(
                f"Successfully refined {refined_count} mole locations in {tgt_path}"
            )
        except Exception as e:
            print(f"Error saving refined moles: {e}")
            return 1
    else:
        print("No moles required refinement")

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
