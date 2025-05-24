"""Refine non-canonical mole locations using DINOv2 feature matching."""

import argparse
import pathlib

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

import mel.lib.image
import mel.lib.math
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
    parser.add_argument(
        "--debug-images",
        action="store_true",
        help="Save debug images showing source patches and target search areas.",
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


def calculate_alignment_transform(
    src_mole, tgt_mole, src_canonical_moles, tgt_canonical_moles
):
    """Calculate scale and rotation to align source with target based on
    neighboring moles."""
    src_uuid = src_mole["uuid"]
    tgt_uuid = tgt_mole["uuid"]
    assert src_uuid == tgt_uuid

    # Find common neighboring canonical moles
    src_canonical_uuids = {m["uuid"] for m in src_canonical_moles}
    tgt_canonical_uuids = {m["uuid"] for m in tgt_canonical_moles}
    common_uuids = src_canonical_uuids & tgt_canonical_uuids
    common_uuids.discard(src_uuid)  # Remove the target mole itself

    if not common_uuids:
        # No common neighbors, return identity transform
        return 1.0, 0.0, None

    # Create lookup dictionaries
    src_mole_lookup = {
        m["uuid"]: (m["x"], m["y"]) for m in src_canonical_moles
    }
    tgt_mole_lookup = {
        m["uuid"]: (m["x"], m["y"]) for m in tgt_canonical_moles
    }

    src_target_pos = np.array([src_mole["x"], src_mole["y"]])
    tgt_target_pos = np.array([tgt_mole["x"], tgt_mole["y"]])

    # Find nearest common neighbor
    nearest_common_uuid = min(
        common_uuids,
        key=lambda u: mel.lib.math.distance_sq_2d(
            src_mole_lookup[u], src_target_pos
        ),
    )

    # Calculate distances from target mole to nearest neighbor
    src_neighbor_pos = np.array(src_mole_lookup[nearest_common_uuid])
    tgt_neighbor_pos = np.array(tgt_mole_lookup[nearest_common_uuid])

    src_distance = mel.lib.math.distance_2d(src_neighbor_pos, src_target_pos)
    tgt_distance = mel.lib.math.distance_2d(tgt_neighbor_pos, tgt_target_pos)

    # Calculate scale factor
    if src_distance > 0:
        scale = tgt_distance / src_distance
    else:
        scale = 1.0

    # Calculate rotation
    src_angle = mel.lib.math.angle(src_neighbor_pos - src_target_pos)
    tgt_angle = mel.lib.math.angle(tgt_neighbor_pos - tgt_target_pos)
    rotation_degrees = tgt_angle - src_angle

    return scale, rotation_degrees, nearest_common_uuid


def apply_transform_to_point(point, scale, rotation_degrees, center):
    """Apply scale and rotation transformation to a point around a center."""
    # Translate to origin
    translated = point - center

    # Apply scale
    scaled = translated * scale

    # Apply rotation
    rotation_radians = np.radians(rotation_degrees)
    cos_rot = np.cos(rotation_radians)
    sin_rot = np.sin(rotation_radians)

    rotation_matrix = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])

    rotated = rotation_matrix @ scaled

    # Translate back
    return rotated + center


def transform_image_patch(
    image, center_x, center_y, patch_size, scale, rotation_degrees
):
    """Extract and transform an image patch with given scale and rotation."""
    half_size = int(patch_size * scale) // 2 + 50  # Extra margin for rotation

    # Extract larger patch to allow for rotation and scaling
    y_start = max(0, center_y - half_size)
    y_end = min(image.shape[0], center_y + half_size)
    x_start = max(0, center_x - half_size)
    x_end = min(image.shape[1], center_x + half_size)

    patch = image[y_start:y_end, x_start:x_end]

    # Apply scale
    if scale != 1.0:
        patch = mel.lib.image.scale_image(patch, scale)

    # Apply rotation
    if abs(rotation_degrees) > 0.1:
        patch = mel.lib.image.rotated(patch, rotation_degrees)

    # Center crop to desired size
    if patch.shape[0] >= patch_size and patch.shape[1] >= patch_size:
        patch_center_y, patch_center_x = (
            patch.shape[0] // 2,
            patch.shape[1] // 2,
        )
        half_patch = patch_size // 2
        y_start = patch_center_y - half_patch
        y_end = patch_center_y + half_patch
        x_start = patch_center_x - half_patch
        x_end = patch_center_x + half_patch
        patch = patch[y_start:y_end, x_start:x_end]
    else:
        # Pad if too small
        padded_patch = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
        y_offset = (patch_size - patch.shape[0]) // 2
        x_offset = (patch_size - patch.shape[1]) // 2
        padded_patch[
            y_offset : y_offset + patch.shape[0],
            x_offset : x_offset + patch.shape[1],
        ] = patch
        patch = padded_patch

    return patch


def save_debug_patch(patch, filename):
    """Save a patch for debugging purposes."""
    try:
        cv2.imwrite(filename, patch)
        print(f"  Debug: Saved patch to {filename}")
    except Exception as e:
        print(f"  Debug: Failed to save patch to {filename}: {e}")


def save_debug_search_area(
    image, center_x, center_y, search_radius, patch_size, filename
):
    """Save the target search area for debugging."""
    try:
        # Calculate search area bounds
        half_patch = patch_size // 2
        search_left = max(0, center_x - search_radius - half_patch)
        search_right = min(
            image.shape[1], center_x + search_radius + half_patch
        )
        search_top = max(0, center_y - search_radius - half_patch)
        search_bottom = min(
            image.shape[0], center_y + search_radius + half_patch
        )

        # Extract search area
        search_area = image[
            search_top:search_bottom, search_left:search_right
        ].copy()

        # Draw search grid and center marker
        # Center point in search area coordinates
        center_in_area_x = center_x - search_left
        center_in_area_y = center_y - search_top

        # Draw center cross
        cv2.line(
            search_area,
            (center_in_area_x - 10, center_in_area_y),
            (center_in_area_x + 10, center_in_area_y),
            (0, 255, 0),
            2,
        )
        cv2.line(
            search_area,
            (center_in_area_x, center_in_area_y - 10),
            (center_in_area_x, center_in_area_y + 10),
            (0, 255, 0),
            2,
        )

        # Draw search radius circle
        cv2.circle(
            search_area,
            (center_in_area_x, center_in_area_y),
            search_radius,
            (255, 0, 0),
            2,
        )

        # Draw patch size box
        cv2.rectangle(
            search_area,
            (center_in_area_x - half_patch, center_in_area_y - half_patch),
            (center_in_area_x + half_patch, center_in_area_y + half_patch),
            (0, 0, 255),
            2,
        )

        cv2.imwrite(filename, search_area)
        print(f"  Debug: Saved search area to {filename}")
    except Exception as e:
        print(f"  Debug: Failed to save search area to {filename}: {e}")


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


def extract_patch_features_from_patch(patch, model, transform):
    """Extract DINOv2 features from a pre-processed patch."""
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
    debug_images = args.debug_images

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

    # Filter to get canonical moles from source and target
    src_canonical_moles = [
        m for m in src_moles if m[mel.rotomap.moles.KEY_IS_CONFIRMED]
    ]
    tgt_canonical_moles = [
        m for m in tgt_moles if m[mel.rotomap.moles.KEY_IS_CONFIRMED]
    ]

    if not src_canonical_moles:
        print("Error: No canonical moles found in source image")
        return 1
    if not tgt_canonical_moles:
        print("Error: No canonical moles found in target image")
        return 1

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

        # Calculate alignment transformation based on neighboring moles
        scale, rotation_degrees, neighbor_uuid = calculate_alignment_transform(
            src_mole, tgt_mole, src_canonical_moles, tgt_canonical_moles
        )

        if neighbor_uuid:
            print(
                f"  Using neighbor {neighbor_uuid} for alignment: scale={scale:.3f}, rotation={rotation_degrees:.1f}°"
            )
        else:
            print("  No common neighbors found, using identity transform")

        # Extract and transform features from source mole location (canonical)
        try:
            if neighbor_uuid:
                # Use transformed patch
                src_patch = transform_image_patch(
                    src_image,
                    src_mole["x"],
                    src_mole["y"],
                    patch_size,
                    scale,
                    rotation_degrees,
                )
                src_features = extract_patch_features_from_patch(
                    src_patch, model, transform
                )

                # Transform the source mole position for reference
                src_center = np.array([src_mole["x"], src_mole["y"]])
                transformed_src_pos = apply_transform_to_point(
                    src_center, scale, rotation_degrees, src_center
                )

                # Save debug image for transformed source patch
                if debug_images:
                    save_debug_patch(src_patch, f"{uuid}_src_transformed.jpg")
            else:
                # Use original patch
                src_features = extract_patch_features(
                    src_image,
                    src_mole["x"],
                    src_mole["y"],
                    patch_size,
                    model,
                    transform,
                )

                # Save debug image for original source patch
                if debug_images:
                    # Extract just the patch without features for debug
                    half_size = patch_size // 2
                    y_start = max(0, src_mole["y"] - half_size)
                    y_end = min(src_image.shape[0], src_mole["y"] + half_size)
                    x_start = max(0, src_mole["x"] - half_size)
                    x_end = min(src_image.shape[1], src_mole["x"] + half_size)
                    src_patch_img = src_image[y_start:y_end, x_start:x_end]

                    # Pad if necessary
                    if (
                        src_patch_img.shape[0] < patch_size
                        or src_patch_img.shape[1] < patch_size
                    ):
                        padded_patch = np.zeros(
                            (patch_size, patch_size, 3),
                            dtype=src_patch_img.dtype,
                        )
                        padded_patch[
                            : src_patch_img.shape[0], : src_patch_img.shape[1]
                        ] = src_patch_img
                        src_patch_img = padded_patch
                    elif (
                        src_patch_img.shape[0] > patch_size
                        or src_patch_img.shape[1] > patch_size
                    ):
                        src_patch_img = src_patch_img[:patch_size, :patch_size]

                    save_debug_patch(src_patch_img, f"{uuid}_src_original.jpg")
        except Exception as e:
            print(f"Error extracting source features for mole {uuid}: {e}")
            continue

        # Save debug image for target search area
        if debug_images:
            save_debug_search_area(
                tgt_image,
                tgt_mole["x"],
                tgt_mole["y"],
                search_radius,
                patch_size,
                f"{uuid}_tgt_search_area.jpg",
            )

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

                # Save debug image for final refined patch
                if debug_images:
                    try:
                        half_size = patch_size // 2
                        y_start = max(0, best_y - half_size)
                        y_end = min(tgt_image.shape[0], best_y + half_size)
                        x_start = max(0, best_x - half_size)
                        x_end = min(tgt_image.shape[1], best_x + half_size)
                        refined_patch = tgt_image[y_start:y_end, x_start:x_end]

                        # Pad if necessary
                        if (
                            refined_patch.shape[0] < patch_size
                            or refined_patch.shape[1] < patch_size
                        ):
                            padded_patch = np.zeros(
                                (patch_size, patch_size, 3),
                                dtype=refined_patch.dtype,
                            )
                            padded_patch[
                                : refined_patch.shape[0],
                                : refined_patch.shape[1],
                            ] = refined_patch
                            refined_patch = padded_patch
                        elif (
                            refined_patch.shape[0] > patch_size
                            or refined_patch.shape[1] > patch_size
                        ):
                            refined_patch = refined_patch[
                                :patch_size, :patch_size
                            ]

                        save_debug_patch(
                            refined_patch, f"{uuid}_tgt_refined.jpg"
                        )
                    except Exception as e:
                        print(f"  Debug: Failed to save refined patch: {e}")
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
