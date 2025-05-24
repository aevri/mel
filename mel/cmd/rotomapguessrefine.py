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
    """Load the DINOv2 model for dense feature extraction."""
    try:
        # Load DINOv2 model that can return patch tokens for dense matching
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

        # Create a wrapper to extract both CLS and patch tokens
        class DenseFeatureExtractor:
            def __init__(self, model):
                self.model = model
                self.model.eval()

            def __call__(self, x):
                # Get features - this returns [batch, feature_dim] (CLS token only)
                cls_features = self.model(x)

                # To get patch tokens, we need to access intermediate outputs
                # For now, let's use a different approach - get features at multiple scales
                return cls_features

            def extract_patch_features(self, x):
                """Extract patch-level features for dense matching."""
                # For dense matching, we need patch tokens, not just CLS
                # This requires accessing the model's intermediate representations

                # Temporarily store the forward hook to capture patch features
                patch_features = []

                def hook_fn(module, input, output):
                    # Capture the output from the last transformer block before pooling
                    if hasattr(output, "shape") and len(output.shape) == 3:
                        patch_features.append(output)

                # Register hook on the last normalization layer
                hook = self.model.norm.register_forward_hook(hook_fn)

                try:
                    # Run forward pass
                    _ = self.model(x)
                    if patch_features:
                        # Return all tokens [batch, seq_len, feature_dim]
                        return patch_features[0]
                    else:
                        # Fallback to CLS token reshaped
                        cls_out = self.model(x)
                        return cls_out.unsqueeze(1)  # [batch, 1, feature_dim]
                finally:
                    hook.remove()

        return DenseFeatureExtractor(model)
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
    return 0.75, 0, "FAKE"

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
    """Transform the full image first, then extract patch to preserve maximum
    context."""
    # Transform the full image first to preserve context
    transformed_image = image.copy()

    # Apply scale to the full image
    if scale != 1.0:
        transformed_image = mel.lib.image.scale_image(transformed_image, scale)
        # Update center coordinates for the scaled image
        scaled_center_x = int(center_x * scale)
        scaled_center_y = int(center_y * scale)
    else:
        scaled_center_x = center_x
        scaled_center_y = center_y

    # Apply rotation to the full image
    if abs(rotation_degrees) > 0.1:
        transformed_image = mel.lib.image.rotated(
            transformed_image, rotation_degrees
        )

        # Transform the mole coordinates for the rotated image
        # Rotation is applied around the center of the image
        img_center_x = transformed_image.shape[1] // 2
        img_center_y = transformed_image.shape[0] // 2

        # Apply rotation transformation to the mole center
        rotated_center = apply_transform_to_point(
            np.array([scaled_center_x, scaled_center_y]),
            1.0,  # No additional scaling
            rotation_degrees,
            np.array([img_center_x, img_center_y]),
        )
        scaled_center_x = int(rotated_center[0])
        scaled_center_y = int(rotated_center[1])

    # Now extract the patch from the transformed image
    half_size = patch_size // 2
    y_start = max(0, scaled_center_y - half_size)
    y_end = min(transformed_image.shape[0], scaled_center_y + half_size)
    x_start = max(0, scaled_center_x - half_size)
    x_end = min(transformed_image.shape[1], scaled_center_x + half_size)

    patch = transformed_image[y_start:y_end, x_start:x_end]

    # Pad if necessary to ensure patch is patch_size x patch_size
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        padded_patch = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
        y_offset = (patch_size - patch.shape[0]) // 2
        x_offset = (patch_size - patch.shape[1]) // 2
        padded_patch[
            y_offset : y_offset + patch.shape[0],
            x_offset : x_offset + patch.shape[1],
        ] = patch
        patch = padded_patch
    elif patch.shape[0] > patch_size or patch.shape[1] > patch_size:
        # Center crop if too large
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


def save_similarity_heatmap(
    image,
    center_x,
    center_y,
    search_radius,
    positions,
    similarities,
    step,
    filename,
):
    """Save a heatmap showing similarity scores across the search area."""
    try:
        # Calculate search area bounds
        half_patch = 112  # Use a reasonable patch size for visualization
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

        # Create heatmap overlay
        heatmap = np.zeros(search_area.shape[:2], dtype=np.float32)

        # Normalize similarities to 0-1 range for visualization
        if similarities:
            valid_similarities = [
                s for s in similarities if s >= -0.5
            ]  # Filter out invalid positions
            if valid_similarities:
                min_sim = min(valid_similarities)
                max_sim = max(valid_similarities)
                sim_range = max_sim - min_sim if max_sim > min_sim else 1.0

                # Draw similarity blocks on heatmap
                for (pos_x, pos_y), similarity in zip(positions, similarities):
                    if similarity < -0.5:  # Skip invalid positions
                        continue

                    # Convert global coordinates to search area coordinates
                    local_x = pos_x - search_left
                    local_y = pos_y - search_top

                    # Normalize similarity to 0-1
                    normalized_sim = (similarity - min_sim) / sim_range

                    # Draw a block centered at the position
                    block_half = step // 2
                    y_start = max(0, local_y - block_half)
                    y_end = min(heatmap.shape[0], local_y + block_half)
                    x_start = max(0, local_x - block_half)
                    x_end = min(heatmap.shape[1], local_x + block_half)

                    heatmap[y_start:y_end, x_start:x_end] = normalized_sim

        # Convert heatmap to red channel overlay
        heatmap_colored = np.zeros(search_area.shape, dtype=np.uint8)
        heatmap_colored[:, :, 2] = (heatmap * 255).astype(
            np.uint8
        )  # Red channel

        # Blend with original image (70% original, 30% heatmap)
        blended = cv2.addWeighted(search_area, 0.7, heatmap_colored, 0.3, 0)

        # Draw center cross
        center_in_area_x = center_x - search_left
        center_in_area_y = center_y - search_top
        cv2.line(
            blended,
            (center_in_area_x - 15, center_in_area_y),
            (center_in_area_x + 15, center_in_area_y),
            (0, 255, 0),
            2,
        )
        cv2.line(
            blended,
            (center_in_area_x, center_in_area_y - 15),
            (center_in_area_x, center_in_area_y + 15),
            (0, 255, 0),
            2,
        )

        # Draw search radius circle
        cv2.circle(
            blended,
            (center_in_area_x, center_in_area_y),
            search_radius,
            (255, 255, 0),
            2,
        )

        cv2.imwrite(filename, blended)
        print(f"  Debug: Saved similarity heatmap to {filename}")
    except Exception as e:
        print(f"  Debug: Failed to save similarity heatmap to {filename}: {e}")


def extract_dense_features(patch_features):
    """Extract dense features from DINOv2 patch tokens for spatial matching.

    Args:
        patch_features: Tensor from DINOv2 model output, shape [batch, seq_len, feature_dim]
                       where seq_len = 257 (1 CLS + 256 patches) and feature_dim = 384

    Returns:
        Tensor: Patch features reshaped to spatial map [batch, 16, 16, feature_dim]
    """
    batch_size, seq_len, feature_dim = patch_features.shape

    # Assert expected dimensions for ViT-S/14 with 224x224 input
    assert batch_size == 1, f"Expected batch size 1, got {batch_size}"
    assert (
        seq_len == 257
    ), f"Expected seq_len=257 (1 CLS + 256 patches), got {seq_len}"
    assert (
        feature_dim == 384
    ), f"Expected feature_dim=384 for ViT-S/14, got {feature_dim}"

    # Remove CLS token (first token) to get patch tokens
    patch_tokens = patch_features[:, 1:, :]  # [batch, 256, 384]

    # Reshape patch tokens to spatial grid (16x16 patches for 224x224 input with patch_size=14)
    patch_grid = patch_tokens.reshape(batch_size, 16, 16, feature_dim)

    return patch_grid


def compute_dense_similarity(src_features, tgt_features):
    """Compute dense similarity between source and target feature maps.

    Args:
        src_features: Source patch features [1, 16, 16, 384]
        tgt_features: Target patch features [1, 16, 16, 384]

    Returns:
        Tensor: Similarity map [1, 16, 16]
    """
    # Normalize features for cosine similarity
    src_norm = torch.nn.functional.normalize(src_features, p=2, dim=-1)
    tgt_norm = torch.nn.functional.normalize(tgt_features, p=2, dim=-1)

    # Compute cosine similarity at each spatial location
    similarity = torch.sum(src_norm * tgt_norm, dim=-1)  # [1, 16, 16]

    return similarity


def find_best_match_dense(
    src_patch_features,
    tgt_image,
    center_x,
    center_y,
    search_radius,
    patch_size,
    model,
    debug_images=False,
    uuid=None,
):
    """Find best match using dense feature comparison."""
    best_similarity = -float("inf")
    best_x, best_y = center_x, center_y

    # Extract dense features from source patch
    src_dense = extract_dense_features(src_patch_features)  # [1, 16, 16, 384]

    # Search in a coarser grid for efficiency (every 14 pixels = patch stride)
    step = 14

    # Store similarity scores for heatmap if debug mode is enabled
    similarity_scores = []
    candidate_positions = []

    for dy in range(-search_radius, search_radius + 1, step):
        for dx in range(-search_radius, search_radius + 1, step):
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
                if debug_images:
                    # Store invalid positions with low similarity for heatmap
                    similarity_scores.append(-1.0)
                    candidate_positions.append((candidate_x, candidate_y))
                continue

            # Extract patch from target at candidate location
            y_start = max(0, candidate_y - half_size)
            y_end = min(tgt_image.shape[0], candidate_y + half_size)
            x_start = max(0, candidate_x - half_size)
            x_end = min(tgt_image.shape[1], candidate_x + half_size)

            tgt_patch = tgt_image[y_start:y_end, x_start:x_end]

            # Ensure patch is correct size
            if (
                tgt_patch.shape[0] < patch_size
                or tgt_patch.shape[1] < patch_size
            ):
                padded_patch = np.zeros(
                    (patch_size, patch_size, 3), dtype=tgt_patch.dtype
                )
                padded_patch[: tgt_patch.shape[0], : tgt_patch.shape[1]] = (
                    tgt_patch
                )
                tgt_patch = padded_patch
            elif (
                tgt_patch.shape[0] > patch_size
                or tgt_patch.shape[1] > patch_size
            ):
                tgt_patch = tgt_patch[:patch_size, :patch_size]

            # Convert to tensor and extract dense features
            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            tgt_tensor = transform(tgt_patch).unsqueeze(0)

            # Extract patch features for target
            tgt_patch_features = model.extract_patch_features(tgt_tensor)
            tgt_dense = extract_dense_features(tgt_patch_features)

            # Compute dense similarity
            similarity_map = compute_dense_similarity(
                src_dense, tgt_dense
            )  # [1, 16, 16]

            # Use maximum similarity across all spatial locations
            max_similarity = torch.max(similarity_map).item()

            if debug_images:
                similarity_scores.append(max_similarity)
                candidate_positions.append((candidate_x, candidate_y))

            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_x, best_y = candidate_x, candidate_y

    # Generate heatmap if debug mode is enabled
    if debug_images and uuid:
        save_similarity_heatmap(
            tgt_image,
            center_x,
            center_y,
            search_radius,
            candidate_positions,
            similarity_scores,
            step,
            f"{uuid}_tgt_heatmap.jpg",
        )

    return best_x, best_y, best_similarity


def extract_patch_features(
    image, center_x, center_y, patch_size, model, transform
):
    """Extract DINOv2 CLS token from a patch centered at (center_x,
    center_y)."""
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
        # Extract patch features for dense matching
        patch_features = model.extract_patch_features(patch_tensor)

    # Assert expected DINOv2 patch features shape
    assert (
        len(patch_features.shape) == 3
    ), f"Expected 3D patch features tensor [batch, seq_len, feature_dim], got shape {patch_features.shape}"
    batch_size, seq_len, feature_dim = patch_features.shape
    assert batch_size == 1, f"Expected batch size 1, got {batch_size}"
    assert (
        seq_len == 257
    ), f"Expected seq_len=257 (1 CLS + 256 patches), got {seq_len}"
    assert (
        feature_dim == 384
    ), f"Expected feature_dim=384 for ViT-S/14, got {feature_dim}"

    # Return dense features for spatial matching
    return patch_features


def extract_patch_features_from_patch(patch, model, transform):
    """Extract DINOv2 CLS token from a pre-processed patch."""
    # Convert to tensor and normalize
    patch_tensor = transform(patch).unsqueeze(0)

    with torch.no_grad():
        # Extract patch features for dense matching
        patch_features = model.extract_patch_features(patch_tensor)

    # Assert expected DINOv2 patch features shape
    assert (
        len(patch_features.shape) == 3
    ), f"Expected 3D patch features tensor [batch, seq_len, feature_dim], got shape {patch_features.shape}"
    batch_size, seq_len, feature_dim = patch_features.shape
    assert batch_size == 1, f"Expected batch size 1, got {batch_size}"
    assert (
        seq_len == 257
    ), f"Expected seq_len=257 (1 CLS + 256 patches), got {seq_len}"
    assert (
        feature_dim == 384
    ), f"Expected feature_dim=384 for ViT-S/14, got {feature_dim}"

    # Return dense features for spatial matching
    return patch_features


def find_best_match_location(
    src_features,
    tgt_image,
    center_x,
    center_y,
    search_radius,
    patch_size,
    model,
    transform,
    debug_images=False,
    uuid=None,
):
    """Find the best matching location within search_radius of the initial
    guess."""
    best_similarity = -float("inf")
    best_x, best_y = center_x, center_y

    step = 14

    # Store similarity scores for heatmap if debug mode is enabled
    similarity_scores = []
    candidate_positions = []

    # Search in a grid around the initial location
    for dy in range(
        -search_radius, search_radius + 1, step
    ):  # Step by step for efficiency
        for dx in range(-search_radius, search_radius + 1, step):
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
                if debug_images:
                    # Store invalid positions with low similarity for heatmap
                    similarity_scores.append(-1.0)
                    candidate_positions.append((candidate_x, candidate_y))
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

            if debug_images:
                similarity_scores.append(similarity)
                candidate_positions.append((candidate_x, candidate_y))

            if similarity > best_similarity:
                best_similarity = similarity
                best_x, best_y = candidate_x, candidate_y

    # Generate heatmap if debug mode is enabled
    if debug_images and uuid:
        save_similarity_heatmap(
            tgt_image,
            center_x,
            center_y,
            search_radius,
            candidate_positions,
            similarity_scores,
            step,
            f"{uuid}_tgt_heatmap.jpg",
        )

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
    print("Using DINOv2 dense patch features for spatial similarity matching")

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

    patch_size = 224
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

        # Find best matching location using dense feature matching
        try:
            best_x, best_y, similarity = find_best_match_dense(
                src_features,
                tgt_image,
                tgt_mole["x"],
                tgt_mole["y"],
                search_radius,
                patch_size,
                model,
                debug_images,
                uuid,
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
