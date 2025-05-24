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
    """Load the DINOv2 model for semantic feature extraction with context."""
    try:
        # Load DINOv2 model for semantic patch features with rich context
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

        # Create a wrapper to extract patch tokens for context-aware matching
        class ContextualFeatureExtractor:
            def __init__(self, model):
                self.model = model
                self.model.eval()

            def extract_contextual_patch_features(
                self, x, center_patch_idx=None
            ):
                """Extract patch features with full context for semantic
                matching.

                Args:
                    x: Input tensor [batch, channels, height, width]
                    center_patch_idx: Index of center patch to extract (if None, extract all)

                Returns:
                    If center_patch_idx is None: All patch features [batch, num_patches, 768]
                    If center_patch_idx is provided: Center patch features [batch, 768]
                """
                # Use forward hook to capture patch tokens with full context
                patch_features = []

                def hook_fn(module, input, output):
                    if hasattr(output, "shape") and len(output.shape) == 3:
                        patch_features.append(output)

                # Register hook on the normalization layer to get contextualized features
                hook = self.model.norm.register_forward_hook(hook_fn)

                try:
                    # Run forward pass to get contextualized features
                    _ = self.model(x)
                    if patch_features:
                        # patch_features[0] shape: [batch, seq_len, feature_dim] where seq_len = 1 + num_patches
                        all_tokens = patch_features[0]
                        # Remove CLS token to get just patch tokens
                        patch_tokens = all_tokens[
                            :, 1:, :
                        ]  # [batch, num_patches, feature_dim]

                        if center_patch_idx is not None:
                            # Extract specific center patch
                            return patch_tokens[
                                :, center_patch_idx, :
                            ]  # [batch, feature_dim]
                        else:
                            # Return all patch features
                            return patch_tokens
                    else:
                        # Fallback: use CLS token
                        cls_features = self.model(x)
                        return (
                            cls_features.unsqueeze(1)
                            if center_patch_idx is None
                            else cls_features
                        )
                finally:
                    hook.remove()

        return ContextualFeatureExtractor(model)
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


def save_contextual_similarity_heatmap(
    image,
    center_x,
    center_y,
    context_size,
    similarities,
    patches_per_side,
    filename,
):
    """Save a heatmap showing cosine similarities for each patch in the context
    window.

    Args:
        image: Target image
        center_x, center_y: Center of context window
        context_size: Size of context window (e.g., 910)
        similarities: Tensor of similarities [num_patches] for each patch
        patches_per_side: Number of patches per side (e.g., 65)
        filename: Output filename
    """
    try:
        # Extract context area for visualization
        half_context = context_size // 2
        context_left = max(0, center_x - half_context)
        context_right = min(image.shape[1], center_x + half_context)
        context_top = max(0, center_y - half_context)
        context_bottom = min(image.shape[0], center_y + half_context)

        context_area = image[
            context_top:context_bottom, context_left:context_right
        ].copy()

        # Pad if necessary
        if (
            context_area.shape[0] < context_size
            or context_area.shape[1] < context_size
        ):
            padded_area = np.zeros(
                (context_size, context_size, 3), dtype=context_area.dtype
            )
            y_offset = (context_size - context_area.shape[0]) // 2
            x_offset = (context_size - context_area.shape[1]) // 2
            padded_area[
                y_offset : y_offset + context_area.shape[0],
                x_offset : x_offset + context_area.shape[1],
            ] = context_area
            context_area = padded_area

        # Create heatmap overlay
        heatmap = np.zeros(context_area.shape[:2], dtype=np.float32)

        # Reshape similarities to 2D grid
        similarity_grid = (
            similarities.cpu()
            .numpy()
            .reshape(patches_per_side, patches_per_side)
        )

        # Normalize similarities to 0-1 range for visualization
        sim_min = similarity_grid.min()
        sim_max = similarity_grid.max()
        sim_range = sim_max - sim_min if sim_max > sim_min else 1.0
        normalized_grid = (similarity_grid - sim_min) / sim_range

        # Map each patch similarity to pixel regions
        patch_size_pixels = 14
        for patch_row in range(patches_per_side):
            for patch_col in range(patches_per_side):
                # Calculate pixel coordinates for this patch
                y_start = patch_row * patch_size_pixels
                y_end = min(context_size, (patch_row + 1) * patch_size_pixels)
                x_start = patch_col * patch_size_pixels
                x_end = min(context_size, (patch_col + 1) * patch_size_pixels)

                # Set heatmap values for this patch region
                heatmap[y_start:y_end, x_start:x_end] = normalized_grid[
                    patch_row, patch_col
                ]

        # Convert heatmap to red channel overlay
        heatmap_colored = np.zeros(context_area.shape, dtype=np.uint8)
        heatmap_colored[:, :, 2] = (heatmap * 255).astype(
            np.uint8
        )  # Red channel

        # Blend with original image (70% original, 30% heatmap)
        blended = cv2.addWeighted(context_area, 0.7, heatmap_colored, 0.3, 0)

        # Find and mark the best match location
        best_patch_idx = torch.argmax(similarities).item()
        best_patch_row = best_patch_idx // patches_per_side
        best_patch_col = best_patch_idx % patches_per_side

        # Convert to pixel coordinates within context
        best_y_pixel = (
            best_patch_row * patch_size_pixels + patch_size_pixels // 2
        )
        best_x_pixel = (
            best_patch_col * patch_size_pixels + patch_size_pixels // 2
        )

        # Draw best match marker (bright green cross)
        cv2.line(
            blended,
            (best_x_pixel - 15, best_y_pixel),
            (best_x_pixel + 15, best_y_pixel),
            (0, 255, 0),
            3,
        )
        cv2.line(
            blended,
            (best_x_pixel, best_y_pixel - 15),
            (best_x_pixel, best_y_pixel + 15),
            (0, 255, 0),
            3,
        )

        # Draw center marker (yellow cross)
        center_pixel_x = context_size // 2
        center_pixel_y = context_size // 2
        cv2.line(
            blended,
            (center_pixel_x - 15, center_pixel_y),
            (center_pixel_x + 15, center_pixel_y),
            (0, 255, 255),
            2,
        )
        cv2.line(
            blended,
            (center_pixel_x, center_pixel_y - 15),
            (center_pixel_x, center_pixel_y + 15),
            (0, 255, 255),
            2,
        )

        cv2.imwrite(filename, blended)
        print(f"  Debug: Saved contextual similarity heatmap to {filename}")
        print(f"  Debug: Similarity range: {sim_min:.3f} to {sim_max:.3f}")

    except Exception as e:
        print(
            f"  Debug: Failed to save contextual similarity heatmap to {filename}: {e}"
        )


def extract_contextual_patch_feature(
    image, center_x, center_y, context_size, model, transform
):
    """Extract contextual patch feature from a large context window.

    Args:
        image: Input image array
        center_x, center_y: Center coordinates of the mole
        context_size: Size of context window (e.g., 910 for 910x910)
        model: DINOv2 model wrapper
        transform: Image transform pipeline

    Returns:
        Tensor: Context-aware feature for center patch [768]
    """
    # Extract large context patch centered on the mole
    half_context = context_size // 2
    y_start = max(0, center_y - half_context)
    y_end = min(image.shape[0], center_y + half_context)
    x_start = max(0, center_x - half_context)
    x_end = min(image.shape[1], center_x + half_context)

    context_patch = image[y_start:y_end, x_start:x_end]

    # Pad if necessary to ensure context_size x context_size
    if (
        context_patch.shape[0] < context_size
        or context_patch.shape[1] < context_size
    ):
        padded_patch = np.zeros(
            (context_size, context_size, 3), dtype=context_patch.dtype
        )
        y_offset = (context_size - context_patch.shape[0]) // 2
        x_offset = (context_size - context_patch.shape[1]) // 2
        padded_patch[
            y_offset : y_offset + context_patch.shape[0],
            x_offset : x_offset + context_patch.shape[1],
        ] = context_patch
        context_patch = padded_patch
    elif (
        context_patch.shape[0] > context_size
        or context_patch.shape[1] > context_size
    ):
        # Center crop if too large
        patch_center_y, patch_center_x = (
            context_patch.shape[0] // 2,
            context_patch.shape[1] // 2,
        )
        half_size = context_size // 2
        context_patch = context_patch[
            patch_center_y - half_size : patch_center_y + half_size,
            patch_center_x - half_size : patch_center_x + half_size,
        ]

    # Convert to tensor and normalize
    context_tensor = transform(context_patch).unsqueeze(0)

    # Calculate which patch corresponds to the center
    # For context_size=910 and patch_size=14: 910/14 = 65 patches per side
    patches_per_side = context_size // 14
    center_patch_row = patches_per_side // 2
    center_patch_col = patches_per_side // 2
    center_patch_idx = center_patch_row * patches_per_side + center_patch_col

    with torch.no_grad():
        # Extract contextual features for the center patch
        center_features = model.extract_contextual_patch_features(
            context_tensor, center_patch_idx=center_patch_idx
        )

    # Remove batch dimension and assert shape
    center_features = center_features.squeeze(0)  # [768]
    assert center_features.shape == (
        768,
    ), f"Expected shape (768,), got {center_features.shape}"

    return center_features


def extract_all_contextual_features(
    image, center_x, center_y, context_size, model, transform
):
    """Extract features for all patches in a context window.

    Args:
        image: Input image array
        center_x, center_y: Center coordinates
        context_size: Size of context window (e.g., 910 for 910x910)
        model: DINOv2 model wrapper
        transform: Image transform pipeline

    Returns:
        Tensor: All patch features [num_patches, 768] where num_patches = (context_size//14)^2
    """
    # Extract large context patch centered on the location
    half_context = context_size // 2
    y_start = max(0, center_y - half_context)
    y_end = min(image.shape[0], center_y + half_context)
    x_start = max(0, center_x - half_context)
    x_end = min(image.shape[1], center_x + half_context)

    context_patch = image[y_start:y_end, x_start:x_end]

    # Pad if necessary to ensure context_size x context_size
    if (
        context_patch.shape[0] < context_size
        or context_patch.shape[1] < context_size
    ):
        padded_patch = np.zeros(
            (context_size, context_size, 3), dtype=context_patch.dtype
        )
        y_offset = (context_size - context_patch.shape[0]) // 2
        x_offset = (context_size - context_patch.shape[1]) // 2
        padded_patch[
            y_offset : y_offset + context_patch.shape[0],
            x_offset : x_offset + context_patch.shape[1],
        ] = context_patch
        context_patch = padded_patch
    elif (
        context_patch.shape[0] > context_size
        or context_patch.shape[1] > context_size
    ):
        # Center crop if too large
        patch_center_y, patch_center_x = (
            context_patch.shape[0] // 2,
            context_patch.shape[1] // 2,
        )
        half_size = context_size // 2
        context_patch = context_patch[
            patch_center_y - half_size : patch_center_y + half_size,
            patch_center_x - half_size : patch_center_x + half_size,
        ]

    # Convert to tensor and normalize
    context_tensor = transform(context_patch).unsqueeze(0)

    with torch.no_grad():
        # Extract features for ALL patches (not just center)
        all_patch_features = model.extract_contextual_patch_features(
            context_tensor, center_patch_idx=None
        )

    # Remove batch dimension: [1, num_patches, 384] -> [num_patches, 384]
    all_patch_features = all_patch_features.squeeze(0)

    # Assert expected shape for 65x65 patches
    patches_per_side = context_size // 14
    expected_patches = patches_per_side * patches_per_side
    assert all_patch_features.shape == (
        expected_patches,
        768,
    ), f"Expected shape ({expected_patches}, 768), got {all_patch_features.shape}"

    return all_patch_features


def find_best_contextual_match(
    src_center_features,
    tgt_image,
    center_x,
    center_y,
    search_radius,
    context_size,
    model,
    transform,
    debug_images=False,
    uuid=None,
):
    """Find best match using contextual semantic features.

    Args:
        src_center_features: Contextual features of source mole center patch [768]
        tgt_image: Target image
        center_x, center_y: Initial target location
        search_radius: Search radius in pixels
        context_size: Size of context window for feature extraction
        model: DINOv2 model wrapper
        transform: Image transform pipeline
        debug_images: Whether to save debug images
        uuid: Mole UUID for debug filenames

    Returns:
        tuple: (best_x, best_y, best_similarity)
    """
    # Check if we can extract a full context window at the initial location
    half_context = context_size // 2
    if (
        center_x - half_context < 0
        or center_x + half_context >= tgt_image.shape[1]
        or center_y - half_context < 0
        or center_y + half_context >= tgt_image.shape[0]
    ):
        print(
            f"  Warning: Cannot extract full context window at ({center_x}, {center_y})"
        )
        return center_x, center_y, -1.0

    try:
        # Extract features for all patches in the target context window ONCE
        tgt_all_features = extract_all_contextual_features(
            tgt_image, center_x, center_y, context_size, model, transform
        )

        # Normalize source and target features for cosine similarity
        src_norm = torch.nn.functional.normalize(
            src_center_features, p=2, dim=0
        )  # [768]
        tgt_norm = torch.nn.functional.normalize(
            tgt_all_features, p=2, dim=1
        )  # [num_patches, 768]

        # Compute cosine similarity between source center and all target patches
        similarities = torch.mm(tgt_norm, src_norm.unsqueeze(1)).squeeze(
            1
        )  # [num_patches]

        # Find the patch with highest similarity
        best_patch_idx = torch.argmax(similarities).item()
        best_similarity = similarities[best_patch_idx].item()

        # Convert patch index to spatial coordinates
        patches_per_side = context_size // 14  # 65 for 910x910 context
        patch_row = best_patch_idx // patches_per_side
        patch_col = best_patch_idx % patches_per_side

        # Convert patch coordinates to pixel coordinates (relative to context center)
        # Each patch is 14x14 pixels, so offset from center of context
        offset_x = (patch_col - patches_per_side // 2) * 14
        offset_y = (patch_row - patches_per_side // 2) * 14

        best_x = center_x + offset_x
        best_y = center_y + offset_y

        print(
            f"  Found best match at patch ({patch_row}, {patch_col}) -> pixel ({best_x}, {best_y})"
        )
        print(f"  Similarity: {best_similarity:.3f}")

        # Generate spatial heatmap if debug mode is enabled
        if debug_images and uuid:
            save_contextual_similarity_heatmap(
                tgt_image,
                center_x,
                center_y,
                context_size,
                similarities,
                patches_per_side,
                f"{uuid}_tgt_heatmap.jpg",
            )

        return best_x, best_y, best_similarity

    except Exception as e:
        print(f"  Error in contextual matching: {e}")
        return center_x, center_y, -1.0


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

            agg_similarity = torch.mean(similarity_map).item()

            if debug_images:
                similarity_scores.append(agg_similarity)
                candidate_positions.append((candidate_x, candidate_y))

            if agg_similarity > best_similarity:
                best_similarity = agg_similarity
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
    print("Using DINOv2 contextual semantic features for mole matching")

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

    context_size = 910  # Large context window for rich semantic features
    refined_count = 0

    # Process each non-canonical mole in target
    for tgt_mole in tgt_non_canonical_to_refine:
        uuid = tgt_mole["uuid"]
        src_mole = src_mole_lookup[uuid]

        print(f"Refining mole {uuid}...")

        # Extract contextual semantic features from source mole location
        try:
            src_features = extract_contextual_patch_feature(
                src_image,
                src_mole["x"],
                src_mole["y"],
                context_size,
                model,
                transform,
            )
            print(
                f"  Extracted contextual features from {context_size}x{context_size} source context"
            )

            # Save debug image for source context patch
            if debug_images:
                # Extract context patch for debug visualization
                half_context = context_size // 2
                y_start = max(0, src_mole["y"] - half_context)
                y_end = min(src_image.shape[0], src_mole["y"] + half_context)
                x_start = max(0, src_mole["x"] - half_context)
                x_end = min(src_image.shape[1], src_mole["x"] + half_context)
                src_context_img = src_image[y_start:y_end, x_start:x_end]

                # Pad if necessary
                if (
                    src_context_img.shape[0] < context_size
                    or src_context_img.shape[1] < context_size
                ):
                    padded_patch = np.zeros(
                        (context_size, context_size, 3),
                        dtype=src_context_img.dtype,
                    )
                    y_offset = (context_size - src_context_img.shape[0]) // 2
                    x_offset = (context_size - src_context_img.shape[1]) // 2
                    padded_patch[
                        y_offset : y_offset + src_context_img.shape[0],
                        x_offset : x_offset + src_context_img.shape[1],
                    ] = src_context_img
                    src_context_img = padded_patch

                save_debug_patch(src_context_img, f"{uuid}_src_context.jpg")

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
                context_size,
                f"{uuid}_tgt_search_area.jpg",
            )

        # Find best matching location using contextual semantic matching
        try:
            best_x, best_y, similarity = find_best_contextual_match(
                src_features,
                tgt_image,
                tgt_mole["x"],
                tgt_mole["y"],
                search_radius,
                context_size,
                model,
                transform,
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

                # Save debug image for final refined context
                if debug_images:
                    try:
                        half_context = context_size // 2
                        y_start = max(0, best_y - half_context)
                        y_end = min(tgt_image.shape[0], best_y + half_context)
                        x_start = max(0, best_x - half_context)
                        x_end = min(tgt_image.shape[1], best_x + half_context)
                        refined_context = tgt_image[
                            y_start:y_end, x_start:x_end
                        ]

                        # Pad if necessary
                        if (
                            refined_context.shape[0] < context_size
                            or refined_context.shape[1] < context_size
                        ):
                            padded_patch = np.zeros(
                                (context_size, context_size, 3),
                                dtype=refined_context.dtype,
                            )
                            y_offset = (
                                context_size - refined_context.shape[0]
                            ) // 2
                            x_offset = (
                                context_size - refined_context.shape[1]
                            ) // 2
                            padded_patch[
                                y_offset : y_offset + refined_context.shape[0],
                                x_offset : x_offset + refined_context.shape[1],
                            ] = refined_context
                            refined_context = padded_patch

                        save_debug_patch(
                            refined_context, f"{uuid}_tgt_refined.jpg"
                        )
                    except Exception as e:
                        print(f"  Debug: Failed to save refined context: {e}")
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
