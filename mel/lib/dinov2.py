"""DINOv2 model loading and feature extraction utils for semantic matching."""

import cv2
import numpy as np


def load_dinov2_model(dino_size="base"):
    """Load the DINOv2 model for semantic feature extraction with context.

    Args:
        dino_size: Model size variant ("small", "base", "large", "giant")

    Returns:
        tuple: (model_wrapper, feature_dimension)
    """
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torch

    # Map size names to actual model names and their feature dimensions
    match dino_size:
        case "small":
            model_name, feature_dim = "dinov2_vits14", 384
        case "base":
            model_name, feature_dim = "dinov2_vitb14", 768
        case "large":
            model_name, feature_dim = "dinov2_vitl14", 1024
        case "giant":
            model_name, feature_dim = "dinov2_vitg14", 1536
        case _:
            raise ValueError(
                f"Invalid dino_size: {dino_size}. Must be one of ['small', 'base', 'large', 'giant']"
            )

    try:
        # Load DINOv2 model for semantic patch features with rich context
        model = torch.hub.load("facebookresearch/dinov2", model_name)

        # Create a wrapper to extract patch tokens for context-aware matching
        class ContextualFeatureExtractor:
            def __init__(self, model):
                self.model = model
                self.model.eval()

            def extract_contextual_patch_features(self, x, center_patch_idx=None):
                """Extract patch features with full context for semantic
                matching.

                Args:
                    x: Input tensor [batch, channels, height, width]
                    center_patch_idx: Index of center patch to extract (if None, extract all)

                Returns:
                    If center_patch_idx is None: All patch features [batch, num_patches, feature_dim]
                    If center_patch_idx is provided: Center patch features [batch, feature_dim]
                """
                # Use forward hook to capture patch tokens with full context
                patch_features = []

                def hook_fn(module, _input_tensor, output):
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
                        # Return all patch features
                        return patch_tokens
                    # Fallback: use CLS token
                    cls_features = self.model(x)
                    return (
                        cls_features.unsqueeze(1)
                        if center_patch_idx is None
                        else cls_features
                    )
                finally:
                    hook.remove()

        return ContextualFeatureExtractor(model), feature_dim
    except Exception as e:
        raise RuntimeError(
            "Failed to load DINOv2 model. Please ensure you have internet access "
            "and the required dependencies. Error: " + str(e)
        ) from e


def extract_contextual_patch_feature(
    image, center_x, center_y, context_size, model, transform, feature_dim
):
    """Extract contextual patch feature from a large context window.

    Args:
        image: Input image array
        center_x, center_y: Center coordinates of the mole
        context_size: Size of context window (e.g., 910 for 910x910)
        model: DINOv2 model wrapper
        transform: Image transform pipeline
        feature_dim: Feature dimension of the model

    Returns:
        Tensor: Context-aware feature for center patch [feature_dim]
    """
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torch

    # Extract large context patch centered on the mole
    half_context = context_size // 2
    y_start = max(0, center_y - half_context)
    y_end = min(image.shape[0], center_y + half_context)
    x_start = max(0, center_x - half_context)
    x_end = min(image.shape[1], center_x + half_context)

    context_patch = image[y_start:y_end, x_start:x_end]

    # Pad if necessary to ensure context_size x context_size
    if context_patch.shape[0] < context_size or context_patch.shape[1] < context_size:
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
    elif context_patch.shape[0] > context_size or context_patch.shape[1] > context_size:
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
    center_features = center_features.squeeze(0)  # [feature_dim]
    assert center_features.shape == (feature_dim,), (
        f"Expected shape ({feature_dim},), got {center_features.shape}"
    )

    return center_features


def extract_all_contextual_features(
    image, center_x, center_y, context_size, model, transform, feature_dim
):
    """Extract features for all patches in a context window.

    Args:
        image: Input image array
        center_x, center_y: Center coordinates
        context_size: Size of context window (e.g., 910 for 910x910)
        model: DINOv2 model wrapper
        transform: Image transform pipeline
        feature_dim: Feature dimension of the model

    Returns:
        Tensor: All patch features [num_patches, feature_dim] where num_patches = (context_size//14)^2
    """
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torch

    # Extract large context patch centered on the location
    half_context = context_size // 2
    y_start = max(0, center_y - half_context)
    y_end = min(image.shape[0], center_y + half_context)
    x_start = max(0, center_x - half_context)
    x_end = min(image.shape[1], center_x + half_context)

    context_patch = image[y_start:y_end, x_start:x_end]

    # Pad if necessary to ensure context_size x context_size
    if context_patch.shape[0] < context_size or context_patch.shape[1] < context_size:
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
    elif context_patch.shape[0] > context_size or context_patch.shape[1] > context_size:
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
        feature_dim,
    ), (
        f"Expected shape ({expected_patches}, {feature_dim}), got {all_patch_features.shape}"
    )

    return all_patch_features


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
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torch

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
        if context_area.shape[0] < context_size or context_area.shape[1] < context_size:
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
            similarities.cpu().numpy().reshape(patches_per_side, patches_per_side)
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
        heatmap_colored[:, :, 2] = (heatmap * 255).astype(np.uint8)  # Red channel

        # Blend with original image (70% original, 30% heatmap)
        blended = cv2.addWeighted(context_area, 0.7, heatmap_colored, 0.3, 0)

        # Find and mark the best match location
        best_patch_idx = torch.argmax(similarities).item()
        best_patch_row = best_patch_idx // patches_per_side
        best_patch_col = best_patch_idx % patches_per_side

        # Convert to pixel coordinates within context
        best_y_pixel = best_patch_row * patch_size_pixels + patch_size_pixels // 2
        best_x_pixel = best_patch_col * patch_size_pixels + patch_size_pixels // 2

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


def find_best_contextual_match(
    src_center_features,
    tgt_image,
    center_x,
    center_y,
    context_size,
    model,
    transform,
    feature_dim,
    debug_images=False,
    uuid=None,
):
    """Find best match using contextual semantic features.

    Args:
        src_center_features: Contextual features of source mole center patch [feature_dim]
        tgt_image: Target image
        center_x, center_y: Initial target location
        context_size: Size of context window for feature extraction
        model: DINOv2 model wrapper
        transform: Image transform pipeline
        feature_dim: Feature dimension of the model
        debug_images: Whether to save debug images
        uuid: Mole UUID for debug filenames

    Returns:
        tuple: (best_x, best_y, best_similarity)
    """
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torch

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
            tgt_image, center_x, center_y, context_size, model, transform, feature_dim
        )

        # Normalize source and target features for cosine similarity
        src_norm = torch.nn.functional.normalize(
            src_center_features, p=2, dim=0
        )  # [feature_dim]
        tgt_norm = torch.nn.functional.normalize(
            tgt_all_features, p=2, dim=1
        )  # [num_patches, feature_dim]

        # Compute cosine similarity between source center and all target patches
        # Calculate Euclidean distance between source and target features
        # Note: We negate the distance to maintain the convention that higher values = better matches
        similarities = -torch.cdist(tgt_norm, src_norm.unsqueeze(0)).squeeze(
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


def extract_patch_features(image, center_x, center_y, patch_size, model, transform):
    """Extract DINOv2 CLS token from a patch centered at (center_x,
    center_y)."""
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torch

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
    assert len(patch_features.shape) == 3, (
        f"Expected 3D patch features tensor [batch, seq_len, feature_dim], got shape {patch_features.shape}"
    )
    batch_size, seq_len, feature_dim = patch_features.shape
    assert batch_size == 1, f"Expected batch size 1, got {batch_size}"
    assert seq_len == 257, f"Expected seq_len=257 (1 CLS + 256 patches), got {seq_len}"
    assert feature_dim == 384, (
        f"Expected feature_dim=384 for ViT-S/14, got {feature_dim}"
    )

    # Return dense features for spatial matching
    return patch_features


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
