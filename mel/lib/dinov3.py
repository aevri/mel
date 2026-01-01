"""DINOv3 model loading and feature extraction utils for semantic matching."""

import cv2
import numpy as np


def load_dinov3_model(dino_size="base"):
    """Load the DINOv3 model for semantic feature extraction with context.

    Args:
        dino_size: Model size variant ("small", "base", "large", "huge", "giant")

    Returns:
        tuple: (model_wrapper, feature_dimension)
    """
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torch
    from transformers import AutoImageProcessor, Dinov3Model

    # Map size names to actual model names and their feature dimensions
    # DINOv3 models from HuggingFace
    model_configs = {
        "small": ("facebook/dinov3-vits16-pretrain-lvd1689m", 384),
        "base": ("facebook/dinov3-vitb16-pretrain-lvd1689m", 768),
        "large": ("facebook/dinov3-vitl16-pretrain-lvd1689m", 1024),
        "huge": ("facebook/dinov3-vith16-pretrain-lvd1689m", 1280),
        "giant": ("facebook/dinov3-vit7b16-pretrain-lvd1689m", 4096),
    }

    if dino_size not in model_configs:
        raise ValueError(
            f"Invalid dino_size: {dino_size}. Must be one of {list(model_configs.keys())}"
        )

    model_name, feature_dim = model_configs[dino_size]

    try:
        # Load DINOv3 model and processor
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = Dinov3Model.from_pretrained(model_name)

        # Create a wrapper to extract patch tokens for context-aware matching
        class ContextualFeatureExtractor:
            def __init__(self, model, processor):
                self.model = model
                self.processor = processor
                self.model.eval()

            def extract_contextual_patch_features(
                self, image_array, center_patch_idx=None
            ):
                """Extract patch features with full context for semantic
                matching.

                Args:
                    image_array: Input numpy array [height, width, channels] in RGB
                    center_patch_idx: Index of center patch to extract (if None, extract all)

                Returns:
                    If center_patch_idx is None: All patch features [num_patches, feature_dim]
                    If center_patch_idx is provided: Center patch features [feature_dim]
                """
                # Process the image array using the processor
                inputs = self.processor(images=image_array, return_tensors="pt")

                with torch.no_grad():
                    # Get model outputs
                    outputs = self.model(**inputs)
                    # last_hidden_state shape: [batch, seq_len, feature_dim]
                    # seq_len = 1 (CLS) + 4 (registers) + num_patches
                    all_tokens = outputs.last_hidden_state

                    # Remove CLS token (index 0) and register tokens (indices 1-4)
                    # to get just patch tokens
                    patch_tokens = all_tokens[
                        :, 5:, :
                    ]  # [batch, num_patches, feature_dim]

                    if center_patch_idx is not None:
                        # Extract specific center patch
                        return patch_tokens[0, center_patch_idx, :]  # [feature_dim]
                    # Return all patch features
                    return patch_tokens[0]  # [num_patches, feature_dim]

        return ContextualFeatureExtractor(model, processor), feature_dim
    except Exception as e:
        raise RuntimeError(
            "Failed to load DINOv3 model. Please ensure you have internet access "
            "and the required dependencies (transformers>=4.56.0). Error: " + str(e)
        ) from e


def extract_contextual_patch_feature(
    image, center_x, center_y, context_size, model, feature_dim
):
    """Extract contextual patch feature from a large context window.

    Args:
        image: Input image array (RGB)
        center_x, center_y: Center coordinates of the mole
        context_size: Size of context window (e.g., 1024 for 1024x1024)
        model: DINOv3 model wrapper
        feature_dim: Feature dimension of the model

    Returns:
        Tensor: Context-aware feature for center patch [feature_dim]
    """
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.

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

    # Calculate which patch corresponds to the center
    # For context_size=1024 and patch_size=16: 1024/16 = 64 patches per side
    patches_per_side = context_size // 16
    center_patch_row = patches_per_side // 2
    center_patch_col = patches_per_side // 2
    center_patch_idx = center_patch_row * patches_per_side + center_patch_col

    # Extract contextual features for the center patch
    center_features = model.extract_contextual_patch_features(
        context_patch, center_patch_idx=center_patch_idx
    )

    # Assert shape
    assert center_features.shape == (feature_dim,), (
        f"Expected shape ({feature_dim},), got {center_features.shape}"
    )

    return center_features


def extract_all_contextual_features(
    image, center_x, center_y, context_size, model, feature_dim
):
    """Extract features for all patches in a context window.

    Args:
        image: Input image array (RGB)
        center_x, center_y: Center coordinates
        context_size: Size of context window (e.g., 1024 for 1024x1024)
        model: DINOv3 model wrapper
        feature_dim: Feature dimension of the model

    Returns:
        Tensor: All patch features [num_patches, feature_dim] where num_patches = (context_size//16)^2
    """
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

    # Extract features for ALL patches (not just center)
    all_patch_features = model.extract_contextual_patch_features(
        context_patch, center_patch_idx=None
    )

    # Assert expected shape for 64x64 patches
    patches_per_side = context_size // 16
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
        image: Target image (RGB)
        center_x, center_y: Center of context window
        context_size: Size of context window (e.g., 1024)
        similarities: Tensor of similarities [num_patches] for each patch
        patches_per_side: Number of patches per side (e.g., 64)
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
        patch_size_pixels = 16
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

        # Convert heatmap to red channel overlay (note: image is RGB, cv2 expects BGR)
        heatmap_colored = np.zeros(context_area.shape, dtype=np.uint8)
        heatmap_colored[:, :, 0] = (heatmap * 255).astype(
            np.uint8
        )  # Red channel in RGB

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
            (255, 255, 0),
            2,
        )
        cv2.line(
            blended,
            (center_pixel_x, center_pixel_y - 15),
            (center_pixel_x, center_pixel_y + 15),
            (255, 255, 0),
            2,
        )

        # Convert RGB to BGR for cv2.imwrite
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, blended_bgr)
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
    feature_dim,
    debug_images=False,
    uuid=None,
):
    """Find best match using contextual semantic features.

    Args:
        src_center_features: Contextual features of source mole center patch [feature_dim]
        tgt_image: Target image (RGB)
        center_x, center_y: Initial target location
        context_size: Size of context window for feature extraction
        model: DINOv3 model wrapper
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
            tgt_image, center_x, center_y, context_size, model, feature_dim
        )

        # Normalize source and target features for cosine similarity
        src_norm = torch.nn.functional.normalize(
            src_center_features, p=2, dim=0
        )  # [feature_dim]
        tgt_norm = torch.nn.functional.normalize(
            tgt_all_features, p=2, dim=1
        )  # [num_patches, feature_dim]

        # Compute cosine similarity between source center and all target patches
        # Using negative Euclidean distance to maintain convention that higher = better
        similarities = -torch.cdist(tgt_norm, src_norm.unsqueeze(0)).squeeze(
            1
        )  # [num_patches]

        # Find the patch with highest similarity
        best_patch_idx = torch.argmax(similarities).item()
        best_similarity = similarities[best_patch_idx].item()

        # Convert patch index to spatial coordinates
        patches_per_side = context_size // 16  # 64 for 1024x1024 context
        patch_row = best_patch_idx // patches_per_side
        patch_col = best_patch_idx % patches_per_side

        # Convert patch coordinates to pixel coordinates (relative to context center)
        # Each patch is 16x16 pixels, so offset from center of context
        offset_x = (patch_col - patches_per_side // 2) * 16
        offset_y = (patch_row - patches_per_side // 2) * 16

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
