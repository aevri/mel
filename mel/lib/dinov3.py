"""DINOv3 model loading and feature extraction utils for semantic matching."""

import cv2
import numpy as np

PATCH_SIZE = 16


def load_dinov3_model(dino_size="base"):
    """Load the DINOv3 model for semantic feature extraction.

    Args:
        dino_size: Model size variant ("small", "base", "large")

    Returns:
        tuple: (model, feature_dimension)
    """
    # Import lazily to avoid slow import times when not using this module.
    import timm
    import torch

    # Map size names to timm model names and their feature dimensions
    model_configs = {
        "small": ("vit_small_patch16_dinov3.lvd1689m", 384),
        "base": ("vit_base_patch16_dinov3.lvd1689m", 768),
        "large": ("vit_large_patch16_dinov3.lvd1689m", 1024),
    }

    if dino_size not in model_configs:
        raise ValueError(
            f"Invalid dino_size: {dino_size}. "
            f"Must be one of {list(model_configs.keys())}"
        )

    model_name, feature_dim = model_configs[dino_size]

    try:
        # DINOv3 uses RoPE so it supports variable input sizes
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        model.eval()

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        return Dinov3Model(model, feature_dim, device), feature_dim

    except Exception as e:
        raise RuntimeError(
            "Failed to load DINOv3 model. Error: " + str(e)
        ) from e


class Dinov3Model:
    """Wrapper for DINOv3 model to extract patch features."""

    # ImageNet normalization constants
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, model, feature_dim, device):
        self.model = model
        self.feature_dim = feature_dim
        self.device = device

    def extract_patch_features(self, image_rgb):
        """Extract patch features from a numpy image.

        Args:
            image_rgb: Input numpy array (H, W, 3) in RGB format.
                       Must be pre-scaled to size divisible by 16.

        Returns:
            Tensor: All patch features [num_patches, feature_dim]
        """
        import torch
        from torchvision import transforms

        # Use simple normalize transform - no resizing
        # DINOv3 uses RoPE so it supports variable input sizes
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

        # Apply transform and add batch dimension
        image_tensor = transform(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get intermediate features - this returns patch tokens
            features = self.model.forward_features(image_tensor)

            # features shape: [batch, num_tokens, feature_dim]
            # For DINOv3 ViT: [CLS] + [4 register tokens] + [patch tokens]
            # Skip first 5 tokens to get patch features
            if len(features.shape) == 3:
                patch_tokens = features[:, 5:, :]
                return patch_tokens[0]  # [num_patches, feature_dim]

            # Fallback: if features is 2D, it's already pooled
            raise RuntimeError(
                f"Unexpected features shape: {features.shape}. "
                "Expected 3D tensor [batch, tokens, dim]"
            )


def scale_image_to_fit(image_rgb, image_size):
    """Scale image to fit within image_size, preserving aspect ratio.

    Args:
        image_rgb: Input numpy array (H, W, 3)
        image_size: Target size (image will fit within this square)

    Returns:
        tuple: (scaled_image, scale_factor)
    """
    h, w = image_rgb.shape[:2]
    scale = image_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    # Ensure dimensions are divisible by patch size
    new_h = (new_h // PATCH_SIZE) * PATCH_SIZE
    new_w = (new_w // PATCH_SIZE) * PATCH_SIZE

    if new_h == 0:
        new_h = PATCH_SIZE
    if new_w == 0:
        new_w = PATCH_SIZE

    scaled = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Recalculate actual scale factor based on final dimensions
    scale_x = new_w / w
    scale_y = new_h / h

    return scaled, (scale_x, scale_y)


def extract_mole_patch_feature(scaled_image, mole_x, mole_y, model, multi_patch=False):
    """Extract feature for the patch containing the mole.

    Args:
        scaled_image: Scaled image in RGB format
        mole_x, mole_y: Mole coordinates (already scaled)
        model: Dinov3Model instance
        multi_patch: If True, average 3x3 patch region around mole

    Returns:
        Tensor: Feature vector [feature_dim]
    """
    # Get all patch features
    all_features = model.extract_patch_features(scaled_image)

    # Calculate which patch the mole is in
    h, w = scaled_image.shape[:2]
    patches_per_row = w // PATCH_SIZE
    patches_per_col = h // PATCH_SIZE
    patch_col = mole_x // PATCH_SIZE
    patch_row = mole_y // PATCH_SIZE

    if multi_patch:
        # Average 3x3 patch region around mole center
        features_to_avg = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r = patch_row + dr
                c = patch_col + dc
                # Check bounds
                if 0 <= r < patches_per_col and 0 <= c < patches_per_row:
                    idx = r * patches_per_row + c
                    if idx < all_features.shape[0]:
                        features_to_avg.append(all_features[idx])

        if features_to_avg:
            import torch

            stacked = torch.stack(features_to_avg)
            return stacked.mean(dim=0)

    # Single patch (default)
    patch_idx = patch_row * patches_per_row + patch_col

    # Clamp to valid range
    num_patches = all_features.shape[0]
    patch_idx = min(patch_idx, num_patches - 1)

    return all_features[patch_idx]


def extract_all_patch_features(scaled_image, model):
    """Extract features for all patches in the scaled image.

    Args:
        scaled_image: Scaled image in RGB format
        model: Dinov3Model instance

    Returns:
        Tensor: All patch features [num_patches, feature_dim]
    """
    return model.extract_patch_features(scaled_image)


def compute_similarities(mole_feature, all_patch_features, similarity_type="cosine"):
    """Compute similarity between mole feature and all patches.

    Args:
        mole_feature: Mole feature vector [feature_dim]
        all_patch_features: All patch features [num_patches, feature_dim]
        similarity_type: One of "cosine", "euclidean", "dot"

    Returns:
        Tensor: Similarities [num_patches] (higher = better match)
    """
    import torch

    if similarity_type == "cosine":
        # Normalize for cosine similarity
        mole_norm = torch.nn.functional.normalize(
            mole_feature.unsqueeze(0), p=2, dim=1
        )
        patches_norm = torch.nn.functional.normalize(all_patch_features, p=2, dim=1)
        # Cosine similarity: dot product of normalized vectors
        return torch.matmul(patches_norm, mole_norm.squeeze(0))

    if similarity_type == "euclidean":
        # Negative L2 distance (higher = closer = better match)
        diff = all_patch_features - mole_feature.unsqueeze(0)
        distances = torch.norm(diff, p=2, dim=1)
        return -distances

    if similarity_type == "dot":
        # Raw dot product without normalization
        return torch.matmul(all_patch_features, mole_feature)

    if similarity_type == "softmax":
        # Cosine similarity with temperature-scaled softmax
        mole_norm = torch.nn.functional.normalize(
            mole_feature.unsqueeze(0), p=2, dim=1
        )
        patches_norm = torch.nn.functional.normalize(all_patch_features, p=2, dim=1)
        cosine_sims = torch.matmul(patches_norm, mole_norm.squeeze(0))
        temperature = 0.0125  # Lower = sharper peaks
        return torch.softmax(cosine_sims / temperature, dim=0)

    raise ValueError(f"Unknown similarity_type: {similarity_type}")


def render_heatmap(
    image_rgb, similarities, image_height, image_width, similarity_type="cosine"
):
    """Render similarity heatmap overlay on the image.

    Args:
        image_rgb: Image in RGB format
        similarities: Tensor of similarities [num_patches]
        image_height: Height of image (for calculating patches per row)
        image_width: Width of image (for calculating patches per row)
        similarity_type: Type of similarity metric used (affects crosshair placement)

    Returns:
        numpy array: Rendered heatmap image (BGR format for OpenCV)
    """
    import torch

    patches_per_row = image_width // PATCH_SIZE
    patches_per_col = image_height // PATCH_SIZE
    num_patches = patches_per_row * patches_per_col

    # Convert to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Create heatmap overlay
    heatmap = np.zeros(image_bgr.shape[:2], dtype=np.float32)

    # Reshape similarities to 2D grid
    sim_values = similarities.cpu().numpy()

    # Handle case where model returns different number of patches
    if len(sim_values) != num_patches:
        # Truncate or pad as needed
        if len(sim_values) > num_patches:
            sim_values = sim_values[:num_patches]
        else:
            padded = np.zeros(num_patches)
            padded[: len(sim_values)] = sim_values
            sim_values = padded

    similarity_grid = sim_values.reshape(patches_per_col, patches_per_row)

    # Normalize similarities to 0-1 range
    sim_min = similarity_grid.min()
    sim_max = similarity_grid.max()
    sim_range = sim_max - sim_min if sim_max > sim_min else 1.0
    normalized_grid = (similarity_grid - sim_min) / sim_range

    # Map each patch similarity to pixel regions
    for patch_row in range(patches_per_col):
        for patch_col in range(patches_per_row):
            y_start = patch_row * PATCH_SIZE
            y_end = min(image_height, (patch_row + 1) * PATCH_SIZE)
            x_start = patch_col * PATCH_SIZE
            x_end = min(image_width, (patch_col + 1) * PATCH_SIZE)
            heatmap[y_start:y_end, x_start:x_end] = normalized_grid[
                patch_row, patch_col
            ]

    # Convert heatmap to red channel overlay
    heatmap_colored = np.zeros(image_bgr.shape, dtype=np.uint8)
    heatmap_colored[:, :, 2] = (heatmap * 255).astype(np.uint8)  # Red channel (BGR)

    # Blend with original image
    blended = cv2.addWeighted(image_bgr, 0.7, heatmap_colored, 0.3, 0)

    # Mark best match location (green cross)
    if similarity_type == "softmax":
        # Weighted centroid using softmax probabilities
        probs = sim_values  # Already a probability distribution
        best_x = 0.0
        best_y = 0.0
        for idx in range(num_patches):
            row = idx // patches_per_row
            col = idx % patches_per_row
            center_x = col * PATCH_SIZE + PATCH_SIZE // 2
            center_y = row * PATCH_SIZE + PATCH_SIZE // 2
            best_x += probs[idx] * center_x
            best_y += probs[idx] * center_y
        best_x = int(best_x)
        best_y = int(best_y)
    else:
        # Argmax for other similarity types
        best_patch_idx = torch.argmax(similarities).item()
        if best_patch_idx >= num_patches:
            best_patch_idx = num_patches - 1
        best_patch_row = best_patch_idx // patches_per_row
        best_patch_col = best_patch_idx % patches_per_row
        best_y = best_patch_row * PATCH_SIZE + PATCH_SIZE // 2
        best_x = best_patch_col * PATCH_SIZE + PATCH_SIZE // 2

    cv2.line(
        blended, (best_x - 15, best_y), (best_x + 15, best_y), (0, 255, 0), 3
    )
    cv2.line(
        blended, (best_x, best_y - 15), (best_x, best_y + 15), (0, 255, 0), 3
    )

    return blended


# -----------------------------------------------------------------------------
# Copyright (C) 2025-2026 Angelos Evripiotis.
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
