"""DINOv3 model loading and feature extraction utils for semantic matching."""

import cv2
import numpy as np

PATCH_SIZE = 16


def load_dinov3_model(dino_size="base", local_files_only=False, pretrained=True):
    """Load the DINOv3 model for semantic feature extraction.

    Args:
        dino_size: Model size variant ("small", "base", "large", "huge", "7b")
        local_files_only: If True, only use cached model weights (no network).

    Returns:
        tuple: (model, feature_dimension)
    """
    # Import os first to set environment variable before other imports
    import os

    # Set offline mode via environment variable BEFORE importing timm,
    # as timm import may trigger HF hub initialization
    old_hf_hub_offline = os.environ.get("HF_HUB_OFFLINE")
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Import lazily to avoid slow import times when not using this module.
    import timm
    import torch

    # Map size names to timm model names and their feature dimensions
    model_configs = {
        "small": ("vit_small_patch16_dinov3.lvd1689m", 384),
        "base": ("vit_base_patch16_dinov3.lvd1689m", 768),
        "large": ("vit_large_patch16_dinov3.lvd1689m", 1024),
        "huge": ("vit_huge_plus_patch16_dinov3.lvd1689m", 1280),
        "7b": ("vit_7b_patch16_dinov3.lvd1689m", 4096),
    }

    if dino_size not in model_configs:
        raise ValueError(
            f"Invalid dino_size: {dino_size}. "
            f"Must be one of {list(model_configs.keys())}"
        )

    model_name, feature_dim = model_configs[dino_size]

    try:
        # DINOv3 uses RoPE so it supports variable input sizes
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        model.eval()

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        return Dinov3Model(model, feature_dim, device), feature_dim

    except Exception as e:
        raise RuntimeError("Failed to load DINOv3 model. Error: " + str(e)) from e

    finally:
        # Restore original HF_HUB_OFFLINE setting
        if local_files_only:
            if old_hf_hub_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_hf_hub_offline


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
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.MEAN, std=self.STD),
            ]
        )

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


def extract_all_patch_features(scaled_image, model):
    """Extract features for all patches in the scaled image.

    Args:
        scaled_image: Scaled image in RGB format
        model: Dinov3Model instance

    Returns:
        Tensor: All patch features [num_patches, feature_dim]
    """
    return model.extract_patch_features(scaled_image)


def apply_mask(image_rgb, mask):
    """Apply mask to image, setting masked-out pixels to ImageNet mean.

    Args:
        image_rgb: Image in RGB format (H, W, 3), uint8
        mask: Grayscale mask (H, W), 255=include, 0=exclude

    Returns:
        Masked image with excluded regions set to ImageNet mean color
    """
    # ImageNet mean in RGB, 0-255 scale
    mean_rgb = (int(0.485 * 255), int(0.456 * 255), int(0.406 * 255))

    # Create output image
    result = image_rgb.copy()

    # Create boolean mask (True = keep)
    keep_mask = mask > 127

    # Set masked-out pixels to ImageNet mean
    for c, mean_val in enumerate(mean_rgb):
        result[:, :, c] = np.where(keep_mask, result[:, :, c], mean_val)

    return result


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
