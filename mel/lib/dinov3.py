"""DINOv3 model loading and feature extraction utils for semantic matching."""

import numpy as np


def load_dinov3_model(dino_size="base"):
    """Load the DINOv3 model for semantic feature extraction.

    Args:
        dino_size: Model size variant ("small", "base", "large", "giant")

    Returns:
        tuple: (model, feature_dimension)
    """
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torch

    # Map size names to actual model names and their feature dimensions
    # DINOv3 uses ViT-16 patch size (not 14 like DINOv2)
    model_configs = {
        "small": ("dinov3_vits16", 384),
        "base": ("dinov3_vitb16", 768),
        "large": ("dinov3_vitl16", 1024),
        "giant": ("dinov3_vit7b16", 1536),
    }

    if dino_size not in model_configs:
        raise ValueError(
            f"Invalid dino_size: {dino_size}. Must be one of {list(model_configs.keys())}"
        )

    model_name, feature_dim = model_configs[dino_size]

    try:
        # Load DINOv3 model for semantic feature extraction
        model = torch.hub.load("facebookresearch/dinov3", model_name)
        model.eval()

        return model, feature_dim
    except Exception as e:
        raise RuntimeError(
            "Failed to load DINOv3 model. Please ensure you have internet access "
            "and the required dependencies. Error: " + str(e)
        ) from e


def extract_patch_feature(image, center_x, center_y, patch_size, model, transform):
    """Extract DINOv3 feature from a patch centered at (center_x, center_y).

    Args:
        image: Input image array (RGB format)
        center_x, center_y: Center coordinates of the patch
        patch_size: Size of the square patch to extract
        model: DINOv3 model
        transform: Image transform pipeline

    Returns:
        Tensor: Feature vector for the patch
    """
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
        y_offset = (patch_size - patch.shape[0]) // 2
        x_offset = (patch_size - patch.shape[1]) // 2
        padded_patch[
            y_offset : y_offset + patch.shape[0], x_offset : x_offset + patch.shape[1]
        ] = patch
        patch = padded_patch
    elif patch.shape[0] > patch_size or patch.shape[1] > patch_size:
        # Center crop if too large
        patch_center_y, patch_center_x = patch.shape[0] // 2, patch.shape[1] // 2
        half_size_actual = patch_size // 2
        patch = patch[
            patch_center_y - half_size_actual : patch_center_y + half_size_actual,
            patch_center_x - half_size_actual : patch_center_x + half_size_actual,
        ]

    # Convert to tensor and normalize
    patch_tensor = transform(patch).unsqueeze(0)

    with torch.no_grad():
        # Extract feature using the model's forward pass
        # DINOv3 returns a dictionary with 'x_norm_clstoken' and 'x_norm_patchtokens'
        output = model(patch_tensor)
        # Use the CLS token as the patch representation
        feature = output["x_norm_clstoken"] if isinstance(output, dict) else output

    return feature.squeeze(0)  # Remove batch dimension


def compute_similarity(feature1, feature2):
    """Compute cosine similarity between two feature vectors.

    Args:
        feature1: First feature tensor
        feature2: Second feature tensor

    Returns:
        float: Cosine similarity score
    """
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torch

    # Normalize features
    feature1_norm = torch.nn.functional.normalize(feature1, p=2, dim=0)
    feature2_norm = torch.nn.functional.normalize(feature2, p=2, dim=0)

    # Compute cosine similarity
    return torch.dot(feature1_norm, feature2_norm).item()


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
