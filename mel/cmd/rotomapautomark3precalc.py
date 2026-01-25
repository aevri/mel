"""Pre-compute DINOv3 features for images.

Saves features to {image}.dino3-{size}-{imagesize}.pt for use by
automark3.
"""

import argparse
import pathlib

import cv2
import torch

import mel.lib.dinov3
import mel.lib.image
import mel.rotomap.mask


def _existing_file_path(string):
    """Argparse type for validating that a file exists."""
    path = pathlib.Path(string)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File does not exist: {string}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Path is not a file: {string}")
    return path


def _get_features_path(image_path, dino_size, image_size):
    """Generate path for cached features file."""
    return pathlib.Path(f"{image_path}.dino3-{dino_size}-{image_size}.pt")


def _load_image_with_mask(image_path, verbose=False):
    """Load an image and apply its mask if available."""
    if verbose:
        print(f"Loading image: {image_path}")
    image_rgb = cv2.cvtColor(mel.lib.image.load_image(image_path), cv2.COLOR_BGR2RGB)
    mask = mel.rotomap.mask.load_or_none(image_path)
    if mask is not None:
        if verbose:
            print("  Applying mask to image")
        image_rgb = mel.lib.dinov3.apply_mask(image_rgb, mask)
    return image_rgb


def setup_parser(parser):
    parser.add_argument(
        "JPG",
        type=_existing_file_path,
        nargs="+",
        help="Image file(s) to pre-compute features for.",
    )
    parser.add_argument(
        "--dino-size",
        type=str,
        choices=["small", "base", "large", "huge", "7b"],
        default="base",
        help="DINOv3 model size variant (default: base).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Scale images to fit this size in pixels (default: 1024). "
        "Must be divisible by 16.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading model weights from Hugging Face Hub. "
        "By default, only cached models are used.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed processing information.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing feature files.",
    )


def process_args(args):
    image_paths = args.JPG
    dino_size = args.dino_size
    image_size = args.image_size
    allow_download = args.allow_download
    verbose = args.verbose
    force = args.force

    # Validate image_size is divisible by patch size
    if image_size % mel.lib.dinov3.PATCH_SIZE != 0:
        print(
            f"Error: --image-size must be divisible by "
            f"{mel.lib.dinov3.PATCH_SIZE}, got {image_size}"
        )
        return 1

    # Filter images that need processing
    images_to_process = []
    for image_path in image_paths:
        features_path = _get_features_path(image_path, dino_size, image_size)
        if features_path.exists() and not force:
            if verbose:
                print(f"Skipping (already exists): {features_path}")
            continue
        images_to_process.append(image_path)

    if not images_to_process:
        if verbose:
            print("All features already computed, nothing to do.")
        return 0

    # Load DINOv3 model
    if verbose:
        print(f"Loading DINOv3 model (size: {dino_size})...")
    try:
        model, feature_dim = mel.lib.dinov3.load_dinov3_model(
            dino_size, local_files_only=not allow_download
        )
        if verbose:
            print(f"Model loaded with {feature_dim} feature dimensions")
    except RuntimeError as e:
        print(f"Error loading DINOv3 model: {e}")
        return 1

    # Process each image
    for image_path in images_to_process:
        features_path = _get_features_path(image_path, dino_size, image_size)

        if verbose:
            print(f"\nProcessing: {image_path}")

        # Load and scale image
        image_rgb = _load_image_with_mask(image_path, verbose)
        scaled_image, (scale_x, scale_y) = mel.lib.dinov3.scale_image_to_fit(
            image_rgb, image_size
        )
        scaled_h, scaled_w = scaled_image.shape[:2]

        if verbose:
            print(f"  Scaled to {scaled_w}x{scaled_h}")

        # Extract features
        if verbose:
            print("  Extracting features...")
        features = mel.lib.dinov3.extract_all_patch_features(scaled_image, model)

        if verbose:
            print(f"  Features shape: {features.shape}")

        # Save features with metadata
        data = {
            "features": features.cpu(),
            "scaled_h": scaled_h,
            "scaled_w": scaled_w,
            "scale_x": scale_x,
            "scale_y": scale_y,
        }
        torch.save(data, features_path)

        if verbose:
            print(f"  Saved: {features_path}")
        else:
            print(f"Saved: {features_path}")

    return 0


# -----------------------------------------------------------------------------
# Copyright (C) 2026 Angelos Evripiotis.
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
