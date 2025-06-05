"""Copy moles from multiple source images to target using DINOv2 semantic
matching."""

import argparse
import pathlib

import cv2
import numpy as np

import mel.lib.dinov2
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
        "SRC_JPGS",
        nargs="+",
        type=_existing_file_path,
        help="Paths to source images with canonical moles to copy.",
    )
    parser.add_argument(
        "TGT_JPG",
        type=_existing_file_path,
        help="Path to target image to copy moles to.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Minimum similarity threshold for copying moles (default: 0.7).",
    )
    parser.add_argument(
        "--dino-size",
        type=str,
        choices=["small", "base", "large", "giant"],
        default="base",
        help="DINOv2 model size variant (default: base). Smaller models are faster.",
    )
    parser.add_argument(
        "--debug-images",
        action="store_true",
        help="Save debug images showing matches and similarities.",
    )


def resize_image_preserve_aspect(image, target_size=910):
    """Resize image to fit in target_size x target_size while preserving aspect
    ratio.

    Args:
        image: Input image array
        target_size: Maximum dimension size (default: 910)

    Returns:
        tuple: (resized_image, scale_factor, padded_image)
            - resized_image: Image resized to fit within target_size
            - scale_factor: Factor to scale coordinates back to original
            - padded_image: Image padded to exact target_size x target_size
    """
    height, width = image.shape[:2]

    # Calculate scale factor to fit the largest dimension
    scale_factor = target_size / max(height, width)

    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize the image
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    # Create padded image to exact target_size x target_size
    padded_image = np.zeros((target_size, target_size, 3), dtype=image.dtype)

    # Calculate padding offsets to center the image
    y_offset = (target_size - new_height) // 2
    x_offset = (target_size - new_width) // 2

    # Place resized image in center of padded image
    padded_image[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
        resized_image
    )

    return resized_image, scale_factor, padded_image


def scale_coordinates_from_context(
    x_context, y_context, context_size, scale_factor, original_width, original_height
):
    """Scale coordinates from context window back to original image size.

    Args:
        x_context, y_context: Coordinates in context window
        context_size: Size of context window (910)
        scale_factor: Scale factor used to resize original image
        original_width, original_height: Original image dimensions

    Returns:
        tuple: (x_original, y_original) in original image coordinates
    """
    # Remove padding offset to get coordinates in resized image
    resized_width = int(original_width * scale_factor)
    resized_height = int(original_height * scale_factor)

    x_offset = (context_size - resized_width) // 2
    y_offset = (context_size - resized_height) // 2

    x_resized = x_context - x_offset
    y_resized = y_context - y_offset

    # Scale back to original image size
    x_original = int(x_resized / scale_factor)
    y_original = int(y_resized / scale_factor)

    # Clamp to image bounds
    x_original = max(0, min(original_width - 1, x_original))
    y_original = max(0, min(original_height - 1, y_original))

    return x_original, y_original


def scale_mole_coordinates_to_context(
    mole, scale_factor, context_size, original_width, original_height
):
    """Scale mole coordinates from original image to context window.

    Args:
        mole: Mole dictionary with x, y coordinates
        scale_factor: Scale factor used to resize original image
        context_size: Size of context window (910)
        original_width, original_height: Original image dimensions

    Returns:
        tuple: (x_context, y_context) in context window coordinates
    """
    # Scale coordinates to resized image
    x_resized = int(mole["x"] * scale_factor)
    y_resized = int(mole["y"] * scale_factor)

    # Add padding offset to get coordinates in context window
    resized_width = int(original_width * scale_factor)
    resized_height = int(original_height * scale_factor)

    x_offset = (context_size - resized_width) // 2
    y_offset = (context_size - resized_height) // 2

    x_context = x_resized + x_offset
    y_context = y_resized + y_offset

    return x_context, y_context


def copy_moles_from_sources(
    src_paths,
    tgt_path,
    similarity_threshold,
    model,
    transform,
    feature_dim,
    context_size=910,
    debug_images=False,
):
    """Copy canonical moles from source images to target image using DINOv2
    matching.

    Args:
        src_paths: List of source image paths
        tgt_path: Target image path
        similarity_threshold: Minimum similarity for copying
        model: DINOv2 model wrapper
        transform: Image transform pipeline
        feature_dim: DINOv2 feature dimension
        context_size: Context window size (default: 910)
        debug_images: Whether to save debug images

    Returns:
        int: Number of moles copied
    """
    # Load target image and moles
    tgt_image = mel.lib.image.load_image(tgt_path)
    tgt_image_rgb = cv2.cvtColor(tgt_image, cv2.COLOR_BGR2RGB)
    tgt_moles = mel.rotomap.moles.load_image_moles(tgt_path)

    print(f"Target image: {tgt_path} ({tgt_image.shape[1]}x{tgt_image.shape[0]})")
    print(f"Target has {len(tgt_moles)} existing moles")

    # Resize target image preserving aspect ratio
    _, tgt_scale_factor, tgt_padded = resize_image_preserve_aspect(
        tgt_image_rgb, context_size
    )
    tgt_original_height, tgt_original_width = tgt_image.shape[:2]

    print(
        f"Target scaled by factor {tgt_scale_factor:.3f} to fit {context_size}x{context_size}"
    )

    # Extract features for all patches in target
    print("Extracting target features...")
    tgt_all_features = mel.lib.dinov2.extract_all_contextual_features(
        tgt_padded,
        context_size // 2,
        context_size // 2,
        context_size,
        model,
        transform,
        feature_dim,
    )

    # Create lookup for existing target moles by UUID
    tgt_moles_by_uuid = {mole["uuid"]: mole for mole in tgt_moles}
    canonical_uuids = {
        mole["uuid"] for mole in tgt_moles if mole[mel.rotomap.moles.KEY_IS_CONFIRMED]
    }

    moles_copied = 0

    # Process each source image
    for src_path in src_paths:
        print(f"\nProcessing source: {src_path}")

        # Load source image and moles
        src_image = mel.lib.image.load_image(src_path)
        src_image_rgb = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        src_moles = mel.rotomap.moles.load_image_moles(src_path)

        # Filter to canonical moles only
        src_canonical_moles = [
            mole for mole in src_moles if mole[mel.rotomap.moles.KEY_IS_CONFIRMED]
        ]

        if not src_canonical_moles:
            print(f"  No canonical moles found in {src_path}")
            continue

        print(f"  Found {len(src_canonical_moles)} canonical moles")

        # Resize source image preserving aspect ratio
        _, src_scale_factor, src_padded = resize_image_preserve_aspect(
            src_image_rgb, context_size
        )
        src_original_height, src_original_width = src_image.shape[:2]

        print(f"  Source scaled by factor {src_scale_factor:.3f}")

        # Process each canonical mole in source
        for src_mole in src_canonical_moles:
            uuid = src_mole["uuid"]

            # Skip if target already has canonical mole with this UUID
            if uuid in canonical_uuids:
                print(
                    f"    Skipping mole {uuid}: canonical mole already exists in target"
                )
                continue

            # Scale source mole coordinates to context window
            src_x_context, src_y_context = scale_mole_coordinates_to_context(
                src_mole,
                src_scale_factor,
                context_size,
                src_original_width,
                src_original_height,
            )

            # Check if source mole is within padded area
            if (
                src_x_context < 0
                or src_x_context >= context_size
                or src_y_context < 0
                or src_y_context >= context_size
            ):
                print(f"    Skipping mole {uuid}: outside context window")
                continue

            # Extract features for source mole
            try:
                src_features = mel.lib.dinov2.extract_contextual_patch_feature(
                    src_padded,
                    src_x_context,
                    src_y_context,
                    context_size,
                    model,
                    transform,
                    feature_dim,
                )
            except Exception as e:
                print(f"    Error extracting features for mole {uuid}: {e}")
                continue

            # Find best match in target using cosine similarity
            # Import torch lazily
            import torch

            # Normalize features for cosine similarity
            src_norm = torch.nn.functional.normalize(
                src_features, p=2, dim=0
            )  # [feature_dim]
            tgt_norm = torch.nn.functional.normalize(
                tgt_all_features, p=2, dim=1
            )  # [num_patches, feature_dim]

            # Compute cosine similarities
            similarities = torch.mm(tgt_norm, src_norm.unsqueeze(1)).squeeze(
                1
            )  # [num_patches]

            # Find best match
            best_patch_idx = torch.argmax(similarities).item()
            best_similarity = similarities[best_patch_idx].item()

            print(f"    Mole {uuid}: best similarity = {best_similarity:.3f}")

            # Check similarity threshold
            if best_similarity < similarity_threshold:
                print(
                    f"    Skipping mole {uuid}: similarity {best_similarity:.3f} below threshold {similarity_threshold}"
                )
                continue

            # Convert patch index to context coordinates
            patches_per_side = context_size // 14  # DINOv2 patch size is 14x14
            patch_row = best_patch_idx // patches_per_side
            patch_col = best_patch_idx % patches_per_side

            # Get center of patch in context coordinates
            patch_center_x = patch_col * 14 + 7
            patch_center_y = patch_row * 14 + 7

            # Scale back to original target coordinates
            tgt_x_original, tgt_y_original = scale_coordinates_from_context(
                patch_center_x,
                patch_center_y,
                context_size,
                tgt_scale_factor,
                tgt_original_width,
                tgt_original_height,
            )

            # Check if we should add/update this mole
            should_add = False

            if uuid not in tgt_moles_by_uuid:
                # New mole
                should_add = True
                action = "Adding new"
            else:
                # Existing non-canonical mole
                existing_mole = tgt_moles_by_uuid[uuid]
                existing_similarity = existing_mole.get("dinov2_similarity", 0.0)

                if best_similarity > existing_similarity:
                    should_add = True
                    action = f"Updating (prev similarity: {existing_similarity:.3f})"
                else:
                    print(
                        f"    Skipping mole {uuid}: existing similarity {existing_similarity:.3f} >= new {best_similarity:.3f}"
                    )
                    continue

            if should_add:
                print(
                    f"    {action} mole {uuid} at ({tgt_x_original}, {tgt_y_original}) with similarity {best_similarity:.3f}"
                )

                # Create or update mole
                new_mole = {
                    "uuid": uuid,
                    "x": tgt_x_original,
                    "y": tgt_y_original,
                    mel.rotomap.moles.KEY_IS_CONFIRMED: False,  # Non-canonical
                    "dinov2_similarity": best_similarity,
                }

                if uuid in tgt_moles_by_uuid:
                    # Update existing mole
                    existing_mole = tgt_moles_by_uuid[uuid]
                    existing_mole.update(new_mole)
                else:
                    # Add new mole
                    tgt_moles.append(new_mole)
                    tgt_moles_by_uuid[uuid] = new_mole

                moles_copied += 1

    # Save updated target moles
    if moles_copied > 0:
        mel.rotomap.moles.save_image_moles(tgt_moles, tgt_path)
        print(f"\nSuccessfully copied {moles_copied} moles to {tgt_path}")
    else:
        print(f"\nNo moles were copied to {tgt_path}")

    return moles_copied


def process_args(args):
    # Import torchvision lazily
    import torchvision.transforms as transforms

    src_paths = args.SRC_JPGS
    tgt_path = args.TGT_JPG
    similarity_threshold = args.similarity_threshold
    dino_size = args.dino_size
    debug_images = args.debug_images

    # Validate arguments
    if tgt_path in src_paths:
        print("Error: Target image cannot be one of the source images")
        return 1

    if not (0.0 <= similarity_threshold <= 1.0):
        print("Error: Similarity threshold must be between 0.0 and 1.0")
        return 1

    print(f"Copying moles from {len(src_paths)} source images to {tgt_path}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"DINOv2 model size: {dino_size}")

    # Load DINOv2 model
    try:
        model, feature_dim = mel.lib.dinov2.load_dinov2_model(dino_size)
        print(
            f"DINOv2 model ({dino_size}) loaded successfully with {feature_dim} feature dimensions"
        )
    except RuntimeError as e:
        print(f"Error loading DINOv2 model: {e}")
        return 1

    # Create transform pipeline
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Copy moles
    try:
        moles_copied = copy_moles_from_sources(
            src_paths,
            tgt_path,
            similarity_threshold,
            model,
            transform,
            feature_dim,
            debug_images=debug_images,
        )

        if moles_copied > 0:
            print(f"\nOperation completed: {moles_copied} moles copied")
            return 0
        print("\nOperation completed: no moles copied")
        return 0

    except Exception as e:
        print(f"Error copying moles: {e}")
        return 1


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
