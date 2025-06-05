"""Copy moles from multiple source images to target using DINOv2 semantic
matching."""

import argparse
import pathlib

import cv2

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


def resize_image_if_needed(image, max_size=910):
    """Resize image only if it exceeds max_size, preserving aspect ratio.

    Args:
        image: Input image array
        max_size: Maximum dimension size (default: 910)

    Returns:
        tuple: (resized_image, scale_factor)
            - resized_image: Image resized to fit within max_size (or original if no resize needed)
            - scale_factor: Factor to scale coordinates back to original (1.0 if no resize)
    """
    height, width = image.shape[:2]
    max_dimension = max(height, width)

    # Only resize if the image exceeds max_size
    if max_dimension <= max_size:
        return image, 1.0

    # Calculate scale factor to fit the largest dimension
    scale_factor = max_size / max_dimension

    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize the image
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    return resized_image, scale_factor


def scale_coordinates_from_resized(
    x_resized, y_resized, scale_factor, original_width, original_height
):
    """Scale coordinates from resized image back to original image size.

    Args:
        x_resized, y_resized: Coordinates in resized image
        scale_factor: Scale factor used to resize original image
        original_width, original_height: Original image dimensions

    Returns:
        tuple: (x_original, y_original) in original image coordinates
    """
    # Scale back to original image size
    x_original = int(x_resized / scale_factor)
    y_original = int(y_resized / scale_factor)

    # Clamp to image bounds
    x_original = max(0, min(original_width - 1, x_original))
    y_original = max(0, min(original_height - 1, y_original))

    return x_original, y_original


def scale_mole_coordinates_to_resized(mole, scale_factor):
    """Scale mole coordinates from original image to resized image.

    Args:
        mole: Mole dictionary with x, y coordinates
        scale_factor: Scale factor used to resize original image

    Returns:
        tuple: (x_resized, y_resized) in resized image coordinates
    """
    # Scale coordinates to resized image
    x_resized = int(mole["x"] * scale_factor)
    y_resized = int(mole["y"] * scale_factor)

    return x_resized, y_resized


def calculate_dinov2_context_size(width, height):
    """Calculate context size for DINOv2 that's a multiple of 14.

    Args:
        width, height: Image dimensions

    Returns:
        int: Context size that's a multiple of 14 and covers the image
    """
    max_dimension = max(width, height)
    # Round up to nearest multiple of 14
    return ((max_dimension + 13) // 14) * 14


def copy_moles_from_sources(
    src_paths,
    tgt_path,
    similarity_threshold,
    model,
    transform,
    feature_dim,
    max_size=910,
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
        max_size: Maximum image dimension (default: 910)

    Returns:
        int: Number of moles copied
    """
    # Load target image and moles
    tgt_image = mel.lib.image.load_image(tgt_path)
    tgt_image_rgb = cv2.cvtColor(tgt_image, cv2.COLOR_BGR2RGB)
    tgt_moles = mel.rotomap.moles.load_image_moles(tgt_path)

    print(f"Target image: {tgt_path} ({tgt_image.shape[1]}x{tgt_image.shape[0]})")
    print(f"Target has {len(tgt_moles)} existing moles")

    # Resize target image if needed
    tgt_resized, tgt_scale_factor = resize_image_if_needed(tgt_image_rgb, max_size)
    tgt_original_height, tgt_original_width = tgt_image.shape[:2]
    tgt_resized_height, tgt_resized_width = tgt_resized.shape[:2]

    if tgt_scale_factor < 1.0:
        print(
            f"Target scaled by factor {tgt_scale_factor:.3f} to {tgt_resized_width}x{tgt_resized_height}"
        )
    else:
        print(f"Target kept at original size {tgt_resized_width}x{tgt_resized_height}")

    # Calculate DINOv2 context size (must be multiple of 14)
    tgt_context_size = calculate_dinov2_context_size(
        tgt_resized_width, tgt_resized_height
    )

    # Extract features for all patches in target
    print("Extracting target features...")
    tgt_all_features = mel.lib.dinov2.extract_all_contextual_features(
        tgt_resized,
        tgt_resized_width // 2,
        tgt_resized_height // 2,
        tgt_context_size,
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

        # Resize source image if needed
        src_resized, src_scale_factor = resize_image_if_needed(src_image_rgb, max_size)
        _, _ = src_image.shape[:2]
        src_resized_height, src_resized_width = src_resized.shape[:2]

        if src_scale_factor < 1.0:
            print(
                f"  Source scaled by factor {src_scale_factor:.3f} to {src_resized_width}x{src_resized_height}"
            )
        else:
            print(
                f"  Source kept at original size {src_resized_width}x{src_resized_height}"
            )

        # Process each canonical mole in source
        for src_mole in src_canonical_moles:
            uuid = src_mole["uuid"]

            # Skip if target already has canonical mole with this UUID
            if uuid in canonical_uuids:
                print(
                    f"    Skipping mole {uuid}: canonical mole already exists in target"
                )
                continue

            # Scale source mole coordinates to resized image
            src_x_resized, src_y_resized = scale_mole_coordinates_to_resized(
                src_mole, src_scale_factor
            )

            # Check if source mole is within resized image
            if (
                src_x_resized < 0
                or src_x_resized >= src_resized_width
                or src_y_resized < 0
                or src_y_resized >= src_resized_height
            ):
                print(f"    Skipping mole {uuid}: outside resized image")
                continue

            # Extract features for source mole
            try:
                src_context_size = calculate_dinov2_context_size(
                    src_resized_width, src_resized_height
                )
                src_features = mel.lib.dinov2.extract_contextual_patch_feature(
                    src_resized,
                    src_x_resized,
                    src_y_resized,
                    src_context_size,
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

            # Convert patch index to resized image coordinates
            patches_per_side = tgt_context_size // 14  # DINOv2 patch size is 14x14
            patch_row = best_patch_idx // patches_per_side
            patch_col = best_patch_idx % patches_per_side

            # Get center of patch in context coordinates
            patch_center_x = patch_col * 14 + 7
            patch_center_y = patch_row * 14 + 7

            # Convert from context coordinates to resized image coordinates
            # Context is centered on the resized image
            context_offset_x = (tgt_context_size - tgt_resized_width) // 2
            context_offset_y = (tgt_context_size - tgt_resized_height) // 2
            resized_x = patch_center_x - context_offset_x
            resized_y = patch_center_y - context_offset_y

            # Scale back to original target coordinates
            tgt_x_original, tgt_y_original = scale_coordinates_from_resized(
                resized_x,
                resized_y,
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
