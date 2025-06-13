"""Copy moles from multiple source images to target using DINOv2 semantic
matching."""

import argparse
import pathlib

import cv2

import mel.lib.dinov2
import mel.lib.image
import mel.rotomap.moles

# Constants
DEFAULT_MAX_SIZE = 910
DINOV2_PATCH_SIZE = 14
DINOV2_PATCH_CENTER_OFFSET = 7  # DINOV2_PATCH_SIZE // 2


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


def resize_image_if_needed(image, max_size=DEFAULT_MAX_SIZE):
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
    """Calculate context size for DINOv2 that's a multiple of patch size.

    Args:
        width, height: Image dimensions

    Returns:
        int: Context size that's a multiple of DINOV2_PATCH_SIZE and covers the image
    """
    max_dimension = max(width, height)
    # Round up to nearest multiple of DINOV2_PATCH_SIZE
    return (
        (max_dimension + DINOV2_PATCH_SIZE - 1) // DINOV2_PATCH_SIZE
    ) * DINOV2_PATCH_SIZE


def _is_mole_within_bounds(x, y, width, height):
    """Check if mole coordinates are within image bounds.

    Args:
        x, y: Mole coordinates
        width, height: Image dimensions

    Returns:
        bool: True if mole is within bounds
    """
    return 0 <= x < width and 0 <= y < height


def _convert_mole_to_patch_index(
    x_resized, y_resized, context_size, resized_width, resized_height
):
    """Convert mole coordinates to DINOv2 patch index.

    Args:
        x_resized, y_resized: Mole coordinates in resized image
        context_size: DINOv2 context size
        resized_width, resized_height: Resized image dimensions

    Returns:
        tuple: (patch_index, patches_per_side) or (None, None) if out of bounds
    """
    # Convert mole coordinates to context coordinates
    context_offset_x = (context_size - resized_width) // 2
    context_offset_y = (context_size - resized_height) // 2
    x_context = x_resized + context_offset_x
    y_context = y_resized + context_offset_y

    # Validate context coordinates are within bounds
    if (
        x_context < 0
        or y_context < 0
        or x_context >= context_size
        or y_context >= context_size
    ):
        return None, None

    # Convert to patch index
    patch_col = x_context // DINOV2_PATCH_SIZE
    patch_row = y_context // DINOV2_PATCH_SIZE
    patches_per_side = context_size // DINOV2_PATCH_SIZE

    # Validate patch indices are within bounds
    if patch_col >= patches_per_side or patch_row >= patches_per_side:
        return None, None

    patch_idx = patch_row * patches_per_side + patch_col

    return patch_idx, patches_per_side


def _convert_patch_index_to_coordinates(
    patch_idx, patches_per_side, context_size, resized_width, resized_height
):
    """Convert DINOv2 patch index to image coordinates.

    Args:
        patch_idx: Patch index
        patches_per_side: Number of patches per side
        context_size: DINOv2 context size
        resized_width, resized_height: Resized image dimensions

    Returns:
        tuple: (x_resized, y_resized) coordinates in resized image
    """
    patch_row = patch_idx // patches_per_side
    patch_col = patch_idx % patches_per_side

    # Get center of patch in context coordinates
    patch_center_x = patch_col * DINOV2_PATCH_SIZE + DINOV2_PATCH_CENTER_OFFSET
    patch_center_y = patch_row * DINOV2_PATCH_SIZE + DINOV2_PATCH_CENTER_OFFSET

    # Convert from context coordinates to resized image coordinates
    context_offset_x = (context_size - resized_width) // 2
    context_offset_y = (context_size - resized_height) // 2
    resized_x = patch_center_x - context_offset_x
    resized_y = patch_center_y - context_offset_y

    return resized_x, resized_y


def copy_moles_from_sources(
    src_paths,
    tgt_path,
    similarity_threshold,
    model,
    transform,
    feature_dim,
    max_size=DEFAULT_MAX_SIZE,
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
        print(f"Processing source: {src_path}")

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
        src_resized_height, src_resized_width = src_resized.shape[:2]

        if src_scale_factor < 1.0:
            print(
                f"  Source scaled by factor {src_scale_factor:.3f} to {src_resized_width}x{src_resized_height}"
            )
        else:
            print(
                f"  Source kept at original size {src_resized_width}x{src_resized_height}"
            )

        # Extract features for entire source image once
        src_context_size = calculate_dinov2_context_size(
            src_resized_width, src_resized_height
        )
        print(f"  Extracting source features (context size: {src_context_size})...")
        src_all_features = mel.lib.dinov2.extract_all_contextual_features(
            src_resized,
            src_resized_width // 2,
            src_resized_height // 2,
            src_context_size,
            model,
            transform,
            feature_dim,
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
            if not _is_mole_within_bounds(
                src_x_resized, src_y_resized, src_resized_width, src_resized_height
            ):
                print(f"    Skipping mole {uuid}: outside resized image")
                continue

            # Get features for source mole from pre-extracted features
            try:
                patch_idx, _ = _convert_mole_to_patch_index(
                    src_x_resized,
                    src_y_resized,
                    src_context_size,
                    src_resized_width,
                    src_resized_height,
                )

                # Check if patch index is valid
                if patch_idx is None:
                    print(f"    Skipping mole {uuid}: out of context bounds")
                    continue

                if patch_idx >= src_all_features.shape[0]:
                    print(
                        f"    Skipping mole {uuid}: patch index {patch_idx} out of bounds"
                    )
                    continue

                # Extract features for this specific patch
                src_features = src_all_features[patch_idx]
            except (IndexError, ValueError) as e:
                print(f"    Error getting features for mole {uuid}: {e}")
                continue

            # Find best match in target using cosine similarity
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
            tgt_patches_per_side = tgt_context_size // DINOV2_PATCH_SIZE
            resized_x, resized_y = _convert_patch_index_to_coordinates(
                best_patch_idx,
                tgt_patches_per_side,
                tgt_context_size,
                tgt_resized_width,
                tgt_resized_height,
            )

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
        print(f"Successfully copied {moles_copied} moles to {tgt_path}")

    else:
        print(f"No moles were copied to {tgt_path}")

    return moles_copied


def process_args(args):
    # Import dependencies lazily
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
            print(f"Operation completed: {moles_copied} moles copied")

            return 0
        print("Operation completed: no moles copied")
        return 0

    except OSError as e:
        print(f"Error accessing files: {e}")
        return 1
    except (ValueError, RuntimeError) as e:
        print(f"Error processing moles: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error copying moles: {e}")
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
