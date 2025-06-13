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


def _calculate_multiscale_zoom_levels(image_width, image_height, target_size=910):
    """Calculate zoom levels for multi-scale feature extraction.

    Args:
        image_width, image_height: Original image dimensions
        target_size: Target context size (default: 910)

    Returns:
        tuple: (wide_scale, medium_scale, close_scale)
    """
    # Wide zoom: scale to fit whole image in target_size
    max_dimension = max(image_width, image_height)
    wide_scale = target_size / max_dimension

    # Close zoom: full resolution
    close_scale = 1.0

    # Medium zoom: midpoint between wide and close
    medium_scale = (wide_scale + close_scale) / 2.0

    return wide_scale, medium_scale, close_scale


def _crop_around_point(image, center_x, center_y, size):
    """Crop a square region around a point, with padding if needed.

    Args:
        image: Input image array
        center_x, center_y: Center coordinates
        size: Target crop size

    Returns:
        Cropped image of exact target size
    """
    h, w = image.shape[:2]
    half_size = size // 2

    # Calculate crop bounds
    x1 = max(0, center_x - half_size)
    y1 = max(0, center_y - half_size)
    x2 = min(w, center_x + half_size)
    y2 = min(h, center_y + half_size)

    # Crop the region
    cropped = image[y1:y2, x1:x2]

    # Pad to exact size if needed (resize to target size)
    if cropped.shape[0] != size or cropped.shape[1] != size:
        cropped = cv2.resize(cropped, (size, size))

    return cropped


def _extract_multiscale_contexts(image, mole_x, mole_y, target_size=910):
    """Extract contexts at multiple zoom levels around a mole.

    Args:
        image: Full resolution image
        mole_x, mole_y: Mole coordinates in full resolution
        target_size: Target context size (default: 910)

    Returns:
        dict: Contexts at different scales {'wide': array, 'medium': array, 'close': array}
    """
    original_height, original_width = image.shape[:2]

    # Calculate zoom levels
    wide_scale, medium_scale, _ = _calculate_multiscale_zoom_levels(
        original_width, original_height, target_size
    )

    contexts = {}

    # Close zoom (1x): Direct crop from full resolution
    contexts["close"] = _crop_around_point(image, mole_x, mole_y, target_size)

    # Medium zoom: Scale down and crop
    medium_width = int(original_width * medium_scale)
    medium_height = int(original_height * medium_scale)
    medium_image = cv2.resize(image, (medium_width, medium_height))
    medium_mole_x = int(mole_x * medium_scale)
    medium_mole_y = int(mole_y * medium_scale)
    contexts["medium"] = _crop_around_point(
        medium_image, medium_mole_x, medium_mole_y, target_size
    )

    # Wide zoom: Scale down to fit whole image, then crop
    wide_width = int(original_width * wide_scale)
    wide_height = int(original_height * wide_scale)
    wide_image = cv2.resize(image, (wide_width, wide_height))
    wide_mole_x = int(mole_x * wide_scale)
    wide_mole_y = int(mole_y * wide_scale)
    contexts["wide"] = _crop_around_point(
        wide_image, wide_mole_x, wide_mole_y, target_size
    )

    return contexts


def _extract_features_per_scale(contexts, model, transform, feature_dim):
    """Extract DINOv2 features for each zoom level context.

    Args:
        contexts: Dict of contexts at different scales
        model: DINOv2 model
        transform: Image transform pipeline
        feature_dim: Feature dimension

    Returns:
        dict: Features for each scale
    """
    scale_features = {}
    target_size = 910
    center_x = target_size // 2
    center_y = target_size // 2

    for scale_name, context in contexts.items():
        try:
            # Extract features from center patch of this context
            features = mel.lib.dinov2.extract_contextual_patch_feature(
                context, center_x, center_y, target_size, model, transform, feature_dim
            )
            scale_features[scale_name] = features
        except Exception as e:
            print(f"    Warning: Failed to extract {scale_name} scale features: {e}")
            continue

    return scale_features


def _concatenate_multiscale_features(scale_features):
    """Concatenate features from different zoom levels.

    Args:
        scale_features: Dict of features for each scale

    Returns:
        Concatenated feature vector or None if no valid features
    """
    import torch

    # Concatenate in order: wide, medium, close
    feature_list = []
    for scale_name in ["wide", "medium", "close"]:
        if scale_name in scale_features:
            feature_list.append(scale_features[scale_name])

    if not feature_list:
        return None

    # Concatenate all available features
    return torch.cat(feature_list, dim=0)


def _extract_all_multiscale_features(
    image, context_size, model, transform, feature_dim
):
    """Extract multi-scale features for all patches in target image.

    Args:
        image: Target image
        context_size: DINOv2 context size for patch grid
        model: DINOv2 model
        transform: Image transform pipeline
        feature_dim: Feature dimension

    Returns:
        Tensor of concatenated multi-scale features for all patches
    """
    import torch

    print("  Extracting multi-scale features for all target patches...")
    original_height, original_width = image.shape[:2]
    patches_per_side = context_size // DINOV2_PATCH_SIZE
    total_patches = patches_per_side * patches_per_side

    # Calculate zoom levels for target image
    wide_scale, medium_scale, _ = _calculate_multiscale_zoom_levels(
        original_width, original_height, context_size
    )

    # Pre-scale images for efficiency
    wide_image = cv2.resize(
        image, (int(original_width * wide_scale), int(original_height * wide_scale))
    )
    medium_image = cv2.resize(
        image, (int(original_width * medium_scale), int(original_height * medium_scale))
    )
    close_image = image  # Full resolution

    all_multiscale_features = []

    for patch_idx in range(total_patches):
        # Convert patch index to coordinates
        patch_row = patch_idx // patches_per_side
        patch_col = patch_idx % patches_per_side

        # Get center coordinates in original image
        patch_center_x = patch_col * DINOV2_PATCH_SIZE + DINOV2_PATCH_CENTER_OFFSET
        patch_center_y = patch_row * DINOV2_PATCH_SIZE + DINOV2_PATCH_CENTER_OFFSET

        # Convert from context coordinates to original image coordinates
        context_offset_x = (context_size - original_width) // 2
        context_offset_y = (context_size - original_height) // 2
        orig_x = patch_center_x - context_offset_x
        orig_y = patch_center_y - context_offset_y

        # Skip if coordinates are outside original image
        if (
            orig_x < 0
            or orig_y < 0
            or orig_x >= original_width
            or orig_y >= original_height
        ):
            # Create zero features for out-of-bounds patches
            zero_features = torch.zeros(feature_dim * 3)  # 3 scales
            all_multiscale_features.append(zero_features)
            continue

        try:
            # Extract contexts at each scale
            contexts = {}

            # Close scale (1x)
            contexts["close"] = _crop_around_point(close_image, orig_x, orig_y, 910)

            # Medium scale
            medium_x = int(orig_x * medium_scale)
            medium_y = int(orig_y * medium_scale)
            contexts["medium"] = _crop_around_point(
                medium_image, medium_x, medium_y, 910
            )

            # Wide scale
            wide_x = int(orig_x * wide_scale)
            wide_y = int(orig_y * wide_scale)
            contexts["wide"] = _crop_around_point(wide_image, wide_x, wide_y, 910)

            # Extract features for each scale
            scale_features = _extract_features_per_scale(
                contexts, model, transform, feature_dim
            )

            # Concatenate features
            patch_features = _concatenate_multiscale_features(scale_features)

            if patch_features is None:
                # Use zero features if extraction failed
                patch_features = torch.zeros(feature_dim * 3)

            all_multiscale_features.append(patch_features)

        except Exception as e:
            print(f"    Warning: Failed to extract features for patch {patch_idx}: {e}")
            zero_features = torch.zeros(feature_dim * 3)
            all_multiscale_features.append(zero_features)

    # Stack all features
    return torch.stack(all_multiscale_features)


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

    # Extract multi-scale features for all patches in target
    print("Extracting multi-scale target features...")
    tgt_all_features = _extract_all_multiscale_features(
        tgt_image_rgb,  # Use original image, not resized
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

        # Process each canonical mole in source
        for src_mole in src_canonical_moles:
            uuid = src_mole["uuid"]

            # Skip if target already has canonical mole with this UUID
            if uuid in canonical_uuids:
                print(
                    f"    Skipping mole {uuid}: canonical mole already exists in target"
                )
                continue

            # Extract multi-scale features for source mole
            try:
                # Extract contexts at multiple zoom levels from original source image
                src_contexts = _extract_multiscale_contexts(
                    src_image_rgb,  # Use original, not resized
                    src_mole["x"],  # Original coordinates
                    src_mole["y"],
                )

                # Extract features for each scale
                src_scale_features = _extract_features_per_scale(
                    src_contexts, model, transform, feature_dim
                )

                # Use concatenated multi-scale features
                src_features = _concatenate_multiscale_features(src_scale_features)

                if src_features is None:
                    print(
                        f"    Skipping mole {uuid}: failed to extract multi-scale features"
                    )
                    continue

            except Exception as e:
                print(f"    Error extracting multi-scale features for mole {uuid}: {e}")
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
