"""Locate canonical moles from a source image in target images using DINOv3.

Uses a global view with images scaled to fit --image-size.

Moles found in target images are added as non-canonical
(is_uuid_canonical=False).
"""

import argparse
import pathlib

import cv2
import torch

import mel.lib.dinov3
import mel.lib.image
import mel.rotomap.mask
import mel.rotomap.moles


def _existing_file_path(string):
    """Argparse type for validating that a file exists."""
    path = pathlib.Path(string)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File does not exist: {string}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Path is not a file: {string}")
    return path


def _load_image_with_mask(image_path, verbose=False):
    """Load an image and apply its mask if available."""
    if verbose:
        print(f"Loading image: {image_path}")
    image_rgb = cv2.cvtColor(mel.lib.image.load_image(image_path), cv2.COLOR_BGR2RGB)
    mask = mel.rotomap.mask.load_or_none(image_path)
    if mask is not None:
        if verbose:
            print("Applying mask to image")
        image_rgb = mel.lib.dinov3.apply_mask(image_rgb, mask)
    return image_rgb


def _get_features_path(image_path, dino_size, image_size):
    """Generate path for cached features file."""
    return pathlib.Path(f"{image_path}.dino3-{dino_size}-{image_size}.pt")


def _load_cached_features(image_path, dino_size, image_size, verbose=False):
    """Load cached features if available, returns None if not found."""
    features_path = _get_features_path(image_path, dino_size, image_size)
    if not features_path.exists():
        return None
    if verbose:
        print(f"Loading cached features: {features_path}")
    # Returns dict with features, scaled_h, scaled_w, scale_x, scale_y
    return torch.load(features_path)


def _tensor_size_mb(tensor):
    """Return size of a tensor in megabytes."""
    return tensor.numel() * tensor.element_size() / (1024 * 1024)


def _get_patch_index(img_w, mole_x, mole_y):
    """Convert scaled coordinates to patch index.

    Args:
        img_w: Scaled image width
        mole_x, mole_y: Mole coordinates (already scaled)

    Returns:
        int: Patch index
    """
    patch_size = mel.lib.dinov3.PATCH_SIZE
    patches_per_row = img_w // patch_size
    patch_col = mole_x // patch_size
    patch_row = mole_y // patch_size
    return patch_row * patches_per_row + patch_col


def _find_best_patch_index(mole_feature, all_features):
    """Find index of best matching patch using cosine similarity.

    Args:
        mole_feature: Feature vector for the mole patch
        all_features: All patch features [num_patches, feature_dim]

    Returns:
        int: Index of best matching patch
    """
    similarities = mel.lib.dinov3.compute_similarities(
        mole_feature, all_features, similarity_type="cosine"
    )
    return similarities.argmax().item()


def _get_mole_feature(all_features, img_w, mole_x, mole_y):
    """Extract mole feature from pre-computed all_features.

    Args:
        all_features: All patch features [num_patches, feature_dim]
        img_w: Scaled image width
        mole_x, mole_y: Mole coordinates (already scaled)

    Returns:
        Tensor: Feature vector [feature_dim]
    """
    patch_idx = _get_patch_index(img_w, mole_x, mole_y)
    return all_features[patch_idx]


def _find_mole_match(
    mole_feature,
    target_features,
    tgt_scale_x,
    tgt_scale_y,
    scaled_tgt_h,
    scaled_tgt_w,
    verbose=False,
):
    """Find best match for a mole using pre-computed features.

    Returns:
        Tuple of (native_x, native_y, scaled_x, scaled_y, max_similarity) where
        native_x, native_y are coordinates in the target image at original size,
        scaled_x, scaled_y are coordinates in the scaled target image.
    """
    similarities = mel.lib.dinov3.compute_similarities(
        mole_feature, target_features, similarity_type="cosine"
    )

    max_sim = similarities.max().item()
    if verbose:
        print(f"    Similarity range: {similarities.min().item():.4f} to {max_sim:.4f}")

    best_x, best_y = mel.lib.dinov3.find_best_match_location(
        similarities, scaled_tgt_h, scaled_tgt_w, "cosine"
    )

    # Convert to native coords
    final_x = int(best_x / tgt_scale_x)
    final_y = int(best_y / tgt_scale_y)

    if verbose:
        print(f"    Best match at native coords: ({final_x}, {final_y})")

    return final_x, final_y, best_x, best_y, max_sim


def setup_parser(parser):
    parser.add_argument(
        "SRC_JPG",
        type=_existing_file_path,
        help="Source image with canonical moles to match from.",
    )
    parser.add_argument(
        "TGT_JPG",
        type=_existing_file_path,
        nargs="+",
        help="Target image(s) where moles will be located.",
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
        "--extra-stem",
        help="Save to alternate mole file (e.g., '0.jpg.EXTRA.json').",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed processing information.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="Minimum similarity threshold for matches (default: 0.0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without saving.",
    )


def process_args(args):
    src_path = args.SRC_JPG
    tgt_paths = args.TGT_JPG
    dino_size = args.dino_size
    image_size = args.image_size
    allow_download = args.allow_download
    extra_stem = args.extra_stem
    verbose = args.verbose
    min_similarity = args.min_similarity
    dry_run = args.dry_run

    # Validate image_size is divisible by patch size
    if image_size % mel.lib.dinov3.PATCH_SIZE != 0:
        print(
            f"Error: --image-size must be divisible by "
            f"{mel.lib.dinov3.PATCH_SIZE}, got {image_size}"
        )
        return 1

    # Load source moles
    try:
        src_moles = mel.rotomap.moles.load_image_moles(src_path)
    except Exception as e:
        print(f"Error loading source moles: {e}")
        return 1

    # Get canonical moles from source
    src_canonical_moles = [
        m for m in src_moles if m[mel.rotomap.moles.KEY_IS_CONFIRMED]
    ]

    if not src_canonical_moles:
        print("Error: No canonical moles found in source image")
        return 1

    if verbose:
        print(f"Source has {len(src_canonical_moles)} canonical moles")

    # Try to load cached source features
    src_cached = _load_cached_features(src_path, dino_size, image_size, verbose)

    # Check which targets have cached features
    tgt_cached = {}
    targets_needing_computation = []
    for tgt_path in tgt_paths:
        cached = _load_cached_features(tgt_path, dino_size, image_size, verbose)
        if cached:
            tgt_cached[tgt_path] = cached
        else:
            targets_needing_computation.append(tgt_path)

    # Only load model if needed
    need_model = src_cached is None or len(targets_needing_computation) > 0
    if need_model:
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
    else:
        model = None
        if verbose:
            print("All features cached, skipping model load")

    # Get source features (from cache or compute)
    if src_cached is not None:
        src_all_features = src_cached["features"]
        src_scale_x = src_cached["scale_x"]
        src_scale_y = src_cached["scale_y"]
        scaled_src_w = src_cached["scaled_w"]
        if verbose:
            print(
                f"Using cached source features: {src_all_features.shape[0]} patches, "
                f"{_tensor_size_mb(src_all_features):.1f} MB"
            )
    else:
        # Load and scale source image once, extract all features
        src_image_rgb = _load_image_with_mask(src_path, verbose)
        if verbose:
            print("Scaling source image and extracting features...")
        scaled_src, (src_scale_x, src_scale_y) = mel.lib.dinov3.scale_image_to_fit(
            src_image_rgb, image_size
        )
        scaled_src_w = scaled_src.shape[1]
        src_all_features = mel.lib.dinov3.extract_all_patch_features(scaled_src, model)
        if verbose:
            print(
                f"  Source features: {src_all_features.shape[0]} patches, "
                f"{_tensor_size_mb(src_all_features):.1f} MB"
            )

    # Get individual mole features and patch indices from the pre-computed features
    src_mole_features = {}
    src_mole_patch_indices = {}
    for mole in src_canonical_moles:
        mole_uuid = mole["uuid"]
        scaled_mole_x = int(mole["x"] * src_scale_x)
        scaled_mole_y = int(mole["y"] * src_scale_y)
        src_mole_features[mole_uuid] = _get_mole_feature(
            src_all_features,
            scaled_src_w,
            scaled_mole_x,
            scaled_mole_y,
        )
        src_mole_patch_indices[mole_uuid] = _get_patch_index(
            scaled_src_w, scaled_mole_x, scaled_mole_y
        )

    src_canonical_uuids = set(src_mole_features.keys())

    # Process each target image
    for tgt_path in tgt_paths:
        if verbose:
            print(f"\nProcessing target: {tgt_path}")

        # Load target moles
        try:
            tgt_moles = mel.rotomap.moles.load_image_moles(
                tgt_path, extra_stem=extra_stem
            )
        except Exception as e:
            print(f"Error loading target moles from {tgt_path}: {e}")
            continue

        # Find UUIDs missing from target
        tgt_all_uuids = {m["uuid"] for m in tgt_moles}
        missing_uuids = src_canonical_uuids - tgt_all_uuids

        if not missing_uuids:
            if verbose:
                print(
                    "  No missing moles - all source canonical moles "
                    "already present in target"
                )
            continue

        if verbose:
            print(f"  Found {len(missing_uuids)} missing canonical moles to locate")

        # Get target features (from cache or compute)
        if tgt_path in tgt_cached:
            cached = tgt_cached[tgt_path]
            target_features = cached["features"]
            tgt_scale_x = cached["scale_x"]
            tgt_scale_y = cached["scale_y"]
            scaled_tgt_h = cached["scaled_h"]
            scaled_tgt_w = cached["scaled_w"]
            if verbose:
                print(
                    f"  Using cached target features: {target_features.shape[0]} "
                    f"patches, {_tensor_size_mb(target_features):.1f} MB"
                )
        else:
            # Load target image, scale it, and extract features once
            tgt_image_rgb = _load_image_with_mask(tgt_path, verbose)
            if verbose:
                print("  Scaling target image and extracting features...")
            scaled_tgt, (tgt_scale_x, tgt_scale_y) = mel.lib.dinov3.scale_image_to_fit(
                tgt_image_rgb, image_size
            )
            scaled_tgt_h, scaled_tgt_w = scaled_tgt.shape[:2]
            target_features = mel.lib.dinov3.extract_all_patch_features(
                scaled_tgt, model
            )
            if verbose:
                print(
                    f"  Target features: {target_features.shape[0]} patches, "
                    f"{_tensor_size_mb(target_features):.1f} MB"
                )

        matched_count = 0
        for missing_uuid in missing_uuids:
            if verbose:
                print(f"  Locating mole {missing_uuid}")

            final_x, final_y, scaled_tgt_x, scaled_tgt_y, final_sim = _find_mole_match(
                src_mole_features[missing_uuid],
                target_features,
                tgt_scale_x,
                tgt_scale_y,
                scaled_tgt_h,
                scaled_tgt_w,
                verbose,
            )

            if final_sim < min_similarity:
                if verbose:
                    print(
                        f"    Skipping: similarity {final_sim:.4f} "
                        f"below threshold {min_similarity}"
                    )
                continue

            # Bidirectional check: verify target patch matches back to source
            tgt_patch_idx = _get_patch_index(scaled_tgt_w, scaled_tgt_x, scaled_tgt_y)
            tgt_patch_feature = target_features[tgt_patch_idx]
            reverse_best_idx = _find_best_patch_index(
                tgt_patch_feature, src_all_features
            )
            src_mole_patch_idx = src_mole_patch_indices[missing_uuid]

            if reverse_best_idx != src_mole_patch_idx:
                if verbose:
                    print(
                        f"    Skipping: bidirectional check failed "
                        f"(reverse matched patch {reverse_best_idx}, "
                        f"expected {src_mole_patch_idx})"
                    )
                continue

            # Add the mole as non-canonical
            new_mole = {
                "uuid": missing_uuid,
                "x": final_x,
                "y": final_y,
                mel.rotomap.moles.KEY_IS_CONFIRMED: False,
            }
            tgt_moles.append(new_mole)
            matched_count += 1

            action = "Would add" if dry_run else "Added"
            print(
                f"  {action} mole {missing_uuid} at ({final_x}, {final_y}) "
                f"[similarity: {final_sim:.4f}]"
            )

        if matched_count > 0 and not dry_run:
            try:
                mel.rotomap.moles.save_image_moles(
                    tgt_moles, str(tgt_path), extra_stem=extra_stem
                )
                if verbose:
                    print(f"  Saved {matched_count} moles to {tgt_path}")
            except Exception as e:
                print(f"Error saving moles to {tgt_path}: {e}")
                return 1
        elif matched_count == 0 and verbose:
            print("  No moles matched above threshold")

    return 0


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
