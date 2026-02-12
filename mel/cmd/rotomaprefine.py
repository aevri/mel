"""Refine mole positions in target images using local DINOv3 feature matching.

After automark3-nn places moles at coarse positions (snapped to DINOv3 patch
centers at scaled resolution), this command improves positions by doing local
feature matching at much higher effective resolution. Each mole's neighborhood
is cropped and scaled up, so each DINOv3 patch covers ~8 native pixels instead
of ~60, yielding approximately 8x better positional accuracy.
"""

import argparse
import pathlib

import cv2

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


def _validate_aspect_ratios(image_sizes):
    """Validate that all images have the same aspect ratio.

    Args:
        image_sizes: List of (path, height, width) tuples.

    Raises:
        ValueError: If aspect ratios differ by more than 1%.
    """
    if not image_sizes:
        return

    base_path, base_h, base_w = image_sizes[0]
    base_ratio = base_w / base_h

    for path, h, w in image_sizes[1:]:
        ratio = w / h
        if abs(ratio - base_ratio) / base_ratio > 0.01:
            raise ValueError(
                f"Aspect ratio mismatch: {base_path} is "
                f"{base_w}x{base_h} ({base_ratio:.4f}) but "
                f"{path} is {w}x{h} ({ratio:.4f})"
            )


def _normalize_resolution(images_with_paths):
    """Resize all images to the smallest resolution.

    Args:
        images_with_paths: List of (path, image_rgb) tuples.

    Returns:
        List of (path, resized_image, scale_factor) tuples, where scale_factor
        is the factor applied (e.g. 0.5 means image was halved).
    """
    if not images_with_paths:
        return []

    # Find smallest max dimension
    smallest_max_dim = min(
        max(img.shape[:2]) for _, img in images_with_paths
    )

    results = []
    for path, img in images_with_paths:
        h, w = img.shape[:2]
        max_dim = max(h, w)
        scale = smallest_max_dim / max_dim

        if abs(scale - 1.0) < 1e-6:
            results.append((path, img, 1.0))
        else:
            new_h = int(h * scale)
            new_w = int(w * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            results.append((path, resized, scale))

    return results


def setup_parser(parser):
    parser.add_argument(
        "--reference",
        "-r",
        type=_existing_file_path,
        nargs="+",
        required=True,
        help="Reference image(s) with canonical moles.",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=_existing_file_path,
        nargs="+",
        required=True,
        help="Target image(s) with non-canonical moles to refine.",
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
        default=896,
        help="Scale crops to fit this size in pixels (default: 896). "
        "Must be divisible by 16.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=448,
        help="Size of local crop around target mole in normalized pixels "
        "(default: 448).",
    )
    parser.add_argument(
        "--ref-crop-size",
        type=int,
        default=None,
        help="Size of local crop around reference mole in normalized pixels "
        "(default: half of --crop-size).",
    )
    parser.add_argument(
        "--similarity",
        type=str,
        choices=["cosine", "euclidean", "dot", "multi3x3", "softmax"],
        default="cosine",
        help="Similarity metric (default: cosine).",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="Minimum similarity threshold for matches (default: 0.0).",
    )
    parser.add_argument(
        "--extra-stem",
        help="Load/save alternate mole file (e.g., '0.jpg.EXTRA.json').",
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
        "--dry-run",
        action="store_true",
        help="Show what would be done without saving.",
    )


def process_args(args):
    ref_paths = args.reference
    tgt_paths = args.target
    dino_size = args.dino_size
    image_size = args.image_size
    crop_size = args.crop_size
    ref_crop_size = args.ref_crop_size if args.ref_crop_size is not None else crop_size // 2
    similarity = args.similarity
    min_similarity = args.min_similarity
    extra_stem = args.extra_stem
    allow_download = args.allow_download
    verbose = args.verbose
    dry_run = args.dry_run

    use_multi_patch = similarity == "multi3x3"
    sim_type = "cosine" if similarity == "multi3x3" else similarity

    # Validate image_size is divisible by patch size
    if image_size % mel.lib.dinov3.PATCH_SIZE != 0:
        print(
            f"Error: --image-size must be divisible by "
            f"{mel.lib.dinov3.PATCH_SIZE}, got {image_size}"
        )
        return 1

    # Load all images
    all_images = []
    ref_images = {}
    for ref_path in ref_paths:
        img = _load_image_with_mask(ref_path, verbose)
        ref_images[ref_path] = img
        all_images.append((ref_path, img))

    tgt_images = {}
    for tgt_path in tgt_paths:
        img = _load_image_with_mask(tgt_path, verbose)
        tgt_images[tgt_path] = img
        all_images.append((tgt_path, img))

    # Print image size summary
    if verbose:
        print("\nImage sizes:")
        for path, img in all_images:
            h, w = img.shape[:2]
            print(f"  {path}: {w}x{h}")

    # Validate aspect ratios
    image_sizes = [(p, img.shape[0], img.shape[1]) for p, img in all_images]
    try:
        _validate_aspect_ratios(image_sizes)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Normalize resolution
    normalized = _normalize_resolution(all_images)

    norm_ref_images = {}
    norm_tgt_images = {}
    scale_factors = {}
    for path, img, scale in normalized:
        scale_factors[path] = scale
        if path in ref_images:
            norm_ref_images[path] = img
        if path in tgt_images:
            norm_tgt_images[path] = img

    if verbose:
        smallest_h, smallest_w = normalized[0][1].shape[:2]
        for _, img, _ in normalized:
            h, w = img.shape[:2]
            if max(h, w) < max(smallest_h, smallest_w):
                smallest_h, smallest_w = h, w
        print(f"\nNormalized resolution: {smallest_w}x{smallest_h}")
        for path, _, scale in normalized:
            if abs(scale - 1.0) > 1e-6:
                print(f"  {path}: scaled by {scale:.4f}")

    # Load moles from references (build uuid -> (ref_path, mole) lookup)
    ref_canonical_by_uuid = {}
    for ref_path in ref_paths:
        try:
            moles = mel.rotomap.moles.load_image_moles(ref_path)
        except Exception as e:
            print(f"Error loading moles from {ref_path}: {e}")
            return 1
        for m in moles:
            if m[mel.rotomap.moles.KEY_IS_CONFIRMED]:
                ref_canonical_by_uuid[m["uuid"]] = (ref_path, m)

    if not ref_canonical_by_uuid:
        print("Error: No canonical moles found in any reference image")
        return 1

    if verbose:
        print(
            f"\nFound {len(ref_canonical_by_uuid)} canonical moles "
            f"across {len(ref_paths)} reference images"
        )

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

    # Pre-compute reference features for each canonical mole
    if verbose:
        print(f"\nPre-computing reference features "
              f"(ref-crop-size={ref_crop_size})...")
    ref_features_by_uuid = {}
    for mole_uuid, (ref_path, ref_mole) in ref_canonical_by_uuid.items():
        ref_scale = scale_factors[ref_path]
        norm_ref_img = norm_ref_images[ref_path]

        norm_ref_x = int(ref_mole["x"] * ref_scale)
        norm_ref_y = int(ref_mole["y"] * ref_scale)

        ref_crop, (ref_off_x, ref_off_y) = mel.lib.dinov3.crop_to_region(
            norm_ref_img, norm_ref_x, norm_ref_y, ref_crop_size
        )

        mole_in_crop_x = norm_ref_x - ref_off_x
        mole_in_crop_y = norm_ref_y - ref_off_y

        scaled_ref_crop, (ref_crop_sx, ref_crop_sy) = (
            mel.lib.dinov3.scale_image_to_fit(ref_crop, image_size)
        )

        scaled_mole_x = int(mole_in_crop_x * ref_crop_sx)
        scaled_mole_y = int(mole_in_crop_y * ref_crop_sy)

        mole_feature = mel.lib.dinov3.extract_mole_patch_feature(
            scaled_ref_crop,
            scaled_mole_x,
            scaled_mole_y,
            model,
            multi_patch=use_multi_patch,
        )
        ref_features_by_uuid[mole_uuid] = mole_feature

    if verbose:
        print(f"  Cached {len(ref_features_by_uuid)} reference features")

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

        # Find non-canonical moles that have matching canonical UUIDs
        non_canonical = [
            m for m in tgt_moles
            if not m[mel.rotomap.moles.KEY_IS_CONFIRMED]
            and m["uuid"] in ref_features_by_uuid
        ]

        if not non_canonical:
            if verbose:
                print("  No non-canonical moles with matching references to refine")
            continue

        if verbose:
            print(f"  Found {len(non_canonical)} non-canonical moles to refine")

        tgt_scale = scale_factors[tgt_path]
        norm_tgt_img = norm_tgt_images[tgt_path]

        refined_count = 0
        for mole in non_canonical:
            mole_uuid = mole["uuid"]
            mole_feature = ref_features_by_uuid[mole_uuid]

            norm_tgt_x = int(mole["x"] * tgt_scale)
            norm_tgt_y = int(mole["y"] * tgt_scale)

            if verbose:
                print(f"  Refining mole {mole_uuid}")
                print(
                    f"    Target pos (normalized): ({norm_tgt_x}, {norm_tgt_y})"
                )

            # Crop around non-canonical mole in target
            tgt_crop, (tgt_off_x, tgt_off_y) = mel.lib.dinov3.crop_to_region(
                norm_tgt_img, norm_tgt_x, norm_tgt_y, crop_size
            )

            # Scale crop to image_size for DINOv3
            scaled_tgt_crop, (tgt_crop_sx, tgt_crop_sy) = (
                mel.lib.dinov3.scale_image_to_fit(tgt_crop, image_size)
            )

            # Extract all features from target crop
            target_features = mel.lib.dinov3.extract_all_patch_features(
                scaled_tgt_crop, model
            )

            # Compute similarities and find best match
            similarities = mel.lib.dinov3.compute_similarities(
                mole_feature, target_features, similarity_type=sim_type
            )

            max_sim = similarities.max().item()
            if verbose:
                print(
                    f"    Similarity range: "
                    f"{similarities.min().item():.4f} to {max_sim:.4f}"
                )

            if max_sim < min_similarity:
                if verbose:
                    print(
                        f"    Skipping: similarity {max_sim:.4f} "
                        f"below threshold {min_similarity}"
                    )
                continue

            scaled_tgt_crop_h, scaled_tgt_crop_w = scaled_tgt_crop.shape[:2]
            match_x, match_y = mel.lib.dinov3.find_best_match_location(
                similarities,
                scaled_tgt_crop_h,
                scaled_tgt_crop_w,
                similarity,
            )

            # Convert back: scaled crop -> normalized crop -> normalized -> native
            crop_native_x = int(match_x / tgt_crop_sx)
            crop_native_y = int(match_y / tgt_crop_sy)
            norm_new_x = tgt_off_x + crop_native_x
            norm_new_y = tgt_off_y + crop_native_y
            native_new_x = int(norm_new_x / tgt_scale)
            native_new_y = int(norm_new_y / tgt_scale)

            # Check if position changed meaningfully (> 1px in native coords)
            dx = abs(native_new_x - mole["x"])
            dy = abs(native_new_y - mole["y"])
            if dx <= 1 and dy <= 1:
                if verbose:
                    print("    Position unchanged (within 1px)")
                continue

            action = "Would update" if dry_run else "Updated"
            print(
                f"  {action} mole {mole_uuid}: "
                f"({mole['x']}, {mole['y']}) -> ({native_new_x}, {native_new_y}) "
                f"[d={dx},{dy} sim={max_sim:.4f}]"
            )

            if not dry_run:
                mole["x"] = native_new_x
                mole["y"] = native_new_y
            refined_count += 1

        if refined_count > 0 and not dry_run:
            try:
                mel.rotomap.moles.save_image_moles(
                    tgt_moles, str(tgt_path), extra_stem=extra_stem
                )
                if verbose:
                    print(f"  Saved {refined_count} refined moles to {tgt_path}")
            except Exception as e:
                print(f"Error saving moles to {tgt_path}: {e}")
                return 1
        elif refined_count == 0 and verbose:
            print("  No moles refined")

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
