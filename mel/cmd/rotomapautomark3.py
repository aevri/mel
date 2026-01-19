"""Locate canonical moles from a source image in target images using DINOv3.

Uses a two-pass approach:
  - Pass 1: Global view with images scaled to fit --image-size
  - Pass 2: Local view at native resolution around the Pass 1 match

Moles found in target images are added as non-canonical (is_uuid_canonical=False).
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


def _run_two_pass_matching(
    src_image,
    src_mole_x,
    src_mole_y,
    target_image,
    model,
    similarity,
    image_size,
    verbose=False,
):
    """Run two-pass DINOv3 matching to locate a mole.

    Returns:
        Tuple of (final_x, final_y, pass1_similarity, pass2_similarity) where
        final_x, final_y are native coordinates in the target image.
    """
    use_multi_patch = similarity == "multi3x3"
    sim_type = "cosine" if similarity == "multi3x3" else similarity

    # Pass 1: Global view
    if verbose:
        print("  Pass 1: Global view")

    scaled_src, (src_scale_x, src_scale_y) = mel.lib.dinov3.scale_image_to_fit(
        src_image, image_size
    )
    scaled_target, (tgt_scale_x, tgt_scale_y) = mel.lib.dinov3.scale_image_to_fit(
        target_image, image_size
    )

    scaled_mole_x = int(src_mole_x * src_scale_x)
    scaled_mole_y = int(src_mole_y * src_scale_y)

    mole_feature = mel.lib.dinov3.extract_mole_patch_feature(
        scaled_src, scaled_mole_x, scaled_mole_y, model, multi_patch=use_multi_patch
    )

    target_features = mel.lib.dinov3.extract_all_patch_features(scaled_target, model)
    similarities = mel.lib.dinov3.compute_similarities(
        mole_feature, target_features, similarity_type=sim_type
    )

    pass1_max_sim = similarities.max().item()
    if verbose:
        print(
            f"    Similarity range: {similarities.min().item():.4f} to {pass1_max_sim:.4f}"
        )

    scaled_tgt_h, scaled_tgt_w = scaled_target.shape[:2]
    pass1_x, pass1_y = mel.lib.dinov3.find_best_match_location(
        similarities, scaled_tgt_h, scaled_tgt_w, similarity
    )

    # Convert to native coords
    pass1_native_x = int(pass1_x / tgt_scale_x)
    pass1_native_y = int(pass1_y / tgt_scale_y)

    if verbose:
        print(f"    Best match at native coords: ({pass1_native_x}, {pass1_native_y})")

    # Pass 2: Local view at native resolution for smallest image
    if verbose:
        print("  Pass 2: Local view")

    src_max_dim = max(src_image.shape[:2])
    tgt_max_dim = max(target_image.shape[:2])
    smallest_max_dim = min(src_max_dim, tgt_max_dim)

    native_coverage = image_size / smallest_max_dim
    src_crop_size = max(int(src_max_dim * native_coverage), mel.lib.dinov3.PATCH_SIZE)
    tgt_crop_size = max(int(tgt_max_dim * native_coverage), mel.lib.dinov3.PATCH_SIZE)

    src_crop, (src_crop_off_x, src_crop_off_y) = mel.lib.dinov3.crop_to_region(
        src_image, src_mole_x, src_mole_y, src_crop_size
    )
    tgt_crop, (tgt_off_x, tgt_off_y) = mel.lib.dinov3.crop_to_region(
        target_image, pass1_native_x, pass1_native_y, tgt_crop_size
    )

    mole_in_crop_x = src_mole_x - src_crop_off_x
    mole_in_crop_y = src_mole_y - src_crop_off_y

    scaled_src_crop, (src_crop_scale_x, src_crop_scale_y) = (
        mel.lib.dinov3.scale_image_to_fit(src_crop, image_size)
    )
    scaled_tgt_crop, (tgt_crop_scale_x, tgt_crop_scale_y) = (
        mel.lib.dinov3.scale_image_to_fit(tgt_crop, image_size)
    )

    scaled_mole_crop_x = int(mole_in_crop_x * src_crop_scale_x)
    scaled_mole_crop_y = int(mole_in_crop_y * src_crop_scale_y)

    mole_feature_2 = mel.lib.dinov3.extract_mole_patch_feature(
        scaled_src_crop,
        scaled_mole_crop_x,
        scaled_mole_crop_y,
        model,
        multi_patch=use_multi_patch,
    )

    target_features_2 = mel.lib.dinov3.extract_all_patch_features(
        scaled_tgt_crop, model
    )
    similarities_2 = mel.lib.dinov3.compute_similarities(
        mole_feature_2, target_features_2, similarity_type=sim_type
    )

    pass2_max_sim = similarities_2.max().item()
    if verbose:
        print(
            f"    Similarity range: {similarities_2.min().item():.4f} to {pass2_max_sim:.4f}"
        )

    scaled_tgt_crop_h, scaled_tgt_crop_w = scaled_tgt_crop.shape[:2]
    pass2_x, pass2_y = mel.lib.dinov3.find_best_match_location(
        similarities_2, scaled_tgt_crop_h, scaled_tgt_crop_w, similarity
    )

    # Convert crop coords back to native target coords
    pass2_crop_native_x = int(pass2_x / tgt_crop_scale_x)
    pass2_crop_native_y = int(pass2_y / tgt_crop_scale_y)

    final_x = tgt_off_x + pass2_crop_native_x
    final_y = tgt_off_y + pass2_crop_native_y

    if verbose:
        print(f"    Final position: ({final_x}, {final_y})")

    return final_x, final_y, pass1_max_sim, pass2_max_sim


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
        default=896,
        help="Scale images to fit this size in pixels (default: 896). "
        "Must be divisible by 16.",
    )
    parser.add_argument(
        "--similarity",
        type=str,
        choices=["cosine", "euclidean", "dot", "multi3x3", "softmax"],
        default="cosine",
        help="Similarity metric: cosine (default), euclidean, dot, "
        "multi3x3 (3x3 patch averaging), or softmax (temperature-scaled).",
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
    similarity = args.similarity
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

    # Load source image once
    src_image_rgb = _load_image_with_mask(src_path, verbose)

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
        src_canonical_uuids = {m["uuid"] for m in src_canonical_moles}
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

        # Load target image
        tgt_image_rgb = _load_image_with_mask(tgt_path, verbose)

        # Build UUID to mole lookup for source
        src_uuid_to_mole = {m["uuid"]: m for m in src_canonical_moles}

        matched_count = 0
        for missing_uuid in missing_uuids:
            src_mole = src_uuid_to_mole[missing_uuid]
            src_mole_x, src_mole_y = src_mole["x"], src_mole["y"]

            if verbose:
                print(f"  Locating mole {missing_uuid}")

            final_x, final_y, _, pass2_sim = _run_two_pass_matching(
                src_image_rgb,
                src_mole_x,
                src_mole_y,
                tgt_image_rgb,
                model,
                similarity,
                image_size,
                verbose,
            )

            # Use pass2 similarity as the final similarity score
            final_sim = pass2_sim

            if final_sim < min_similarity:
                if verbose:
                    print(
                        f"    Skipping: similarity {final_sim:.4f} "
                        f"below threshold {min_similarity}"
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
