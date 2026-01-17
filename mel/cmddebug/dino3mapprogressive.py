"""Generate progressive DINOv3 heatmaps with zoom refinement to locate a mole.

Given a source image with a marked mole (TARGET_UUID), this command generates
a series of heatmaps at increasing zoom levels, progressively refining the
location estimate until native resolution is reached.

Usage example:

    $ mel-debug dino3-map-progressive output.jpg target.jpg abc123 source.jpg

This extracts DINOv3 features from the mole with UUID 'abc123' in source.jpg,
then progressively zooms in on the best match location in target.jpg, outputting
heatmaps at each level (output_1.jpg, output_2.jpg, etc.) and source images
with crosshairs (output_source_1.jpg, output_source_2.jpg, etc.).
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


def setup_parser(parser):
    parser.add_argument(
        "OUTPUT_JPG",
        type=str,
        help="Base path for output images (will append _1, _2, etc.).",
    )
    parser.add_argument(
        "TARGET_JPG",
        type=_existing_file_path,
        help="Path to the target image where we want to locate the mole.",
    )
    parser.add_argument(
        "TARGET_UUID",
        type=str,
        help="UUID of the mole to locate (must exist in SRC_JPG's moles file).",
    )
    parser.add_argument(
        "SRC_JPG",
        type=_existing_file_path,
        help="Path to source image containing the marked mole.",
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
        "--zoom-factor",
        type=float,
        default=2.0,
        help="Factor to zoom by at each step (default: 2.0).",
    )
    parser.add_argument(
        "--similarity",
        type=str,
        choices=["cosine", "euclidean", "dot", "multi3x3", "softmax"],
        default="cosine",
        help="Similarity metric: cosine (default), euclidean, dot, "
        "multi3x3 (3x3 patch averaging), or softmax (temperature-scaled).",
    )


def _get_output_paths(output_base, level):
    """Generate output file paths for a given level."""
    # Remove .jpg extension if present to get base
    base = str(output_base)
    if base.lower().endswith(".jpg"):
        base = base[:-4]

    return f"{base}_{level}.jpg", f"{base}_{level}_source.jpg"


def process_args(args):
    output_base = args.OUTPUT_JPG
    target_path = args.TARGET_JPG
    target_uuid = args.TARGET_UUID
    src_path = args.SRC_JPG
    dino_size = args.dino_size
    image_size = args.image_size
    zoom_factor = args.zoom_factor
    similarity = args.similarity

    # Validate image_size is divisible by patch size
    if image_size % mel.lib.dinov3.PATCH_SIZE != 0:
        print(
            f"Error: --image-size must be divisible by "
            f"{mel.lib.dinov3.PATCH_SIZE}, got {image_size}"
        )
        return 1

    # Step 1: Load source moles and find TARGET_UUID
    print(f"Loading moles from: {src_path}")
    src_moles = mel.rotomap.moles.load_image_moles(src_path)
    mole_index = mel.rotomap.moles.uuid_mole_index(src_moles, target_uuid)

    if mole_index is None:
        print(f"Error: Mole UUID '{target_uuid}' not found in {src_path}")
        print(f"Available UUIDs: {[m['uuid'] for m in src_moles]}")
        return 1

    src_mole = src_moles[mole_index]
    src_mole_x, src_mole_y = src_mole["x"], src_mole["y"]
    print(f"Found mole '{target_uuid}' at ({src_mole_x}, {src_mole_y}) in source")

    # Step 2: Load DINOv3 model
    print(f"Loading DINOv3 model (size: {dino_size})...")
    try:
        model, feature_dim = mel.lib.dinov3.load_dinov3_model(dino_size)
        print(f"Model loaded with {feature_dim} feature dimensions")
    except RuntimeError as e:
        print(f"Error loading DINOv3 model: {e}")
        return 1

    # Step 3: Load source and target images at native resolution
    print(f"Loading source image: {src_path}")
    src_image = mel.lib.image.load_image(src_path)
    src_image_rgb = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)

    # Apply source mask if available
    src_mask = mel.rotomap.mask.load_or_none(src_path)
    if src_mask is not None:
        print("Applying mask to source image")
        src_image_rgb = mel.lib.dinov3.apply_mask(src_image_rgb, src_mask)

    print(f"Loading target image: {target_path}")
    target_image = mel.lib.image.load_image(target_path)
    target_image_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    # Apply target mask if available
    target_mask = mel.rotomap.mask.load_or_none(target_path)
    if target_mask is not None:
        print("Applying mask to target image")
        target_image_rgb = mel.lib.dinov3.apply_mask(target_image_rgb, target_mask)

    # Compute resolution ratio between source and target
    # (used to scale crop sizes so they cover the same physical area)
    src_full_h, src_full_w = src_image_rgb.shape[:2]
    tgt_full_h, tgt_full_w = target_image_rgb.shape[:2]
    resolution_ratio = max(src_full_h, src_full_w) / max(tgt_full_h, tgt_full_w)
    print(f"Resolution ratio (src/tgt): {resolution_ratio:.3f}")

    # Initialize regions for progressive zoom
    src_region = src_image_rgb
    target_region = target_image_rgb
    src_mole_local = (src_mole_x, src_mole_y)
    cumulative_offset = (0, 0)

    level = 1
    tgt_native_x, tgt_native_y = 0, 0

    print(f"\n=== Progressive zoom with factor {zoom_factor} ===\n")

    while True:
        print(f"--- Level {level} ---")
        target_h, target_w = target_region.shape[:2]
        src_h, src_w = src_region.shape[:2]
        print(f"Source region: {src_w}x{src_h}, Target region: {target_w}x{target_h}")

        # Scale both regions to image_size
        scaled_src, (src_scale_x, src_scale_y) = mel.lib.dinov3.scale_image_to_fit(
            src_region, image_size
        )
        scaled_target, (tgt_scale_x, tgt_scale_y) = mel.lib.dinov3.scale_image_to_fit(
            target_region, image_size
        )
        scaled_src_h, scaled_src_w = scaled_src.shape[:2]
        scaled_tgt_h, scaled_tgt_w = scaled_target.shape[:2]
        print(f"Scaled source: {scaled_src_w}x{scaled_src_h}")
        print(f"Scaled target: {scaled_tgt_w}x{scaled_tgt_h}")

        # Scale mole coords and extract source feature
        scaled_mole_x = int(src_mole_local[0] * src_scale_x)
        scaled_mole_y = int(src_mole_local[1] * src_scale_y)
        print(f"Scaled mole coords: ({scaled_mole_x}, {scaled_mole_y})")

        use_multi_patch = similarity == "multi3x3"
        mole_feature = mel.lib.dinov3.extract_mole_patch_feature(
            scaled_src, scaled_mole_x, scaled_mole_y, model, multi_patch=use_multi_patch
        )

        # Extract target features and compute similarities
        target_features = mel.lib.dinov3.extract_all_patch_features(scaled_target, model)
        sim_type = "cosine" if similarity == "multi3x3" else similarity
        similarities = mel.lib.dinov3.compute_similarities(
            mole_feature, target_features, similarity_type=sim_type
        )
        sim_min = similarities.min().item()
        sim_max = similarities.max().item()
        print(f"Similarity range: {sim_min:.4f} to {sim_max:.4f}")

        # Find best match location
        best_x, best_y = mel.lib.dinov3.find_best_match_location(
            similarities, scaled_tgt_h, scaled_tgt_w, similarity
        )
        print(f"Best match at scaled coords: ({best_x}, {best_y})")

        # Render and save target heatmap
        heatmap = mel.lib.dinov3.render_heatmap(
            scaled_target, similarities, scaled_tgt_h, scaled_tgt_w, similarity
        )
        output_path, src_output_path = _get_output_paths(output_base, level)
        mel.lib.image.save_image(heatmap, output_path)
        print(f"Saved target heatmap: {output_path}")

        # Render and save source image with crosshair
        src_with_cross = mel.lib.dinov3.render_crosshair(
            scaled_src, scaled_mole_x, scaled_mole_y
        )
        mel.lib.image.save_image(src_with_cross, src_output_path)
        print(f"Saved source with crosshair: {src_output_path}")

        # Check termination: can we zoom further?
        # Stop if region fits in image_size, or if crop wouldn't reduce region
        tgt_crop_size = int(image_size * zoom_factor)
        if max(target_h, target_w) <= image_size:
            print(f"\nReached native resolution at level {level}")
            # Convert final best match to target_region coords for final position
            tgt_native_x = int(best_x / tgt_scale_x)
            tgt_native_y = int(best_y / tgt_scale_y)
            break
        if tgt_crop_size >= max(target_h, target_w):
            print(f"\nReached maximum zoom at level {level} (crop wouldn't reduce region)")
            # Convert final best match to target_region coords for final position
            tgt_native_x = int(best_x / tgt_scale_x)
            tgt_native_y = int(best_y / tgt_scale_y)
            break

        # Convert best match to target_region coords
        tgt_native_x = int(best_x / tgt_scale_x)
        tgt_native_y = int(best_y / tgt_scale_y)
        print(f"Best match at native coords: ({tgt_native_x}, {tgt_native_y})")

        # Crop both images for next level
        # Scale source crop size by resolution ratio to cover same physical area
        src_crop_size = max(
            int(tgt_crop_size * resolution_ratio), mel.lib.dinov3.PATCH_SIZE
        )
        print(f"Cropping target to {tgt_crop_size}x{tgt_crop_size}")
        print(f"Cropping source to {src_crop_size}x{src_crop_size}")

        # Crop source around mole (using scaled crop size)
        src_region, (src_off_x, src_off_y) = mel.lib.dinov3.crop_to_region(
            src_region, src_mole_local[0], src_mole_local[1], src_crop_size
        )
        # Update mole position relative to new crop
        src_mole_local = (
            src_mole_local[0] - src_off_x,
            src_mole_local[1] - src_off_y,
        )

        # Crop target around best match
        target_region, (off_x, off_y) = mel.lib.dinov3.crop_to_region(
            target_region, tgt_native_x, tgt_native_y, tgt_crop_size
        )
        cumulative_offset = (cumulative_offset[0] + off_x, cumulative_offset[1] + off_y)

        level += 1
        print()

    # Report final position in original image coords
    final_x = cumulative_offset[0] + tgt_native_x
    final_y = cumulative_offset[1] + tgt_native_y
    print(f"\nFinal position in original image: ({final_x}, {final_y})")
    print("Done!")

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
