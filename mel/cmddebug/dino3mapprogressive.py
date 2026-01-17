"""Generate two-pass DINOv3 heatmaps to locate a mole.

Given a source image with a marked mole (TARGET_UUID), this command generates
two heatmaps: a global view (full images scaled down) and a local view (native
resolution for the smallest image, with proportional coverage on both).

Usage example:

    $ mel-debug dino3-map-progressive output.jpg target.jpg abc123 source.jpg

This extracts DINOv3 features from the mole with UUID 'abc123' in source.jpg,
then finds the best match in target.jpg using a two-pass approach:
  - Pass 1: Global view with both images scaled to fit --image-size
  - Pass 2: Local view at native resolution for the smallest image
Outputs heatmaps (output_1.jpg, output_2.jpg) and source images with crosshairs
(output_1_source.jpg, output_2_source.jpg).
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


def _load_image_with_mask(image_path, label):
    """Load an image and apply its mask if available.

    Args:
        image_path: Path to the image file.
        label: Label for logging (e.g., "source", "target").

    Returns:
        RGB image with mask applied if available.
    """
    print(f"Loading {label} image: {image_path}")
    image = mel.lib.image.load_image(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = mel.rotomap.mask.load_or_none(image_path)
    if mask is not None:
        print(f"Applying mask to {label} image")
        image_rgb = mel.lib.dinov3.apply_mask(image_rgb, mask)

    return image_rgb


def _run_matching_pass(
    src_image,
    src_mole_x,
    src_mole_y,
    target_image,
    model,
    similarity,
    image_size,
):
    """Run a single matching pass: extract features, compute similarities, find match.

    Args:
        src_image: Source image (RGB).
        src_mole_x: Mole X coordinate in src_image.
        src_mole_y: Mole Y coordinate in src_image.
        target_image: Target image (RGB).
        model: DINOv3 model.
        similarity: Similarity metric name.
        image_size: Size to scale images to.

    Returns:
        Tuple of (best_x, best_y, scaled_src, scaled_src_mole_x, scaled_src_mole_y,
                  scaled_target, target_scale_x, target_scale_y, similarities).
    """
    scaled_src, (src_scale_x, src_scale_y) = mel.lib.dinov3.scale_image_to_fit(
        src_image, image_size
    )
    scaled_target, (tgt_scale_x, tgt_scale_y) = mel.lib.dinov3.scale_image_to_fit(
        target_image, image_size
    )
    scaled_src_h, scaled_src_w = scaled_src.shape[:2]
    scaled_tgt_h, scaled_tgt_w = scaled_target.shape[:2]
    print(f"Scaled source: {scaled_src_w}x{scaled_src_h}")
    print(f"Scaled target: {scaled_tgt_w}x{scaled_tgt_h}")

    scaled_mole_x = int(src_mole_x * src_scale_x)
    scaled_mole_y = int(src_mole_y * src_scale_y)
    print(f"Scaled mole coords: ({scaled_mole_x}, {scaled_mole_y})")

    use_multi_patch = similarity == "multi3x3"
    mole_feature = mel.lib.dinov3.extract_mole_patch_feature(
        scaled_src, scaled_mole_x, scaled_mole_y, model, multi_patch=use_multi_patch
    )

    target_features = mel.lib.dinov3.extract_all_patch_features(scaled_target, model)
    sim_type = "cosine" if similarity == "multi3x3" else similarity
    similarities = mel.lib.dinov3.compute_similarities(
        mole_feature, target_features, similarity_type=sim_type
    )
    sim_min = similarities.min().item()
    sim_max = similarities.max().item()
    print(f"Similarity range: {sim_min:.4f} to {sim_max:.4f}")

    best_x, best_y = mel.lib.dinov3.find_best_match_location(
        similarities, scaled_tgt_h, scaled_tgt_w, similarity
    )
    print(f"Best match at scaled coords: ({best_x}, {best_y})")

    return (
        best_x,
        best_y,
        scaled_src,
        scaled_mole_x,
        scaled_mole_y,
        scaled_target,
        tgt_scale_x,
        tgt_scale_y,
        similarities,
    )


def _save_pass_outputs(
    output_base,
    level,
    scaled_target,
    similarities,
    similarity,
    scaled_src,
    scaled_mole_x,
    scaled_mole_y,
):
    """Render and save heatmap and source crosshair for a pass.

    Args:
        output_base: Base path for output files.
        level: Pass level (1 or 2).
        scaled_target: Scaled target image.
        similarities: Similarity tensor.
        similarity: Similarity metric name.
        scaled_src: Scaled source image.
        scaled_mole_x: Mole X in scaled source.
        scaled_mole_y: Mole Y in scaled source.
    """
    scaled_tgt_h, scaled_tgt_w = scaled_target.shape[:2]

    heatmap = mel.lib.dinov3.render_heatmap(
        scaled_target, similarities, scaled_tgt_h, scaled_tgt_w, similarity
    )
    output_path, src_output_path = _get_output_paths(output_base, level)
    mel.lib.image.save_image(heatmap, output_path)
    print(f"Saved target heatmap: {output_path}")

    src_with_cross = mel.lib.dinov3.render_crosshair(
        scaled_src, scaled_mole_x, scaled_mole_y
    )
    mel.lib.image.save_image(src_with_cross, src_output_path)
    print(f"Saved source with crosshair: {src_output_path}")


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
    similarity = args.similarity
    allow_download = args.allow_download

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
        model, feature_dim = mel.lib.dinov3.load_dinov3_model(
            dino_size, local_files_only=not allow_download
        )
        print(f"Model loaded with {feature_dim} feature dimensions")
    except RuntimeError as e:
        print(f"Error loading DINOv3 model: {e}")
        return 1

    # Step 3: Load source and target images at native resolution
    src_image_rgb = _load_image_with_mask(src_path, "source")
    target_image_rgb = _load_image_with_mask(target_path, "target")

    # Compute dimensions
    src_full_h, src_full_w = src_image_rgb.shape[:2]
    tgt_full_h, tgt_full_w = target_image_rgb.shape[:2]
    src_max_dim = max(src_full_h, src_full_w)
    tgt_max_dim = max(tgt_full_h, tgt_full_w)
    smallest_max_dim = min(src_max_dim, tgt_max_dim)
    print(f"Source: {src_full_w}x{src_full_h}, Target: {tgt_full_w}x{tgt_full_h}")
    print(f"Smallest max dimension: {smallest_max_dim}")

    print("\n=== Two-pass approach ===\n")

    # =========================================================================
    # PASS 1: Global view - scale full images to fit image_size
    # =========================================================================
    print("--- Pass 1: Global view ---")

    (
        best_x,
        best_y,
        scaled_src,
        scaled_mole_x,
        scaled_mole_y,
        scaled_target,
        tgt_scale_x,
        tgt_scale_y,
        similarities,
    ) = _run_matching_pass(
        src_image_rgb,
        src_mole_x,
        src_mole_y,
        target_image_rgb,
        model,
        similarity,
        image_size,
    )

    # Convert to native target coords
    pass1_native_x = int(best_x / tgt_scale_x)
    pass1_native_y = int(best_y / tgt_scale_y)
    print(f"Best match at native coords: ({pass1_native_x}, {pass1_native_y})")

    _save_pass_outputs(
        output_base,
        1,
        scaled_target,
        similarities,
        similarity,
        scaled_src,
        scaled_mole_x,
        scaled_mole_y,
    )

    # =========================================================================
    # PASS 2: Local view - native resolution for smallest image
    # =========================================================================
    print("\n--- Pass 2: Local view (native resolution for smallest) ---")

    # Calculate coverage ratio - what fraction of smallest image fits at native res
    native_coverage = image_size / smallest_max_dim
    print(f"Native coverage ratio: {native_coverage:.3f}")

    # Apply same coverage to both images (preserves zoom ratio)
    src_crop_size = max(int(src_max_dim * native_coverage), mel.lib.dinov3.PATCH_SIZE)
    tgt_crop_size = max(int(tgt_max_dim * native_coverage), mel.lib.dinov3.PATCH_SIZE)
    print(f"Source crop size: {src_crop_size}")
    print(f"Target crop size: {tgt_crop_size}")

    # Crop source around mole
    src_crop, (src_crop_off_x, src_crop_off_y) = mel.lib.dinov3.crop_to_region(
        src_image_rgb, src_mole_x, src_mole_y, src_crop_size
    )
    # Crop target around pass 1 best match
    tgt_crop, (tgt_off_x, tgt_off_y) = mel.lib.dinov3.crop_to_region(
        target_image_rgb, pass1_native_x, pass1_native_y, tgt_crop_size
    )
    print(f"Target crop offset: ({tgt_off_x}, {tgt_off_y})")

    # Compute mole position in cropped source
    mole_in_crop_x = src_mole_x - src_crop_off_x
    mole_in_crop_y = src_mole_y - src_crop_off_y

    (
        best_x_2,
        best_y_2,
        scaled_src_crop,
        scaled_mole_crop_x,
        scaled_mole_crop_y,
        scaled_tgt_crop,
        tgt_crop_scale_x,
        tgt_crop_scale_y,
        similarities_2,
    ) = _run_matching_pass(
        src_crop,
        mole_in_crop_x,
        mole_in_crop_y,
        tgt_crop,
        model,
        similarity,
        image_size,
    )

    # Convert to native target coords within crop
    pass2_crop_x = int(best_x_2 / tgt_crop_scale_x)
    pass2_crop_y = int(best_y_2 / tgt_crop_scale_y)
    print(f"Best match in crop native coords: ({pass2_crop_x}, {pass2_crop_y})")

    _save_pass_outputs(
        output_base,
        2,
        scaled_tgt_crop,
        similarities_2,
        similarity,
        scaled_src_crop,
        scaled_mole_crop_x,
        scaled_mole_crop_y,
    )

    # Report final position in original image coords
    final_x = tgt_off_x + pass2_crop_x
    final_y = tgt_off_y + pass2_crop_y
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
