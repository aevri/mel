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
    """Load an image and apply its mask if available."""
    print(f"Loading {label} image: {image_path}")
    image_rgb = cv2.cvtColor(mel.lib.image.load_image(image_path), cv2.COLOR_BGR2RGB)
    mask = mel.rotomap.mask.load_or_none(image_path)
    if mask is not None:
        print(f"Applying mask to {label} image")
        image_rgb = mel.lib.dinov3.apply_mask(image_rgb, mask)
    return image_rgb


def _run_matching_pass(
    src_image, src_mole_x, src_mole_y, target_image, model, similarity, image_size
):
    """Run a single matching pass and return native coords of best match."""
    scaled_src, (src_scale_x, src_scale_y) = mel.lib.dinov3.scale_image_to_fit(
        src_image, image_size
    )
    scaled_target, (tgt_scale_x, tgt_scale_y) = mel.lib.dinov3.scale_image_to_fit(
        target_image, image_size
    )
    print(f"Scaled source: {scaled_src.shape[1]}x{scaled_src.shape[0]}")
    print(f"Scaled target: {scaled_target.shape[1]}x{scaled_target.shape[0]}")

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
    print(
        f"Similarity range: {similarities.min().item():.4f} to "
        f"{similarities.max().item():.4f}"
    )

    scaled_tgt_h, scaled_tgt_w = scaled_target.shape[:2]
    best_x, best_y = mel.lib.dinov3.find_best_match_location(
        similarities, scaled_tgt_h, scaled_tgt_w, similarity
    )
    print(f"Best match at scaled coords: ({best_x}, {best_y})")

    # Convert to native coords
    native_x = int(best_x / tgt_scale_x)
    native_y = int(best_y / tgt_scale_y)

    return (
        native_x,
        native_y,
        scaled_src,
        scaled_mole_x,
        scaled_mole_y,
        scaled_target,
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
    """Render and save heatmap and source crosshair for a pass."""
    h, w = scaled_target.shape[:2]
    heatmap = mel.lib.dinov3.render_heatmap(
        scaled_target, similarities, h, w, similarity
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

    print(f"Loading DINOv3 model (size: {dino_size})...")
    try:
        model, feature_dim = mel.lib.dinov3.load_dinov3_model(
            dino_size, local_files_only=not allow_download
        )
        print(f"Model loaded with {feature_dim} feature dimensions")
    except RuntimeError as e:
        print(f"Error loading DINOv3 model: {e}")
        return 1

    src_image_rgb = _load_image_with_mask(src_path, "source")
    target_image_rgb = _load_image_with_mask(target_path, "target")

    src_max_dim = max(src_image_rgb.shape[:2])
    tgt_max_dim = max(target_image_rgb.shape[:2])
    smallest_max_dim = min(src_max_dim, tgt_max_dim)
    print(
        f"Source: {src_image_rgb.shape[1]}x{src_image_rgb.shape[0]}, "
        f"Target: {target_image_rgb.shape[1]}x{target_image_rgb.shape[0]}"
    )
    print(f"Smallest max dimension: {smallest_max_dim}")

    print("\n=== Two-pass approach ===\n")
    print("--- Pass 1: Global view ---")

    (
        pass1_native_x,
        pass1_native_y,
        scaled_src,
        scaled_mole_x,
        scaled_mole_y,
        scaled_target,
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

    print("\n--- Pass 2: Local view (native resolution for smallest) ---")

    native_coverage = image_size / smallest_max_dim
    print(f"Native coverage ratio: {native_coverage:.3f}")

    src_crop_size = max(int(src_max_dim * native_coverage), mel.lib.dinov3.PATCH_SIZE)
    tgt_crop_size = max(int(tgt_max_dim * native_coverage), mel.lib.dinov3.PATCH_SIZE)
    print(f"Source crop size: {src_crop_size}")
    print(f"Target crop size: {tgt_crop_size}")

    src_crop, (src_crop_off_x, src_crop_off_y) = mel.lib.dinov3.crop_to_region(
        src_image_rgb, src_mole_x, src_mole_y, src_crop_size
    )
    tgt_crop, (tgt_off_x, tgt_off_y) = mel.lib.dinov3.crop_to_region(
        target_image_rgb, pass1_native_x, pass1_native_y, tgt_crop_size
    )
    print(f"Target crop offset: ({tgt_off_x}, {tgt_off_y})")

    mole_in_crop_x = src_mole_x - src_crop_off_x
    mole_in_crop_y = src_mole_y - src_crop_off_y

    (
        pass2_crop_x,
        pass2_crop_y,
        scaled_src_crop,
        scaled_mole_crop_x,
        scaled_mole_crop_y,
        scaled_tgt_crop,
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

    print(
        f"\nFinal position in original image: "
        f"({tgt_off_x + pass2_crop_x}, {tgt_off_y + pass2_crop_y})"
    )
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
