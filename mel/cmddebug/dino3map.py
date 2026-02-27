"""Generate DINOv3 cosine similarity heatmap to locate a mole.

Given a source image with a marked mole (TARGET_UUID), this command generates
a heatmap showing where that mole likely appears in the target image.

Usage example:

    $ mel-debug dino3-map output.jpg target.jpg abc123 source.jpg

This extracts DINOv3 features from the mole with UUID 'abc123' in source.jpg,
then computes cosine similarity across target.jpg, outputting a heatmap to
output.jpg showing where the mole likely is.
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
        help="Path for the output heatmap image.",
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


def process_args(args):
    output_path = args.OUTPUT_JPG
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

    # Step 3: Load and scale source image, extract mole feature
    print(f"Loading source image: {src_path}")
    src_image = mel.lib.image.load_image(src_path)
    src_image_rgb = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)

    # Apply mask if available
    src_mask = mel.rotomap.mask.load_or_none(src_path)
    if src_mask is not None:
        print("Applying mask to source image")
        src_image_rgb = mel.lib.dinov3.apply_mask(src_image_rgb, src_mask)

    print(f"Scaling source image to fit {image_size}px...")
    scaled_src, (scale_x, scale_y) = mel.lib.dinov3.scale_image_to_fit(
        src_image_rgb, image_size
    )
    scaled_src_h, scaled_src_w = scaled_src.shape[:2]
    print(f"Scaled source: {scaled_src_w}x{scaled_src_h}")

    # Adjust mole coordinates by scale factor
    scaled_mole_x = int(src_mole_x * scale_x)
    scaled_mole_y = int(src_mole_y * scale_y)
    print(f"Scaled mole coords: ({scaled_mole_x}, {scaled_mole_y})")

    # Step 4: Extract mole patch feature from source
    use_multi_patch = similarity == "multi3x3"
    if use_multi_patch:
        print("Extracting mole 3x3 patch features from source...")
    else:
        print("Extracting mole patch feature from source...")
    mole_feature = mel.lib.dinov3.extract_mole_patch_feature(
        scaled_src, scaled_mole_x, scaled_mole_y, model, multi_patch=use_multi_patch
    )
    print(f"Mole feature shape: {mole_feature.shape}")

    # Step 5: Load and scale target image
    print(f"Loading target image: {target_path}")
    target_image = mel.lib.image.load_image(target_path)
    target_image_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    # Apply mask if available
    target_mask = mel.rotomap.mask.load_or_none(target_path)
    if target_mask is not None:
        print("Applying mask to target image")
        target_image_rgb = mel.lib.dinov3.apply_mask(target_image_rgb, target_mask)

    print(f"Scaling target image to fit {image_size}px...")
    scaled_target, _ = mel.lib.dinov3.scale_image_to_fit(target_image_rgb, image_size)
    scaled_target_h, scaled_target_w = scaled_target.shape[:2]
    print(f"Scaled target: {scaled_target_w}x{scaled_target_h}")

    # Step 6: Extract all patch features from target
    print("Extracting all patch features from target...")
    target_features = mel.lib.dinov3.extract_all_patch_features(scaled_target, model)
    print(f"Target features shape: {target_features.shape}")

    # Step 7: Compute similarities
    # For multi3x3, use cosine similarity (the multi-patch averaging was done above)
    sim_type = "cosine" if similarity == "multi3x3" else similarity
    print(f"Computing {similarity} similarities...")
    similarities = mel.lib.dinov3.compute_similarities(
        mole_feature, target_features, similarity_type=sim_type
    )
    sim_min = similarities.min().item()
    sim_max = similarities.max().item()
    print(f"Similarity range: {sim_min:.4f} to {sim_max:.4f}")

    # Step 8: Render heatmap and save
    print("Rendering heatmap...")
    heatmap = mel.lib.dinov3.render_heatmap(
        scaled_target, similarities, scaled_target_h, scaled_target_w, similarity
    )

    print(f"Saving output to: {output_path}")
    mel.lib.image.save_image(heatmap, output_path)
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
