"""Refine mole positions using template matching (normalized cross-correlation).

Crops a template around each canonical mole in the reference image, then
slides it over a local search region in the target image to find the best
match. This is a classical CV approach that gives sub-pixel precision
without requiring a neural network.
"""

import argparse
import pathlib

import cv2
import numpy as np

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


def _load_image_with_mask(image_path):
    """Load an image and apply its mask if available.

    Returns the image in BGR format (for OpenCV template matching).
    """
    image_bgr = mel.lib.image.load_image(image_path)
    mask = mel.rotomap.mask.load_or_none(image_path)
    if mask is not None:
        # Set masked-out pixels to mean gray to avoid false matches
        keep_mask = mask > 127
        mean_val = 128
        for c in range(3):
            image_bgr[:, :, c] = np.where(
                keep_mask, image_bgr[:, :, c], mean_val
            )
    return image_bgr


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
        images_with_paths: List of (path, image) tuples.

    Returns:
        List of (path, resized_image, scale_factor) tuples, where scale_factor
        is the factor applied (e.g. 0.5 means image was halved).
    """
    if not images_with_paths:
        return []

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
        "--template-size",
        type=int,
        default=31,
        help="Size of template crop around reference mole in normalized "
        "pixels (default: 31). Should be odd.",
    )
    parser.add_argument(
        "--search-size",
        type=int,
        default=121,
        help="Size of search region around target mole in normalized "
        "pixels (default: 121). Should be odd and larger than template-size.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum normalized cross-correlation score (default: 0.0). "
        "Range is -1.0 to 1.0.",
    )
    parser.add_argument(
        "--extra-stem",
        help="Load/save alternate mole file (e.g., '0.jpg.EXTRA.json').",
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
    template_size = args.template_size
    search_size = args.search_size
    min_score = args.min_score
    extra_stem = args.extra_stem
    verbose = args.verbose
    dry_run = args.dry_run

    if search_size <= template_size:
        print(
            f"Error: --search-size ({search_size}) must be larger "
            f"than --template-size ({template_size})"
        )
        return 1

    # Load all images
    all_images = []
    ref_images = {}
    for ref_path in ref_paths:
        img = _load_image_with_mask(ref_path)
        ref_images[ref_path] = img
        all_images.append((ref_path, img))

    tgt_images = {}
    for tgt_path in tgt_paths:
        img = _load_image_with_mask(tgt_path)
        tgt_images[tgt_path] = img
        all_images.append((tgt_path, img))

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

    # Pre-extract templates for each canonical mole
    half_tmpl = template_size // 2
    templates_by_uuid = {}
    for mole_uuid, (ref_path, ref_mole) in ref_canonical_by_uuid.items():
        ref_scale = scale_factors[ref_path]
        norm_ref_img = norm_ref_images[ref_path]

        norm_ref_x = int(ref_mole["x"] * ref_scale)
        norm_ref_y = int(ref_mole["y"] * ref_scale)

        h, w = norm_ref_img.shape[:2]
        y1 = max(0, norm_ref_y - half_tmpl)
        y2 = min(h, norm_ref_y + half_tmpl + 1)
        x1 = max(0, norm_ref_x - half_tmpl)
        x2 = min(w, norm_ref_x + half_tmpl + 1)

        template = norm_ref_img[y1:y2, x1:x2].copy()

        # Skip templates that are too small (near image edges)
        if template.shape[0] < template_size // 2 or template.shape[1] < template_size // 2:
            continue

        templates_by_uuid[mole_uuid] = template

    if verbose:
        print(
            f"Loaded {len(ref_paths)} ref, {len(tgt_paths)} target images; "
            f"{len(templates_by_uuid)} templates"
        )

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

        # Find non-canonical moles that have matching templates
        non_canonical = [
            m for m in tgt_moles
            if not m[mel.rotomap.moles.KEY_IS_CONFIRMED]
            and m["uuid"] in templates_by_uuid
        ]

        if not non_canonical:
            if verbose:
                print("  No non-canonical moles with matching references to refine")
            continue

        if verbose:
            print(f"  {len(non_canonical)} moles to refine")

        tgt_scale = scale_factors[tgt_path]
        norm_tgt_img = norm_tgt_images[tgt_path]
        tgt_h, tgt_w = norm_tgt_img.shape[:2]

        refined_count = 0
        skipped_edge = 0
        skipped_score = 0
        skipped_unchanged = 0
        for mole in non_canonical:
            mole_uuid = mole["uuid"]
            template = templates_by_uuid[mole_uuid]

            norm_tgt_x = int(mole["x"] * tgt_scale)
            norm_tgt_y = int(mole["y"] * tgt_scale)

            # Extract search region around target mole
            half_search = search_size // 2
            sy1 = max(0, norm_tgt_y - half_search)
            sy2 = min(tgt_h, norm_tgt_y + half_search + 1)
            sx1 = max(0, norm_tgt_x - half_search)
            sx2 = min(tgt_w, norm_tgt_x + half_search + 1)

            search_region = norm_tgt_img[sy1:sy2, sx1:sx2]

            # Template must be smaller than search region
            tmpl_h, tmpl_w = template.shape[:2]
            sr_h, sr_w = search_region.shape[:2]
            if tmpl_h >= sr_h or tmpl_w >= sr_w:
                skipped_edge += 1
                continue

            # Run normalized cross-correlation
            result = cv2.matchTemplate(
                search_region, template, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val < min_score:
                skipped_score += 1
                continue

            # max_loc is top-left of template match; convert to center
            match_center_x = sx1 + max_loc[0] + tmpl_w // 2
            match_center_y = sy1 + max_loc[1] + tmpl_h // 2

            # Convert back to native coordinates
            native_new_x = int(match_center_x / tgt_scale)
            native_new_y = int(match_center_y / tgt_scale)

            # Check if position changed meaningfully (> 1px in native coords)
            dx = abs(native_new_x - mole["x"])
            dy = abs(native_new_y - mole["y"])
            if dx <= 1 and dy <= 1:
                skipped_unchanged += 1
                continue

            if not dry_run:
                mole["x"] = native_new_x
                mole["y"] = native_new_y
            refined_count += 1

        if verbose:
            print(
                f"  Refined {refined_count}/{len(non_canonical)} "
                f"(unchanged={skipped_unchanged}, "
                f"edge={skipped_edge}, "
                f"low_score={skipped_score})"
            )

        if refined_count > 0 and not dry_run:
            try:
                mel.rotomap.moles.save_image_moles(
                    tgt_moles, str(tgt_path), extra_stem=extra_stem
                )
            except Exception as e:
                print(f"Error saving moles to {tgt_path}: {e}")
                return 1

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
