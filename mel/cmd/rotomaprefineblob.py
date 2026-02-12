"""Refine mole positions using blob detection in local regions.

Detects dark blobs near each mole's approximate position and snaps the mole
to the nearest blob center. This is a simple classical CV approach that
requires no reference image features -- only the approximate location and
the assumption that moles appear as dark spots on skin.
"""

import argparse
import pathlib

import cv2

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


def _load_image(image_path, verbose=False):
    """Load an image in BGR format."""
    if verbose:
        print(f"Loading image: {image_path}")
    return mel.lib.image.load_image(image_path)


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


def _find_nearest_blob(image_bgr, center_x, center_y, search_size,
                       min_area, max_area, verbose=False):
    """Find the nearest dark blob to the given center point.

    Args:
        image_bgr: Image in BGR format.
        center_x, center_y: Center of the search region.
        search_size: Size of the square search region.
        min_area: Minimum blob area in pixels.
        max_area: Maximum blob area in pixels.
        verbose: Print debug info.

    Returns:
        Tuple of (blob_x, blob_y, distance) in image coordinates,
        or None if no blob found.
    """
    h, w = image_bgr.shape[:2]

    # Extract search region
    half = search_size // 2
    y1 = max(0, center_y - half)
    y2 = min(h, center_y + half + 1)
    x1 = max(0, center_x - half)
    x2 = min(w, center_x + half + 1)

    region = image_bgr[y1:y2, x1:x2]

    # Convert to grayscale
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Set up SimpleBlobDetector to find dark blobs
    params = cv2.SimpleBlobDetector_Params()

    # Filter by color (dark blobs)
    params.filterByColor = True
    params.blobColor = 0  # Dark blobs

    # Filter by area
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area

    # Filter by circularity (moles are roughly circular)
    params.filterByCircularity = True
    params.minCircularity = 0.3

    # Filter by convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Filter by inertia (roundness)
    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    # Thresholds for blob detection
    params.minThreshold = 10
    params.maxThreshold = 200
    params.thresholdStep = 10

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)

    if verbose:
        print(f"    Found {len(keypoints)} blobs in search region")

    if not keypoints:
        return None

    # Find blob nearest to search region center
    region_cx = center_x - x1
    region_cy = center_y - y1

    best_kp = None
    best_dist = float("inf")
    for kp in keypoints:
        dx = kp.pt[0] - region_cx
        dy = kp.pt[1] - region_cy
        dist = (dx * dx + dy * dy) ** 0.5
        if verbose:
            blob_img_x = int(x1 + kp.pt[0])
            blob_img_y = int(y1 + kp.pt[1])
            print(
                f"      Blob at ({blob_img_x}, {blob_img_y}) "
                f"size={kp.size:.1f} dist={dist:.1f}"
            )
        if dist < best_dist:
            best_dist = dist
            best_kp = kp

    blob_x = int(x1 + best_kp.pt[0])
    blob_y = int(y1 + best_kp.pt[1])
    return blob_x, blob_y, best_dist


def setup_parser(parser):
    parser.add_argument(
        "--reference",
        "-r",
        type=_existing_file_path,
        nargs="+",
        required=True,
        help="Reference image(s) with canonical moles (used only for "
        "UUID lookup, not for visual matching).",
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
        "--search-size",
        type=int,
        default=121,
        help="Size of search region around each mole in normalized "
        "pixels (default: 121).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=10.0,
        help="Minimum blob area in pixels (default: 10.0).",
    )
    parser.add_argument(
        "--max-area",
        type=float,
        default=5000.0,
        help="Maximum blob area in pixels (default: 5000.0).",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.0,
        help="Maximum distance from current position to accept a blob "
        "(default: 0.0 = no limit). In normalized pixels.",
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
    search_size = args.search_size
    min_area = args.min_area
    max_area = args.max_area
    max_distance = args.max_distance
    extra_stem = args.extra_stem
    verbose = args.verbose
    dry_run = args.dry_run

    # Load all images (no masks needed for blob detection - we want the
    # actual skin/mole appearance)
    all_images = []
    ref_images = {}
    for ref_path in ref_paths:
        img = _load_image(ref_path, verbose)
        ref_images[ref_path] = img
        all_images.append((ref_path, img))

    tgt_images = {}
    for tgt_path in tgt_paths:
        img = _load_image(tgt_path, verbose)
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

    norm_tgt_images = {}
    scale_factors = {}
    for path, img, scale in normalized:
        scale_factors[path] = scale
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

    # Load moles from references (build set of canonical UUIDs)
    canonical_uuids = set()
    for ref_path in ref_paths:
        try:
            moles = mel.rotomap.moles.load_image_moles(ref_path)
        except Exception as e:
            print(f"Error loading moles from {ref_path}: {e}")
            return 1
        for m in moles:
            if m[mel.rotomap.moles.KEY_IS_CONFIRMED]:
                canonical_uuids.add(m["uuid"])

    if not canonical_uuids:
        print("Error: No canonical moles found in any reference image")
        return 1

    if verbose:
        print(
            f"\nFound {len(canonical_uuids)} canonical UUIDs "
            f"across {len(ref_paths)} reference images"
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

        # Find non-canonical moles that have matching canonical UUIDs
        non_canonical = [
            m for m in tgt_moles
            if not m[mel.rotomap.moles.KEY_IS_CONFIRMED]
            and m["uuid"] in canonical_uuids
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

            norm_tgt_x = int(mole["x"] * tgt_scale)
            norm_tgt_y = int(mole["y"] * tgt_scale)

            if verbose:
                print(f"  Refining mole {mole_uuid}")
                print(
                    f"    Target pos (normalized): ({norm_tgt_x}, {norm_tgt_y})"
                )

            result = _find_nearest_blob(
                norm_tgt_img,
                norm_tgt_x,
                norm_tgt_y,
                search_size,
                min_area,
                max_area,
                verbose,
            )

            if result is None:
                if verbose:
                    print("    No blob found in search region")
                continue

            blob_x, blob_y, dist = result

            if max_distance > 0 and dist > max_distance:
                if verbose:
                    print(
                        f"    Skipping: blob distance {dist:.1f} "
                        f"exceeds max {max_distance}"
                    )
                continue

            # Convert back to native coordinates
            native_new_x = int(blob_x / tgt_scale)
            native_new_y = int(blob_y / tgt_scale)

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
                f"[d={dx},{dy} dist={dist:.1f}]"
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
