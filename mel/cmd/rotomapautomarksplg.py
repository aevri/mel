"""Locate moles in target images using SuperPoint+LightGlue keypoint matching.

Uses geometric triangulation: SuperPoint detects interest points across each
image, LightGlue matches them between reference and target. For each canonical
mole, nearby matched keypoints define a local affine transform that maps the
mole's reference position to the target image. Works on full-resolution images
without GPU -- no crops or DINOv3 needed.
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


def _load_image_with_mask(image_path, verbose=False):
    """Load an image and apply its mask if available.

    Returns the image in BGR format. Masked-out pixels are set to constant
    gray to suppress spurious keypoints on background.
    """
    if verbose:
        print(f"Loading image: {image_path}")
    image_bgr = mel.lib.image.load_image(image_path)
    mask = mel.rotomap.mask.load_or_none(image_path)
    if mask is not None:
        if verbose:
            print("  Applying mask to image")
        keep_mask = mask > 127
        mean_val = 128
        for c in range(3):
            image_bgr[:, :, c] = np.where(
                keep_mask, image_bgr[:, :, c], mean_val
            )
    return image_bgr


def _image_to_tensor(image_bgr, device):
    """Convert BGR image to grayscale tensor for SuperPoint.

    Returns tensor of shape (1, 1, H, W), float32 in [0, 1].
    """
    import torch

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    tensor = torch.from_numpy(gray).float()[None, None] / 255.0
    return tensor.to(device)


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


def _predict_mole_position(ref_kpts, tgt_kpts, mole_x, mole_y,
                           keypoint_radius, min_matches):
    """Predict mole position in target using nearby keypoint matches.

    Args:
        ref_kpts: Matched keypoints in reference, shape (N, 2).
        tgt_kpts: Matched keypoints in target, shape (N, 2).
        mole_x, mole_y: Mole position in reference (normalized coords).
        keypoint_radius: Max distance from mole to include keypoints.
        min_matches: Minimum nearby matches required.

    Returns:
        Tuple of (pred_x, pred_y, num_nearby, num_inliers) or None if
        not enough nearby matches.
    """
    mole_pos = np.array([mole_x, mole_y], dtype=np.float64)
    distances = np.linalg.norm(ref_kpts - mole_pos, axis=1)
    nearby_mask = distances < keypoint_radius

    num_nearby = int(nearby_mask.sum())
    if num_nearby < min_matches:
        return None

    src_pts = ref_kpts[nearby_mask]
    dst_pts = tgt_kpts[nearby_mask]

    if num_nearby >= 2:
        # Fit similarity transform (rotation + uniform scale + translation)
        transform, inliers = cv2.estimateAffinePartial2D(
            src_pts.reshape(-1, 1, 2).astype(np.float64),
            dst_pts.reshape(-1, 1, 2).astype(np.float64),
        )
        if transform is None:
            return None
        num_inliers = int(inliers.sum()) if inliers is not None else num_nearby
        predicted = transform @ np.array([mole_x, mole_y, 1.0])
        return int(predicted[0]), int(predicted[1]), num_nearby, num_inliers

    # Single match: translation only
    offset_x = dst_pts[0, 0] - src_pts[0, 0]
    offset_y = dst_pts[0, 1] - src_pts[0, 1]
    return (
        int(mole_x + offset_x),
        int(mole_y + offset_y),
        1,
        1,
    )


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
        help="Target image(s) where moles will be located.",
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=2048,
        help="Maximum SuperPoint keypoints per image (default: 2048).",
    )
    parser.add_argument(
        "--keypoint-radius",
        type=float,
        default=300.0,
        help="Max distance (normalized pixels) from mole to include "
        "matched keypoints for triangulation (default: 300.0).",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=3,
        help="Minimum nearby matched keypoints required to predict a "
        "mole position (default: 3).",
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
        "--dry-run",
        action="store_true",
        help="Show what would be done without saving.",
    )


def process_args(args):
    ref_paths = args.reference
    tgt_paths = args.target
    max_keypoints = args.max_keypoints
    keypoint_radius = args.keypoint_radius
    min_matches = args.min_matches
    extra_stem = args.extra_stem
    verbose = args.verbose
    dry_run = args.dry_run

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

    norm_images = {}
    scale_factors = {}
    for path, img, scale in normalized:
        scale_factors[path] = scale
        norm_images[path] = img

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

    # Load moles from references
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

    # Import and initialize SuperPoint + LightGlue
    try:
        import torch
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import rbd
    except ImportError:
        print(
            "Error: lightglue package required. "
            "Install with: pip install lightglue"
        )
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"\nUsing device: {device}")
        print(f"Loading SuperPoint (max_keypoints={max_keypoints})...")

    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    # Extract features for all images (keep batch dims for matcher)
    if verbose:
        print("Extracting keypoints...")
    raw_feats = {}
    keypoint_counts = {}
    with torch.no_grad():
        for path in list(ref_images) + list(tgt_images):
            tensor = _image_to_tensor(norm_images[path], device)
            raw_feats[path] = extractor.extract(tensor)
            num_kpts = raw_feats[path]["keypoints"].shape[1]
            keypoint_counts[path] = num_kpts
            if verbose:
                print(f"  {path}: {num_kpts} keypoints")

    # Compute matches for each (ref, target) pair
    if verbose:
        print("\nMatching keypoints between image pairs...")
    pair_matches = {}
    with torch.no_grad():
        for ref_path in ref_paths:
            for tgt_path in tgt_paths:
                result = matcher({
                    "image0": raw_feats[ref_path],
                    "image1": raw_feats[tgt_path],
                })

                feats0_ub = rbd(raw_feats[ref_path])
                feats1_ub = rbd(raw_feats[tgt_path])
                result_ub = rbd(result)

                matches = result_ub["matches"]
                points0 = feats0_ub["keypoints"]
                points1 = feats1_ub["keypoints"]

                valid = matches > -1
                mkpts0 = points0[valid].cpu().numpy()
                mkpts1 = points1[matches[valid]].cpu().numpy()

                pair_matches[(ref_path, tgt_path)] = (mkpts0, mkpts1)

                if verbose:
                    print(
                        f"  {ref_path.name} <-> {tgt_path.name}: "
                        f"{len(mkpts0)} matches"
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

        # Find UUIDs missing from target
        tgt_all_uuids = {m["uuid"] for m in tgt_moles}
        all_canonical_uuids = set(ref_canonical_by_uuid.keys())
        missing_uuids = all_canonical_uuids - tgt_all_uuids

        if not missing_uuids:
            if verbose:
                print(
                    "  No missing moles - all reference canonical moles "
                    "already present in target"
                )
            continue

        if verbose:
            print(f"  {len(missing_uuids)} missing canonical moles to locate")

        tgt_scale = scale_factors[tgt_path]

        matched_count = 0
        skipped_no_matches = 0
        for missing_uuid in sorted(missing_uuids):
            ref_path, ref_mole = ref_canonical_by_uuid[missing_uuid]
            ref_scale = scale_factors[ref_path]

            # Mole position in normalized coordinates
            norm_ref_x = int(ref_mole["x"] * ref_scale)
            norm_ref_y = int(ref_mole["y"] * ref_scale)

            # Get matches for this ref-target pair
            ref_kpts, tgt_kpts = pair_matches[(ref_path, tgt_path)]

            if len(ref_kpts) == 0:
                skipped_no_matches += 1
                continue

            result = _predict_mole_position(
                ref_kpts, tgt_kpts,
                norm_ref_x, norm_ref_y,
                keypoint_radius, min_matches,
            )

            if result is None:
                skipped_no_matches += 1
                if verbose:
                    print(
                        f"  Skipping {missing_uuid}: not enough nearby "
                        f"matches (need {min_matches})"
                    )
                continue

            pred_x, pred_y, num_nearby, num_inliers = result

            # Convert to native target coordinates
            native_x = int(pred_x / tgt_scale)
            native_y = int(pred_y / tgt_scale)

            new_mole = {
                "uuid": missing_uuid,
                "x": native_x,
                "y": native_y,
                mel.rotomap.moles.KEY_IS_CONFIRMED: False,
            }
            tgt_moles.append(new_mole)
            matched_count += 1

            action = "Would add" if dry_run else "Added"
            if verbose:
                print(
                    f"  {action} mole {missing_uuid} at "
                    f"({native_x}, {native_y}) "
                    f"[nearby={num_nearby}, inliers={num_inliers}]"
                )

        if verbose:
            print(
                f"  Located {matched_count}/{len(missing_uuids)} moles "
                f"(skipped={skipped_no_matches})"
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
