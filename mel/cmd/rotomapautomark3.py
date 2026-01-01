"""Automatically mark moles in target images using DINOv3 feature matching from
reference images."""

import argparse
import pathlib
from collections import defaultdict

import cv2

import mel.lib.dinov3
import mel.lib.image
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
        "--reference",
        type=_existing_file_path,
        nargs="+",
        required=True,
        help="Path(s) to reference image(s) with canonical mole locations.",
    )
    parser.add_argument(
        "--target",
        type=_existing_file_path,
        nargs="+",
        required=True,
        help="Path(s) to target image(s) where moles should be automatically marked.",
    )
    parser.add_argument(
        "--dino-size",
        type=str,
        choices=["small", "base", "large", "huge", "giant"],
        default="base",
        help="DINOv3 model size variant (default: base). Larger models are more accurate but slower.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=-0.5,
        help="Minimum similarity score for marking a mole (default: -0.5). Higher is more strict.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=1024,
        help="Size of context window for feature extraction (default: 1024).",
    )
    parser.add_argument(
        "--debug-images",
        action="store_true",
        help="Save debug images showing heatmaps of feature matching.",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["mean", "first", "median"],
        default="mean",
        help="How to aggregate features when a mole appears in multiple reference images (default: mean).",
    )


def aggregate_features(features_list, aggregation_method):
    """Aggregate multiple feature vectors into a single representation.

    Args:
        features_list: List of feature tensors [feature_dim]
        aggregation_method: "mean", "first", or "median"

    Returns:
        Aggregated feature tensor [feature_dim]
    """
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torch

    if aggregation_method == "first":
        return features_list[0]
    if aggregation_method == "mean":
        stacked = torch.stack(features_list)
        return torch.mean(stacked, dim=0)
    if aggregation_method == "median":
        stacked = torch.stack(features_list)
        return torch.median(stacked, dim=0).values
    raise ValueError(f"Unknown aggregation method: {aggregation_method}")


def process_args(args):
    reference_paths = args.reference
    target_paths = args.target
    dino_size = args.dino_size
    similarity_threshold = args.similarity_threshold
    context_size = args.context_size
    debug_images = args.debug_images
    aggregation = args.aggregation

    print(f"Loading DINOv3 model ({dino_size})...")
    try:
        model, feature_dim = mel.lib.dinov3.load_dinov3_model(dino_size)
        print(f"DINOv3 model loaded successfully with {feature_dim} feature dimensions")
    except RuntimeError as e:
        print(f"Error loading DINOv3 model: {e}")
        return 1

    # Step 1: Gather all canonical moles from reference images
    print(f"\nProcessing {len(reference_paths)} reference image(s)...")
    reference_mole_features = defaultdict(
        list
    )  # uuid -> list of (features, x, y, image_path)

    for ref_path in reference_paths:
        print(f"  Loading reference image: {ref_path}")
        try:
            ref_image = mel.lib.image.load_image(ref_path)
            ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
            ref_moles = mel.rotomap.moles.load_image_moles(ref_path)
        except Exception as e:
            print(f"  Error loading reference image {ref_path}: {e}")
            continue

        # Filter to canonical moles only
        canonical_moles = [
            m for m in ref_moles if m[mel.rotomap.moles.KEY_IS_CONFIRMED]
        ]

        if not canonical_moles:
            print(f"  Warning: No canonical moles found in {ref_path}")
            continue

        print(f"  Found {len(canonical_moles)} canonical mole(s)")

        # Extract features for each canonical mole
        for mole in canonical_moles:
            uuid = mole["uuid"]
            try:
                features = mel.lib.dinov3.extract_contextual_patch_feature(
                    ref_image,
                    mole["x"],
                    mole["y"],
                    context_size,
                    model,
                    feature_dim,
                )
                reference_mole_features[uuid].append(
                    (features, mole["x"], mole["y"], str(ref_path))
                )
                print(f"    Extracted features for mole {uuid[:8]}...")
            except Exception as e:
                print(f"    Error extracting features for mole {uuid}: {e}")
                continue

    if not reference_mole_features:
        print("Error: No canonical moles with features found in reference images")
        return 1

    # Step 2: Aggregate features for moles appearing in multiple reference images
    print(
        f"\nAggregating features for {len(reference_mole_features)} unique mole(s)..."
    )
    aggregated_reference_moles = {}  # uuid -> (aggregated_features, representative_x, representative_y)

    for uuid, feature_list in reference_mole_features.items():
        print(f"  Mole {uuid[:8]}: appears in {len(feature_list)} reference image(s)")
        features_only = [f[0] for f in feature_list]
        aggregated_features = aggregate_features(features_only, aggregation)

        # Use the first occurrence as representative location (for reference)
        rep_x, rep_y = feature_list[0][1], feature_list[0][2]
        aggregated_reference_moles[uuid] = (aggregated_features, rep_x, rep_y)

    # Step 3: Process each target image
    print(f"\nProcessing {len(target_paths)} target image(s)...")
    total_marked = 0
    total_updated = 0
    total_skipped_canonical = 0

    for tgt_path in target_paths:
        print(f"\n  Processing target image: {tgt_path}")
        try:
            tgt_image = mel.lib.image.load_image(tgt_path)
            tgt_image = cv2.cvtColor(tgt_image, cv2.COLOR_BGR2RGB)
            tgt_moles = mel.rotomap.moles.load_image_moles(tgt_path)
        except Exception as e:
            print(f"    Error loading target image {tgt_path}: {e}")
            continue

        # Create lookup for existing moles
        existing_mole_uuids = {m["uuid"]: m for m in tgt_moles}
        moles_marked_in_this_image = 0
        moles_updated_in_this_image = 0

        # For each reference mole, try to find it in the target image
        for uuid, (ref_features, _ref_x, _ref_y) in aggregated_reference_moles.items():
            # Check if mole already exists in target
            if uuid in existing_mole_uuids:
                existing_mole = existing_mole_uuids[uuid]
                if existing_mole[mel.rotomap.moles.KEY_IS_CONFIRMED]:
                    # Skip canonical moles
                    print(f"    Mole {uuid[:8]}: already canonical, skipping")
                    total_skipped_canonical += 1
                    continue
                print(
                    f"    Mole {uuid[:8]}: exists but not canonical, will try to refine"
                )
                # Use existing location as starting point
                initial_x, initial_y = existing_mole["x"], existing_mole["y"]
            else:
                # New mole - use image center as starting point
                initial_x = tgt_image.shape[1] // 2
                initial_y = tgt_image.shape[0] // 2
                print(f"    Mole {uuid[:8]}: not found, searching from image center")

            # Find best match in target image
            try:
                best_x, best_y, similarity = mel.lib.dinov3.find_best_contextual_match(
                    ref_features,
                    tgt_image,
                    initial_x,
                    initial_y,
                    context_size,
                    model,
                    feature_dim,
                    debug_images,
                    uuid if debug_images else None,
                )

                # Check if similarity passes threshold
                if similarity >= similarity_threshold:
                    if uuid in existing_mole_uuids:
                        # Update existing mole location
                        existing_mole_uuids[uuid]["x"] = best_x
                        existing_mole_uuids[uuid]["y"] = best_y
                        moles_updated_in_this_image += 1
                        total_updated += 1
                        print(
                            f"    Mole {uuid[:8]}: updated to ({best_x}, {best_y}), similarity={similarity:.3f}"
                        )
                    else:
                        # Add new mole
                        new_mole = {
                            "uuid": uuid,
                            "x": best_x,
                            "y": best_y,
                            mel.rotomap.moles.KEY_IS_CONFIRMED: False,
                        }
                        tgt_moles.append(new_mole)
                        existing_mole_uuids[uuid] = new_mole
                        moles_marked_in_this_image += 1
                        total_marked += 1
                        print(
                            f"    Mole {uuid[:8]}: marked at ({best_x}, {best_y}), similarity={similarity:.3f}"
                        )
                else:
                    print(
                        f"    Mole {uuid[:8]}: similarity {similarity:.3f} below threshold {similarity_threshold}, skipping"
                    )

            except Exception as e:
                print(f"    Error matching mole {uuid}: {e}")
                continue

        # Save updated moles for this target image
        if moles_marked_in_this_image > 0 or moles_updated_in_this_image > 0:
            try:
                mel.rotomap.moles.save_image_moles(tgt_moles, tgt_path)
                print(
                    f"    Saved {moles_marked_in_this_image} new + {moles_updated_in_this_image} updated moles to {tgt_path}"
                )
            except Exception as e:
                print(f"    Error saving moles to {tgt_path}: {e}")
                continue
        else:
            print(f"    No moles marked or updated in {tgt_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Total new moles marked: {total_marked}")
    print(f"  Total moles updated: {total_updated}")
    print(f"  Total canonical moles skipped: {total_skipped_canonical}")
    print(f"{'=' * 60}")

    return 0


# -----------------------------------------------------------------------------
# Copyright (C) 2025 Angelos Evripiotis.
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
