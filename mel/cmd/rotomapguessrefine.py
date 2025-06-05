"""Refine non-canonical mole locations using DINOv2 feature matching."""

import argparse
import pathlib

import cv2
import numpy as np

import mel.lib.dinov2
import mel.lib.image
import mel.lib.math
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
        "SRC_JPG",
        type=_existing_file_path,
        help="Path to the source image with canonical mole locations.",
    )
    parser.add_argument(
        "TGT_JPG",
        type=_existing_file_path,
        help="Path to the target image with non-canonical moles to refine.",
    )
    parser.add_argument(
        "--debug-images",
        action="store_true",
        help="Save debug images showing source patches and target search areas.",
    )
    parser.add_argument(
        "--max-moles",
        type=int,
        default=None,
        help="Maximum number of moles to refine (for testing, default: refine all).",
    )
    parser.add_argument(
        "--dino-size",
        type=str,
        choices=["small", "base", "large", "giant"],
        default="base",
        help="DINOv2 model size variant (default: base). Smaller models are faster.",
    )


def save_debug_patch(patch, filename):
    """Save a patch for debugging purposes."""
    try:
        cv2.imwrite(filename, patch)
        print(f"  Debug: Saved patch to {filename}")
    except Exception as e:
        print(f"  Debug: Failed to save patch to {filename}: {e}")


def save_debug_search_area(image, center_x, center_y, patch_size, filename):
    """Save the target search area for debugging."""
    try:
        # Calculate search area bounds (use actual context size for accuracy)
        half_context = patch_size // 2
        # DINOv2 patch size is 14x14 pixels
        half_patch = 7
        search_left = max(0, center_x - half_context)
        search_right = min(image.shape[1], center_x + half_context)
        search_top = max(0, center_y - half_context)
        search_bottom = min(image.shape[0], center_y + half_context)

        # Extract search area
        search_area = image[search_top:search_bottom, search_left:search_right].copy()

        # Draw search grid and center marker
        # Center point in search area coordinates
        center_in_area_x = center_x - search_left
        center_in_area_y = center_y - search_top

        # Draw center cross
        cv2.line(
            search_area,
            (center_in_area_x - 10, center_in_area_y),
            (center_in_area_x + 10, center_in_area_y),
            (0, 255, 0),
            2,
        )
        cv2.line(
            search_area,
            (center_in_area_x, center_in_area_y - 10),
            (center_in_area_x, center_in_area_y + 10),
            (0, 255, 0),
            2,
        )

        # Draw context boundary circle
        cv2.circle(
            search_area,
            (center_in_area_x, center_in_area_y),
            half_context,
            (255, 0, 0),
            2,
        )

        # Draw patch size box
        cv2.rectangle(
            search_area,
            (center_in_area_x - half_patch, center_in_area_y - half_patch),
            (center_in_area_x + half_patch, center_in_area_y + half_patch),
            (0, 0, 255),
            2,
        )

        cv2.imwrite(filename, search_area)
        print(f"  Debug: Saved search area to {filename}")
    except Exception as e:
        print(f"  Debug: Failed to save search area to {filename}: {e}")


def process_args(args):
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torchvision.transforms as transforms

    src_path = args.SRC_JPG
    tgt_path = args.TGT_JPG
    debug_images = args.debug_images
    max_moles = args.max_moles
    dino_size = args.dino_size

    # Load images
    try:
        src_image = mel.lib.image.load_image(src_path)
        tgt_image = mel.lib.image.load_image(tgt_path)

        # Convert BGR to RGB for DINOv2
        src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        tgt_image = cv2.cvtColor(tgt_image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading images: {e}")
        return 1

    # Load moles
    try:
        src_moles = mel.rotomap.moles.load_image_moles(src_path)
        tgt_moles = mel.rotomap.moles.load_image_moles(tgt_path)
    except Exception as e:
        print(f"Error loading moles: {e}")
        return 1

    # Filter to get canonical moles from source and target
    src_canonical_moles = [
        m for m in src_moles if m[mel.rotomap.moles.KEY_IS_CONFIRMED]
    ]
    tgt_canonical_moles = [
        m for m in tgt_moles if m[mel.rotomap.moles.KEY_IS_CONFIRMED]
    ]

    if not src_canonical_moles:
        print("Error: No canonical moles found in source image")
        return 1
    if not tgt_canonical_moles:
        print("Error: No canonical moles found in target image")
        return 1

    # Find non-canonical moles in target that have matching UUIDs with canonical moles in source
    src_canonical_uuids = {m["uuid"] for m in src_canonical_moles}
    tgt_non_canonical_to_refine = [
        m
        for m in tgt_moles
        if not m[mel.rotomap.moles.KEY_IS_CONFIRMED]
        and m["uuid"] in src_canonical_uuids
    ]

    if not tgt_non_canonical_to_refine:
        print("No non-canonical moles in target that match canonical moles in source")
        return 0

    # Limit the number of moles to refine if max_moles is specified
    if max_moles is not None and max_moles > 0:
        original_count = len(tgt_non_canonical_to_refine)
        tgt_non_canonical_to_refine = tgt_non_canonical_to_refine[:max_moles]
        print(
            f"Found {original_count} non-canonical moles to refine, "
            f"limiting to first {len(tgt_non_canonical_to_refine)} moles"
        )
    else:
        print(f"Found {len(tgt_non_canonical_to_refine)} non-canonical moles to refine")
    print("Using DINOv2 contextual semantic features for mole matching")

    # Load DINOv2 model
    try:
        model, feature_dim = mel.lib.dinov2.load_dinov2_model(dino_size)
        print(
            f"DINOv2 model ({dino_size}) loaded successfully with {feature_dim} feature dimensions"
        )
    except RuntimeError as e:
        print(f"Error loading DINOv2 model: {e}")
        return 1

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create lookup dict for src canonical moles
    src_mole_lookup = {m["uuid"]: m for m in src_canonical_moles}

    context_size = 910  # Large context window for rich semantic features
    refined_count = 0

    # Process each non-canonical mole in target
    for tgt_mole in tgt_non_canonical_to_refine:
        uuid = tgt_mole["uuid"]
        src_mole = src_mole_lookup[uuid]

        print(f"Refining mole {uuid}...")

        # Extract contextual semantic features from source mole location
        try:
            src_features = mel.lib.dinov2.extract_contextual_patch_feature(
                src_image,
                src_mole["x"],
                src_mole["y"],
                context_size,
                model,
                transform,
                feature_dim,
            )
            print(
                f"  Extracted contextual features from {context_size}x{context_size} source context"
            )

            # Save debug image for source context patch
            if debug_images:
                # Extract context patch for debug visualization
                half_context = context_size // 2
                y_start = max(0, src_mole["y"] - half_context)
                y_end = min(src_image.shape[0], src_mole["y"] + half_context)
                x_start = max(0, src_mole["x"] - half_context)
                x_end = min(src_image.shape[1], src_mole["x"] + half_context)
                src_context_img = src_image[y_start:y_end, x_start:x_end]

                # Pad if necessary
                if (
                    src_context_img.shape[0] < context_size
                    or src_context_img.shape[1] < context_size
                ):
                    padded_patch = np.zeros(
                        (context_size, context_size, 3),
                        dtype=src_context_img.dtype,
                    )
                    y_offset = (context_size - src_context_img.shape[0]) // 2
                    x_offset = (context_size - src_context_img.shape[1]) // 2
                    padded_patch[
                        y_offset : y_offset + src_context_img.shape[0],
                        x_offset : x_offset + src_context_img.shape[1],
                    ] = src_context_img
                    src_context_img = padded_patch

                save_debug_patch(src_context_img, f"{uuid}_src_context.jpg")

        except Exception as e:
            print(f"Error extracting source features for mole {uuid}: {e}")
            continue

        # Save debug image for target search area
        if debug_images:
            save_debug_search_area(
                tgt_image,
                tgt_mole["x"],
                tgt_mole["y"],
                context_size,
                f"{uuid}_tgt_search_area.jpg",
            )

        # Find best matching location using contextual semantic matching
        try:
            best_x, best_y, similarity = mel.lib.dinov2.find_best_contextual_match(
                src_features,
                tgt_image,
                tgt_mole["x"],
                tgt_mole["y"],
                context_size,
                model,
                transform,
                feature_dim,
                debug_images,
                uuid,
            )

            # Update the mole location if we found a better match
            old_x, old_y = tgt_mole["x"], tgt_mole["y"]
            distance_moved = ((best_x - old_x) ** 2 + (best_y - old_y) ** 2) ** 0.5

            if distance_moved > 1:  # Only update if we moved more than 1 pixel
                tgt_mole["x"] = best_x
                tgt_mole["y"] = best_y
                refined_count += 1
                print(
                    f"  Refined from ({old_x}, {old_y}) to ({best_x}, {best_y}) "
                    f"(moved {distance_moved:.1f} pixels, similarity: {similarity:.3f})"
                )

                # Save debug image for final refined context
                if debug_images:
                    try:
                        half_context = context_size // 2
                        y_start = max(0, best_y - half_context)
                        y_end = min(tgt_image.shape[0], best_y + half_context)
                        x_start = max(0, best_x - half_context)
                        x_end = min(tgt_image.shape[1], best_x + half_context)
                        refined_context = tgt_image[y_start:y_end, x_start:x_end]

                        # Pad if necessary
                        if (
                            refined_context.shape[0] < context_size
                            or refined_context.shape[1] < context_size
                        ):
                            padded_patch = np.zeros(
                                (context_size, context_size, 3),
                                dtype=refined_context.dtype,
                            )
                            y_offset = (context_size - refined_context.shape[0]) // 2
                            x_offset = (context_size - refined_context.shape[1]) // 2
                            padded_patch[
                                y_offset : y_offset + refined_context.shape[0],
                                x_offset : x_offset + refined_context.shape[1],
                            ] = refined_context
                            refined_context = padded_patch

                        save_debug_patch(refined_context, f"{uuid}_tgt_refined.jpg")
                    except Exception as e:
                        print(f"  Debug: Failed to save refined context: {e}")
            else:
                print(f"  No refinement needed (similarity: {similarity:.3f})")

        except Exception as e:
            print(f"Error refining mole {uuid}: {e}")
            continue

    # Save refined moles if any were updated
    if refined_count > 0:
        try:
            mel.rotomap.moles.save_image_moles(tgt_moles, tgt_path)
            print(f"Successfully refined {refined_count} mole locations in {tgt_path}")
        except Exception as e:
            print(f"Error saving refined moles: {e}")
            return 1
    else:
        print("No moles required refinement")

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
