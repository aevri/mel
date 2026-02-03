"""Locate canonical moles using a neural network classifier on DINOv3 features.

Trains a temporary in-memory classifier to recognize moles from reference images,
then applies it to find moles in target images.
"""

import argparse
import pathlib

import cv2
import torch

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


def _get_features_path(image_path, dino_size, image_size):
    """Generate path for cached features file."""
    return pathlib.Path(f"{image_path}.dino3-{dino_size}-{image_size}.pt")


def _load_cached_features(image_path, dino_size, image_size, verbose=False):
    """Load cached features if available, returns None if not found."""
    features_path = _get_features_path(image_path, dino_size, image_size)
    if not features_path.exists():
        return None
    if verbose:
        print(f"Loading cached features: {features_path}")
    return torch.load(features_path)


def _tensor_size_mb(tensor):
    """Return size of a tensor in megabytes."""
    return tensor.numel() * tensor.element_size() / (1024 * 1024)


def _get_patch_index(img_w, mole_x, mole_y):
    """Convert scaled coordinates to patch index."""
    patch_size = mel.lib.dinov3.PATCH_SIZE
    patches_per_row = img_w // patch_size
    patch_col = mole_x // patch_size
    patch_row = mole_y // patch_size
    return patch_row * patches_per_row + patch_col


def _patch_index_to_coords(patch_idx, img_w):
    """Convert patch index to patch center coordinates."""
    patch_size = mel.lib.dinov3.PATCH_SIZE
    patches_per_row = img_w // patch_size
    patch_row = patch_idx // patches_per_row
    patch_col = patch_idx % patches_per_row
    x = patch_col * patch_size + patch_size // 2
    y = patch_row * patch_size + patch_size // 2
    return x, y


def _get_patch_distance(idx1, idx2, img_w):
    """Compute distance between two patches in patch units."""
    patch_size = mel.lib.dinov3.PATCH_SIZE
    patches_per_row = img_w // patch_size
    row1, col1 = idx1 // patches_per_row, idx1 % patches_per_row
    row2, col2 = idx2 // patches_per_row, idx2 % patches_per_row
    return ((row1 - row2) ** 2 + (col1 - col2) ** 2) ** 0.5


class MoleClassifier(torch.nn.Module):
    """MLP classifier for mole identification."""

    def __init__(self, feature_dim, num_classes, hidden_layers):
        """Initialize the classifier.

        Args:
            feature_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_layers: List of hidden layer sizes, e.g., [256] or [512, 256]
        """
        super().__init__()
        layers = []
        prev_dim = feature_dim
        for hidden_dim in hidden_layers:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.2))
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, num_classes))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def _collect_training_data(
    ref_moles_by_path,
    ref_data,
    uuid_to_class,
    negative_ratio,
    min_patch_distance,
    verbose,
):
    """Collect training features and labels from reference images.

    Args:
        ref_moles_by_path: {ref_path: [canonical_moles]}
        ref_data: {ref_path: {features, scale_x, scale_y, scaled_w}}
        uuid_to_class: {uuid: class_index} mapping (1-indexed, 0 is no-mole)
        negative_ratio: Ratio of negative samples to positive samples
        min_patch_distance: Min distance (in patches) from moles for negative samples
        verbose: Print detailed info

    Returns:
        Tuple of (features_tensor, labels_tensor)
    """
    positive_features = []
    positive_labels = []
    all_mole_patches = {}  # {ref_path: set of patch indices containing moles}

    # Collect positive samples (mole patches)
    for ref_path, moles in ref_moles_by_path.items():
        data = ref_data[ref_path]
        features = data["features"]
        scale_x = data["scale_x"]
        scale_y = data["scale_y"]
        scaled_w = data["scaled_w"]

        mole_patch_indices = set()
        for mole in moles:
            scaled_x = int(mole["x"] * scale_x)
            scaled_y = int(mole["y"] * scale_y)
            patch_idx = _get_patch_index(scaled_w, scaled_x, scaled_y)
            mole_patch_indices.add(patch_idx)

            mole_feature = features[patch_idx]
            class_idx = uuid_to_class[mole["uuid"]]

            positive_features.append(mole_feature)
            positive_labels.append(class_idx)

        all_mole_patches[ref_path] = mole_patch_indices

    if verbose:
        print(f"  Collected {len(positive_features)} positive samples")

    # Collect negative samples (non-mole patches)
    negative_features = []
    num_negatives_needed = int(len(positive_features) * negative_ratio)

    for ref_path, _moles in ref_moles_by_path.items():
        data = ref_data[ref_path]
        features = data["features"]
        scaled_w = data["scaled_w"]
        mole_patch_indices = all_mole_patches[ref_path]

        # Find patches far enough from any mole
        num_patches = features.shape[0]
        valid_negative_indices = []
        for patch_idx in range(num_patches):
            is_far_enough = True
            for mole_idx in mole_patch_indices:
                dist = _get_patch_distance(patch_idx, mole_idx, scaled_w)
                if dist < min_patch_distance:
                    is_far_enough = False
                    break
            if is_far_enough:
                valid_negative_indices.append(patch_idx)

        # Sample from valid negative patches
        samples_from_this_ref = min(
            len(valid_negative_indices),
            num_negatives_needed // len(ref_moles_by_path),
        )
        if samples_from_this_ref > 0:
            # Deterministic sampling: pick evenly spaced indices
            step = max(1, len(valid_negative_indices) // samples_from_this_ref)
            selected = valid_negative_indices[::step][:samples_from_this_ref]
            for patch_idx in selected:
                negative_features.append(features[patch_idx])

    if verbose:
        print(f"  Collected {len(negative_features)} negative samples")

    # Combine into tensors
    all_features = positive_features + negative_features
    all_labels = positive_labels + [0] * len(negative_features)  # 0 = no mole

    features_tensor = torch.stack(all_features)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    return features_tensor, labels_tensor


def _train_classifier(
    features, labels, num_classes, hidden_layers, epochs, weight_decay, verbose
):
    """Train the mole classifier.

    Args:
        features: Training features [N, feature_dim]
        labels: Training labels [N]
        num_classes: Number of classes (num_moles + 1)
        hidden_layers: List of hidden layer sizes
        epochs: Number of training epochs
        weight_decay: L2 regularization strength
        verbose: Print training progress

    Returns:
        Trained MoleClassifier model
    """
    feature_dim = features.shape[1]
    model = MoleClassifier(feature_dim, num_classes, hidden_layers)

    # Move to same device as features
    device = features.device
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=weight_decay)
    # OneCycleLR with max_lr=0.01: higher values (e.g. 0.03) cause training
    # instability with large loss spikes around the warmup peak.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, total_steps=epochs
    )

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose and (epoch + 1) % 20 == 0:
            with torch.no_grad():
                predictions = outputs.argmax(dim=1)
                accuracy = (predictions == labels).float().mean().item()
                print(f"    Epoch {epoch + 1}/{epochs}: loss={loss.item():.4f}, "
                      f"accuracy={accuracy:.4f}")

    model.eval()
    return model


def setup_parser(parser):
    parser.add_argument(
        "--reference",
        "-r",
        type=_existing_file_path,
        nargs="+",
        required=True,
        help="Reference image(s) with canonical moles to match from.",
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
        "--dino-size",
        type=str,
        choices=["small", "base", "large", "huge", "7b"],
        default="7b",
        help="DINOv3 model size variant (default: 7b).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Scale images to fit this size in pixels (default: 1024). "
        "Must be divisible by 16.",
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
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for matches (default: 0.5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without saving.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=3.0,
        help="Ratio of negative samples to positive samples (default: 3.0).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="L2 regularization strength for AdamW optimizer (default: 0.01).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible training.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[256],
        help="Hidden layer sizes (default: 256). E.g., --hidden-layers 512 256.",
    )


def process_args(args):
    ref_paths = args.reference
    tgt_paths = args.target
    dino_size = args.dino_size
    image_size = args.image_size
    allow_download = args.allow_download
    extra_stem = args.extra_stem
    verbose = args.verbose
    min_confidence = args.min_confidence
    dry_run = args.dry_run
    epochs = args.epochs
    negative_ratio = args.negative_ratio
    weight_decay = args.weight_decay
    seed = args.seed
    hidden_layers = args.hidden_layers

    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if verbose:
            print(f"Random seed set to {seed}")

    # Validate image_size is divisible by patch size
    if image_size % mel.lib.dinov3.PATCH_SIZE != 0:
        print(
            f"Error: --image-size must be divisible by "
            f"{mel.lib.dinov3.PATCH_SIZE}, got {image_size}"
        )
        return 1

    # Collect canonical moles from all reference images
    ref_moles_by_path = {}
    all_canonical_uuids = set()
    for ref_path in ref_paths:
        try:
            moles = mel.rotomap.moles.load_image_moles(ref_path)
        except Exception as e:
            print(f"Error loading moles from {ref_path}: {e}")
            return 1
        canonical = [m for m in moles if m[mel.rotomap.moles.KEY_IS_CONFIRMED]]
        if canonical:
            ref_moles_by_path[ref_path] = canonical
            all_canonical_uuids.update(m["uuid"] for m in canonical)

    if not all_canonical_uuids:
        print("Error: No canonical moles found in any reference image")
        return 1

    if verbose:
        print(
            f"Found {len(all_canonical_uuids)} unique canonical moles "
            f"across {len(ref_moles_by_path)} reference images"
        )

    # Create UUID to class mapping (class 0 = no mole, 1..N = mole UUIDs)
    uuid_list = sorted(all_canonical_uuids)
    uuid_to_class = {uuid: i + 1 for i, uuid in enumerate(uuid_list)}
    num_classes = len(uuid_list) + 1  # +1 for "no mole" class

    if verbose:
        print(f"Training classifier with {num_classes} classes "
              f"({len(uuid_list)} moles + 1 no-mole)")

    # Try to load cached features for references
    ref_cached = {}
    refs_needing_computation = []
    for ref_path in ref_moles_by_path:
        cached = _load_cached_features(ref_path, dino_size, image_size, verbose)
        if cached:
            ref_cached[ref_path] = cached
        else:
            refs_needing_computation.append(ref_path)

    # Check which targets have cached features
    tgt_cached = {}
    targets_needing_computation = []
    for tgt_path in tgt_paths:
        cached = _load_cached_features(tgt_path, dino_size, image_size, verbose)
        if cached:
            tgt_cached[tgt_path] = cached
        else:
            targets_needing_computation.append(tgt_path)

    # Only load DINOv3 model if needed for feature extraction
    need_dino = (
        len(refs_needing_computation) > 0 or len(targets_needing_computation) > 0
    )
    if need_dino:
        if verbose:
            print(f"Loading DINOv3 model (size: {dino_size})...")
            print("References needing computation: ", refs_needing_computation)
            print("Targets needing computation: ", targets_needing_computation)
        try:
            dino_model, feature_dim = mel.lib.dinov3.load_dinov3_model(
                dino_size, local_files_only=not allow_download
            )
            if verbose:
                print(f"Model loaded with {feature_dim} feature dimensions")
        except RuntimeError as e:
            print(f"Error loading DINOv3 model: {e}")
            return 1
    else:
        dino_model = None
        if verbose:
            print("All features cached, skipping DINOv3 model load")

    # Load/compute features for all references
    ref_data = {}
    for ref_path in ref_moles_by_path:
        if ref_path in ref_cached:
            cached = ref_cached[ref_path]
            ref_data[ref_path] = {
                "features": cached["features"],
                "scale_x": cached["scale_x"],
                "scale_y": cached["scale_y"],
                "scaled_w": cached["scaled_w"],
            }
            if verbose:
                print(
                    f"Using cached reference features for {ref_path}: "
                    f"{cached['features'].shape[0]} patches, "
                    f"{_tensor_size_mb(cached['features']):.1f} MB"
                )
        else:
            ref_image_rgb = _load_image_with_mask(ref_path, verbose)
            if verbose:
                print(f"Scaling reference image {ref_path} and extracting features...")
            scaled_ref, (scale_x, scale_y) = mel.lib.dinov3.scale_image_to_fit(
                ref_image_rgb, image_size
            )
            scaled_w = scaled_ref.shape[1]
            features = mel.lib.dinov3.extract_all_patch_features(scaled_ref, dino_model)
            ref_data[ref_path] = {
                "features": features,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "scaled_w": scaled_w,
            }
            if verbose:
                print(
                    f"  Reference features: {features.shape[0]} patches, "
                    f"{_tensor_size_mb(features):.1f} MB"
                )

    # Collect training data
    if verbose:
        print("Collecting training data...")
    min_patch_distance = 2  # Minimum distance in patches for negative samples
    train_features, train_labels = _collect_training_data(
        ref_moles_by_path,
        ref_data,
        uuid_to_class,
        negative_ratio,
        min_patch_distance,
        verbose,
    )

    # Train classifier
    if verbose:
        print("Training classifier...")
    classifier = _train_classifier(
        train_features, train_labels, num_classes, hidden_layers, epochs, weight_decay,
        verbose
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
        missing_uuids = all_canonical_uuids - tgt_all_uuids

        if not missing_uuids:
            if verbose:
                print(
                    "  No missing moles - all reference canonical moles "
                    "already present in target"
                )
            continue

        if verbose:
            print(f"  Found {len(missing_uuids)} missing canonical moles to locate")

        # Get target features (from cache or compute)
        if tgt_path in tgt_cached:
            cached = tgt_cached[tgt_path]
            target_features = cached["features"]
            tgt_scale_x = cached["scale_x"]
            tgt_scale_y = cached["scale_y"]
            scaled_tgt_w = cached["scaled_w"]
            if verbose:
                print(
                    f"  Using cached target features: {target_features.shape[0]} "
                    f"patches, {_tensor_size_mb(target_features):.1f} MB"
                )
        else:
            tgt_image_rgb = _load_image_with_mask(tgt_path, verbose)
            if verbose:
                print("  Scaling target image and extracting features...")
            scaled_tgt, (tgt_scale_x, tgt_scale_y) = mel.lib.dinov3.scale_image_to_fit(
                tgt_image_rgb, image_size
            )
            scaled_tgt_w = scaled_tgt.shape[1]
            target_features = mel.lib.dinov3.extract_all_patch_features(
                scaled_tgt, dino_model
            )
            if verbose:
                print(
                    f"  Target features: {target_features.shape[0]} patches, "
                    f"{_tensor_size_mb(target_features):.1f} MB"
                )

        # Run classifier on all target patches
        with torch.no_grad():
            logits = classifier(target_features)
            probs = torch.nn.functional.softmax(logits, dim=1)

        matched_count = 0
        for missing_uuid in missing_uuids:
            class_idx = uuid_to_class[missing_uuid]

            # Find patch with highest probability for this mole class
            class_probs = probs[:, class_idx]
            best_patch_idx = class_probs.argmax().item()
            confidence = class_probs[best_patch_idx].item()

            if verbose:
                print(
                    f"  Mole {missing_uuid}: best patch {best_patch_idx} "
                    f"with confidence {confidence:.4f}"
                )

            if confidence < min_confidence:
                if verbose:
                    print(
                        f"    Skipping: confidence {confidence:.4f} "
                        f"below threshold {min_confidence}"
                    )
                continue

            # Convert patch index to coordinates
            scaled_x, scaled_y = _patch_index_to_coords(best_patch_idx, scaled_tgt_w)
            final_x = int(scaled_x / tgt_scale_x)
            final_y = int(scaled_y / tgt_scale_y)

            # Add the mole as non-canonical
            new_mole = {
                "uuid": missing_uuid,
                "x": final_x,
                "y": final_y,
                mel.rotomap.moles.KEY_IS_CONFIRMED: False,
                "confidence": confidence,
            }
            tgt_moles.append(new_mole)
            matched_count += 1

            action = "Would add" if dry_run else "Added"
            print(
                f"  {action} mole {missing_uuid} at ({final_x}, {final_y}) "
                f"[confidence: {confidence:.4f}]"
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
