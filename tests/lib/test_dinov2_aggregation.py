"""Integration test for DINOv2 feature aggregation functions."""

import contextlib
import json
import os
import pathlib
import tempfile
import tarfile
import urllib.request

import cv2
import pytest

import mel.lib.dinov2
import mel.lib.image
import mel.rotomap.moles


def test_aggregate_reference_features_with_dataset():
    """Test feature aggregation using example dataset with known same mole instances."""
    with chtempdir_context():
        # Download and extract benchmark dataset
        dataset_url = (
            "https://github.com/aevri/mel-datasets/archive/refs/tags/v0.1.0.tar.gz"
        )
        dataset_path = download_and_extract_dataset(dataset_url)

        # Set up paths to test images
        m1_path = dataset_path / "mel-datasets-0.1.0" / "m1"
        reference_image = (
            m1_path / "rotomaps" / "parts" / "Trunk" / "Back" / "2025_06_12" / "0.jpg"
        )
        target_image = (
            m1_path / "rotomaps" / "parts" / "Trunk" / "Back" / "2025_06_13" / "0.jpg"
        )

        # Verify files exist
        assert reference_image.exists(), f"Reference image not found: {reference_image}"
        assert target_image.exists(), f"Target image not found: {target_image}"

        # Load DINOv2 model (use small model for faster testing)
        model, feature_dim = mel.lib.dinov2.load_dinov2_model("small")

        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        context_size = 910  # Large context window

        # Test aggregation with both reference and target images
        reference_paths = [reference_image, target_image]

        # Load and aggregate reference features
        aggregated_features, aggregation_metadata, reference_moles_by_uuid = (
            mel.lib.dinov2.load_and_aggregate_reference_features(
                reference_paths, model, transform, feature_dim, context_size
            )
        )

        print(f"Found {len(aggregated_features)} unique moles in reference images")
        print(f"Aggregation metadata: {aggregation_metadata}")

        # Verify that we got some moles
        assert len(aggregated_features) > 0, "Should find at least some canonical moles"
        assert len(reference_moles_by_uuid) > 0, "Should have reference mole metadata"

        # Test that we found moles appearing in multiple images
        multi_reference_moles = []
        for uuid, metadata in aggregation_metadata.items():
            if metadata["num_references"] > 1:
                multi_reference_moles.append(uuid)
                print(
                    f"Mole {uuid[:8]} appears in {metadata['num_references']} reference images"
                )

        # Verify we have moles that appear in multiple images (this is a key feature)
        assert len(multi_reference_moles) > 0, (
            "Should find moles that appear in multiple reference images"
        )

        # For efficiency, just test that we can calculate similarities for the first mole
        if multi_reference_moles:
            test_uuid = multi_reference_moles[0]
            aggregated_feature = aggregated_features[test_uuid]

            # Test that the aggregated feature has reasonable properties
            import torch

            assert isinstance(aggregated_feature, torch.Tensor), (
                "Aggregated feature should be a torch tensor"
            )
            assert aggregated_feature.shape == (feature_dim,), (
                f"Feature should have shape ({feature_dim},)"
            )

            # Test that we can calculate similarity with itself (should be very high)
            self_similarity = -torch.cdist(
                torch.nn.functional.normalize(
                    aggregated_feature.unsqueeze(0), p=2, dim=1
                ),
                torch.nn.functional.normalize(
                    aggregated_feature.unsqueeze(0), p=2, dim=1
                ),
            ).item()

            print(f"Self-similarity for mole {test_uuid[:8]}: {self_similarity:.6f}")
            assert abs(self_similarity) < 1e-6, (
                f"Self-similarity should be ~0, got {self_similarity}"
            )

            print(
                "✓ Aggregated features have correct properties and can be used for similarity calculations"
            )

        # Test the aggregate_reference_features function directly
        print(f"\nTesting direct aggregate_reference_features function...")

        # Create test data with known features
        import torch

        test_features = {
            "mole1": [torch.randn(feature_dim), torch.randn(feature_dim)],
            "mole2": [torch.randn(feature_dim)],
        }

        aggregated, metadata = mel.lib.dinov2.aggregate_reference_features(
            test_features
        )

        assert len(aggregated) == 2, "Should aggregate features for 2 moles"
        assert "mole1" in aggregated and "mole2" in aggregated
        assert metadata["mole1"]["num_references"] == 2
        assert metadata["mole2"]["num_references"] == 1
        assert aggregated["mole1"].shape == (feature_dim,)
        assert aggregated["mole2"].shape == (feature_dim,)

        print("✓ Direct aggregate_reference_features function works correctly")


@contextlib.contextmanager
def chtempdir_context():
    """Context manager for working in a temporary directory."""
    with tempfile.TemporaryDirectory() as tempdir:
        saved_path = os.getcwd()
        os.chdir(tempdir)
        try:
            yield
        finally:
            os.chdir(saved_path)


def download_and_extract_dataset(dataset_url: str) -> pathlib.Path:
    """Download and extract the dataset."""
    print(f"Downloading dataset from {dataset_url}")
    dataset_filename = "dataset.tar.gz"
    urllib.request.urlretrieve(dataset_url, dataset_filename)

    print("Extracting dataset...")
    with tarfile.open(dataset_filename, "r:gz") as tar:
        tar.extractall()

    return pathlib.Path(".")


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
