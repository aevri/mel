#! /usr/bin/env python3
"""Tests that exercise model downloads to verify network environment."""

import urllib.request


def test_download_benchmark_dataset():
    """Test that the benchmark dataset can be downloaded from GitHub."""
    dataset_url = (
        "https://github.com/aevri/mel-datasets/archive/refs/tags/v0.1.0.tar.gz"
    )
    print(f"Testing download from {dataset_url} ...")
    request = urllib.request.Request(dataset_url, method="HEAD")
    response = urllib.request.urlopen(request, timeout=30)
    status = response.getcode()
    print(f"  Response status: {status}")
    assert status == 200, f"Expected status 200, got {status}"
    print("  Benchmark dataset URL is reachable.")


def test_download_dinov2_model():
    """Test that the DINOv2 model can be downloaded via torch.hub."""
    import torch

    print("Testing DINOv2 model download via torch.hub ...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    assert model is not None
    print("  DINOv2 model downloaded successfully.")


def test_download_fasterrcnn_model():
    """Test that Faster R-CNN ResNet50 FPN weights can be downloaded."""
    import torchvision

    print("Testing Faster R-CNN ResNet50 FPN download ...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    assert model is not None
    print("  Faster R-CNN model downloaded successfully.")


def test_download_efficientnet_model():
    """Test that EfficientNet-B0 ImageNet weights can be downloaded."""
    import torchvision

    print("Testing EfficientNet-B0 download ...")
    model = torchvision.models.efficientnet_b0(
        weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    assert model is not None
    print("  EfficientNet-B0 model downloaded successfully.")


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
