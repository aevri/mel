"""Unit tests for mel.rotomap.mask."""

import pathlib

import cv2
import numpy as np
import pytest

from mel.rotomap import mask

# -----------------------------------------------------------------------------
# path()
# -----------------------------------------------------------------------------


def test_path_string():
    result = mask.path("/some/image.jpg")
    assert result == "/some/image.jpg.mask.png"


def test_path_pathlib():
    result = mask.path(pathlib.Path("/some/image.jpg"))
    assert result == "/some/image.jpg.mask.png"


def test_path_with_spaces():
    result = mask.path("/some/path with spaces/image.jpg")
    assert result == "/some/path with spaces/image.jpg.mask.png"


def test_path_with_special_characters():
    result = mask.path("/some/path-with_special.chars/image.jpg")
    assert result == "/some/path-with_special.chars/image.jpg.mask.png"


# -----------------------------------------------------------------------------
# has_mask()
# -----------------------------------------------------------------------------


def test_has_mask_exists(tmp_path):
    image_path = tmp_path / "image.jpg"
    mask_path = tmp_path / "image.jpg.mask.png"
    mask_path.write_bytes(b"")
    assert mask.has_mask(image_path) is True


def test_has_mask_not_exists(tmp_path):
    image_path = tmp_path / "image.jpg"
    assert mask.has_mask(image_path) is False


# -----------------------------------------------------------------------------
# load()
# -----------------------------------------------------------------------------


def _write_mask(path, shape=(10, 10)):
    img = np.zeros(shape, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_load_valid(tmp_path):
    image_path = tmp_path / "image.jpg"
    mask_path = tmp_path / "image.jpg.mask.png"
    _write_mask(mask_path)
    result = mask.load(image_path)
    assert result is not None
    assert isinstance(result, np.ndarray)


def test_load_missing_raises(tmp_path):
    image_path = tmp_path / "nonexistent.jpg"
    # Source uses bare Exception, so we match that broad type intentionally.
    with pytest.raises(Exception, match="Failed to load mask"):
        mask.load(image_path)


# -----------------------------------------------------------------------------
# load_or_none()
# -----------------------------------------------------------------------------


def test_load_or_none_valid(tmp_path):
    image_path = tmp_path / "image.jpg"
    mask_path = tmp_path / "image.jpg.mask.png"
    _write_mask(mask_path)
    result = mask.load_or_none(image_path)
    assert result is not None
    assert isinstance(result, np.ndarray)


def test_load_or_none_missing(tmp_path):
    image_path = tmp_path / "nonexistent.jpg"
    result = mask.load_or_none(image_path)
    assert result is None


# -----------------------------------------------------------------------------
# mask_biggest_region()
# -----------------------------------------------------------------------------


def test_mask_biggest_region_single_region():
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), 255, -1)
    result = mask.mask_biggest_region(img)
    assert result.shape == img.shape
    assert result.sum() > 0


def test_mask_biggest_region_keeps_largest():
    img = np.zeros((200, 200), dtype=np.uint8)
    # Large region
    cv2.rectangle(img, (10, 10), (100, 100), 255, -1)
    # Small region
    cv2.rectangle(img, (150, 150), (160, 160), 255, -1)

    result = mask.mask_biggest_region(img)

    assert result.shape == img.shape
    # The small region at (150,150)-(160,160) should be gone
    assert result[155, 155] == 0
    # The large region should still be present
    assert result[50, 50] == 255


def test_mask_biggest_region_all_black():
    img = np.zeros((100, 100), dtype=np.uint8)
    result = mask.mask_biggest_region(img)
    assert result.shape == img.shape
    assert result.sum() == 0


def test_mask_biggest_region_same_shape():
    img = np.zeros((50, 80), dtype=np.uint8)
    cv2.circle(img, (25, 40), 10, 255, -1)
    result = mask.mask_biggest_region(img)
    assert result.shape == (50, 80)


def test_mask_biggest_region_small_contour_filtered():
    # A 2x2 white square produces a contour with <= 5 points, which the
    # implementation filters out via `len(c) > 5`. Result should be all-black.
    img = np.zeros((50, 50), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (11, 11), 255, -1)
    result = mask.mask_biggest_region(img)
    assert result.shape == img.shape
    assert result.sum() == 0


# -----------------------------------------------------------------------------
# guess_mask_otsu()
# -----------------------------------------------------------------------------


def test_guess_mask_otsu_bright_object_on_dark():
    # Create a BGR image with a bright white circle on black background
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, (255, 255, 255), -1)
    result = mask.guess_mask_otsu(img)
    assert result.ndim == 2
    assert result.dtype == np.uint8
    # Centre pixel should be in the detected region
    assert result[50, 50] == 255


def test_guess_mask_otsu_uniform_image():
    # Uniform image - Otsu threshold on a uniform image produces all-zero mask
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = mask.guess_mask_otsu(img)
    assert result.ndim == 2
    assert result.dtype == np.uint8


def test_guess_mask_otsu_output_single_channel():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (40, 40), (200, 200, 200), -1)
    result = mask.guess_mask_otsu(img)
    assert result.ndim == 2
    assert result.dtype == np.uint8


# -----------------------------------------------------------------------------
# Copyright 2026 Angelos Evripiotis.
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
