"""Tests for mel.lib.moleimaging module."""

import math

import cv2
import numpy as np
import pytest

import mel.lib.moleimaging

# --- log10_zero ---


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0, 0),
        (1, 0),
        (10, 1),
        (100, 2),
        (0.1, -1),
        (0.01, -2),
        (1000, 3),
    ],
)
def test_log10_zero(x, expected):
    assert mel.lib.moleimaging.log10_zero(x) == pytest.approx(expected)


def test_log10_zero_fractional():
    result = mel.lib.moleimaging.log10_zero(5)
    assert result == pytest.approx(math.log10(5))


# --- point_to_int_point ---


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((1.7, 2.3), (1, 2)),
        ((0, 0), (0, 0)),
        ((-1.9, -2.1), (-1, -2)),
        ((3, 4), (3, 4)),
    ],
)
def test_point_to_int_point(point, expected):
    assert mel.lib.moleimaging.point_to_int_point(point) == expected


def test_point_to_int_point_numpy():
    point = np.array([1.7, 2.3])
    result = mel.lib.moleimaging.point_to_int_point(point)
    assert result == (1, 2)


# --- rotate_point_around_pivot ---


def test_rotate_zero_degrees():
    point = (10.0, 5.0)
    pivot = (0.0, 0.0)
    result = mel.lib.moleimaging.rotate_point_around_pivot(point, pivot, 0)
    assert result[0] == pytest.approx(10.0)
    assert result[1] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("degrees", "expected"),
    [
        (90, (0.0, 1.0)),
        (180, (-1.0, 0.0)),
        (270, (0.0, -1.0)),
    ],
)
def test_rotate_cardinal_around_origin(degrees, expected):
    point = (1.0, 0.0)
    pivot = (0.0, 0.0)
    result = mel.lib.moleimaging.rotate_point_around_pivot(point, pivot, degrees)
    assert result[0] == pytest.approx(expected[0], abs=1e-9)
    assert result[1] == pytest.approx(expected[1], abs=1e-9)


def test_rotate_around_arbitrary_pivot():
    point = (3.0, 0.0)
    pivot = (2.0, 0.0)
    result = mel.lib.moleimaging.rotate_point_around_pivot(point, pivot, 90)
    assert result[0] == pytest.approx(2.0, abs=1e-9)
    assert result[1] == pytest.approx(1.0, abs=1e-9)


def test_rotate_360_roundtrip():
    point = (7.5, 3.2)
    pivot = (1.0, -2.0)
    result = mel.lib.moleimaging.rotate_point_around_pivot(point, pivot, 360)
    assert result[0] == pytest.approx(point[0], abs=1e-9)
    assert result[1] == pytest.approx(point[1], abs=1e-9)


# --- MoleAcquirer ---


def test_mole_acquirer_initial_state():
    acq = mel.lib.moleimaging.MoleAcquirer()
    assert acq.is_locked is False


def test_mole_acquirer_update_none_stays_unlocked():
    acq = mel.lib.moleimaging.MoleAcquirer()
    acq.update(None)
    assert acq.is_locked is False
    acq.update(None)
    assert acq.is_locked is False


def test_mole_acquirer_identical_stats_locks():
    acq = mel.lib.moleimaging.MoleAcquirer()
    stats = (50.0, 80.0, 90.0, 10.0, 20.0, 30.0)
    # MoleAcquirer uses lerp(0.5) exponential smoothing on stat diffs, so
    # identical inputs drive the smoothed diff to zero. ~20 iterations suffice
    # but 100 gives a comfortable margin.
    for _ in range(100):
        acq.update(stats)
    assert acq.is_locked is True


def test_mole_acquirer_varying_stats_unlocks():
    acq = mel.lib.moleimaging.MoleAcquirer()
    stats = (50.0, 80.0, 90.0, 10.0, 20.0, 30.0)
    # Lock it first.
    for _ in range(100):
        acq.update(stats)
    assert acq.is_locked is True

    # Feed very different stats to unlock.
    different = (500.0, 800.0, 900.0, 100.0, 200.0, 300.0)
    acq.update(different)
    assert acq.is_locked is False


# --- biggest_contour ---


def test_biggest_contour_single():
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(image, (50, 50), 20, 255, -1)
    contour = mel.lib.moleimaging.biggest_contour(image)
    assert contour is not None
    assert len(contour) > 5


def test_biggest_contour_returns_largest():
    image = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(image, (50, 50), 10, 255, -1)  # small
    cv2.circle(image, (150, 150), 40, 255, -1)  # large
    contour = mel.lib.moleimaging.biggest_contour(image)
    area = cv2.contourArea(contour)
    # The bigger circle has radius 40 -> area ~ pi*40^2 ~ 5027
    assert area > 3000


def test_biggest_contour_empty_image_raises():
    image = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(Exception, match="No contours found"):
        mel.lib.moleimaging.biggest_contour(image)


# --- find_mole_contour ---


def test_find_mole_contour_empty_list():
    contour, area = mel.lib.moleimaging.find_mole_contour([], (100, 100))
    assert contour is None
    assert area is None


def test_find_mole_contour_prefers_center():
    image = np.zeros((200, 200), dtype=np.uint8)
    # Circle near center.
    cv2.circle(image, (100, 100), 15, 255, -1)
    # Circle far from center.
    cv2.circle(image, (10, 10), 15, 255, -1)
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    assert len(contours) == 2

    contour, _area = mel.lib.moleimaging.find_mole_contour(contours, (200, 200))
    assert contour is not None
    # The returned contour should be the one closer to center.
    moments = cv2.moments(contour)
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    # Should be near (100, 100), not (10, 10).
    assert abs(cx - 100) < 20
    assert abs(cy - 100) < 20


def test_find_mole_contour_large_overrides_distance():
    image = np.zeros((300, 300), dtype=np.uint8)
    # Small circle near center.
    cv2.circle(image, (150, 150), 5, 255, -1)
    # Very large circle far from center.
    cv2.circle(image, (50, 50), 40, 255, -1)
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    contour, area = mel.lib.moleimaging.find_mole_contour(contours, (300, 300))
    assert contour is not None
    # The large contour should have area >> the small one. The logic selects the
    # large one when area > 10 * mole_area, but order matters. If the center one
    # is found first, the far one needs 10x area to override. The far circle
    # (r=40) has area ~5027 vs small (r=5) ~78, so 5027 > 780.
    assert area > 1000
    # Verify the returned contour is the large one near (50, 50).
    moments = cv2.moments(contour)
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    assert abs(cx - 50) < 10
    assert abs(cy - 50) < 10


# --- calc_hist ---


def test_calc_hist_sums_to_100():
    rng = np.random.default_rng()
    image = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = mel.lib.moleimaging.calc_hist(hsv, 1, None)
    assert len(hist) == 8
    assert sum(hist) == pytest.approx(100.0)


def test_calc_hist_with_mask():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    hist = mel.lib.moleimaging.calc_hist(hsv, 1, mask)
    assert len(hist) == 8
    assert sum(hist) == pytest.approx(100.0)


# --- find_mole_ellipse (smoke tests) ---


def test_find_mole_ellipse_with_mole_image():
    """Test find_mole_ellipse with an image that contains a detectable mole."""
    # Create a simple test image with a dark spot (simulating a mole)
    image = np.full((100, 100, 3), 200, dtype=np.uint8)  # Light gray background

    # Add a dark circular region in the center
    center_y, center_x = 50, 50
    y, x = np.ogrid[:100, :100]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= 10**2
    image[mask] = [50, 50, 50]  # Dark gray mole

    centre = np.array([50, 50], dtype=int)
    radius = 20

    # Should process without error
    result = mel.lib.moleimaging.find_mole_ellipse(image, centre, radius)
    # Result could be None or a valid ellipse tuple
    assert result is None or isinstance(result, tuple)


def test_find_mole_ellipse_parameter_types():
    """Test that find_mole_ellipse handles different input types correctly."""
    # Test image
    image = np.zeros((50, 50, 3), dtype=np.uint8)

    # Test with numpy array centre (correct type)
    centre_array = np.array([25, 25], dtype=int)
    result = mel.lib.moleimaging.find_mole_ellipse(image, centre_array, 10)
    assert result is None or isinstance(result, tuple)

    # Test with tuple centre (might be passed incorrectly)
    centre_tuple = (25, 25)
    try:
        result = mel.lib.moleimaging.find_mole_ellipse(image, centre_tuple, 10)
        # Should handle gracefully
        assert result is None or isinstance(result, tuple)
    except (TypeError, ValueError):
        # If it fails with tuple input, that's also acceptable
        pass


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
