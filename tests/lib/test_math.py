"""Test suite for mel.lib.math."""

# Generated with assistance from Claude Code.

import math

import numpy as np
import pytest

import mel.lib.math as mlmath


class TestLerp:
    def test_factor_zero_returns_origin(self):
        assert mlmath.lerp(0, 10, 0) == 0

    def test_factor_one_returns_target(self):
        assert mlmath.lerp(0, 10, 1) == 10

    def test_factor_half_returns_midpoint(self):
        assert mlmath.lerp(0, 10, 0.5) == 5.0

    def test_negative_values(self):
        assert mlmath.lerp(-10, 10, 0.5) == 0.0

    def test_factor_beyond_one_extrapolates(self):
        assert mlmath.lerp(0, 10, 2) == 20


class TestDistanceSq2d:
    def test_same_point(self):
        assert mlmath.distance_sq_2d((0, 0), (0, 0)) == 0

    def test_unit_distance(self):
        assert mlmath.distance_sq_2d((0, 0), (1, 0)) == 1

    def test_diagonal(self):
        assert mlmath.distance_sq_2d((0, 0), (3, 4)) == 25

    def test_negative_coords(self):
        assert mlmath.distance_sq_2d((-1, -1), (2, 3)) == 25

    def test_symmetric(self):
        a, b = (1, 2), (4, 6)
        assert mlmath.distance_sq_2d(a, b) == mlmath.distance_sq_2d(b, a)


class TestDistance2d:
    def test_same_point(self):
        assert mlmath.distance_2d((0, 0), (0, 0)) == 0.0

    def test_known_distance(self):
        assert mlmath.distance_2d((0, 0), (3, 4)) == 5.0

    def test_unit_distance(self):
        assert mlmath.distance_2d((0, 0), (1, 0)) == 1.0

    def test_negative_coords(self):
        assert mlmath.distance_2d((-3, 0), (0, 4)) == 5.0


class TestNormalized:
    def test_unit_x(self):
        assert mlmath.normalized((5, 0)) == (1.0, 0.0)

    def test_unit_y(self):
        assert mlmath.normalized((0, 3)) == (0.0, 1.0)

    def test_result_has_unit_length(self):
        result = mlmath.normalized((3, 4))
        length = math.sqrt(result[0] ** 2 + result[1] ** 2)
        assert length == pytest.approx(1.0)

    def test_negative_vector(self):
        result = mlmath.normalized((-3, -4))
        assert result[0] == pytest.approx(-0.6)
        assert result[1] == pytest.approx(-0.8)

    def test_zero_vector_raises(self):
        with pytest.raises(ZeroDivisionError):
            mlmath.normalized((0, 0))


class TestAngle:
    @pytest.mark.parametrize(
        "vec,expected",
        [
            ((1, 0), 0.0),
            ((-1, 0), 180.0),
            ((0, 1), -90.0),  # Positive y maps to -90 degrees.
            ((0, -1), 90.0),
        ],
    )
    def test_cardinal_directions(self, vec, expected):
        assert mlmath.angle(vec) == pytest.approx(expected)


class TestRadsToDegs:
    def test_zero(self):
        assert mlmath.rads_to_degs(0) == 0.0

    def test_pi(self):
        assert mlmath.rads_to_degs(math.pi) == pytest.approx(180.0)

    def test_half_pi(self):
        assert mlmath.rads_to_degs(math.pi / 2) == pytest.approx(90.0)

    def test_negative(self):
        assert mlmath.rads_to_degs(-math.pi) == pytest.approx(-180.0)


class TestRaiseIfNotIntVector2:
    def test_valid_int_array(self):
        v = np.array([1, 2], dtype=np.int64)
        mlmath.raise_if_not_int_vector2(v)  # Should not raise.

    def test_float_array_raises(self):
        v = np.array([1.0, 2.0], dtype=np.float64)
        with pytest.raises(ValueError, match="not an int vector2"):
            mlmath.raise_if_not_int_vector2(v)

    def test_non_array_raises(self):
        with pytest.raises(ValueError, match="not a numpy array"):
            mlmath.raise_if_not_int_vector2([1, 2])

    def test_tuple_raises(self):
        with pytest.raises(ValueError, match="not a numpy array"):
            mlmath.raise_if_not_int_vector2((1, 2))

    def test_int32_valid(self):
        v = np.array([1, 2], dtype=np.int32)
        mlmath.raise_if_not_int_vector2(v)  # Should not raise.


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
