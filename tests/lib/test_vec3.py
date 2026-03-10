"""Test suite for mel.lib.vec3."""

# Generated with assistance from Claude Code.

import warnings

import numpy as np
import pytest

import mel.lib.vec3 as vec3


class TestMake:
    def test_shape(self):
        v = vec3.make(1, 2, 3)
        assert v.shape == (1, 3)

    def test_values(self):
        v = vec3.make(4, 5, 6)
        assert v[0, 0] == 4
        assert v[0, 1] == 5
        assert v[0, 2] == 6

    def test_negative_values(self):
        v = vec3.make(-1, -2, -3)
        np.testing.assert_array_equal(v, np.array([[-1, -2, -3]]))


class TestIsVec3:
    def test_valid_single(self):
        assert vec3.is_vec3(vec3.make(1, 2, 3))

    def test_valid_multiple(self):
        v = np.array([[1, 2, 3], [4, 5, 6]])
        assert vec3.is_vec3(v)

    def test_1d_array_invalid(self):
        assert not vec3.is_vec3(np.array([1, 2, 3]))

    def test_wrong_columns_invalid(self):
        assert not vec3.is_vec3(np.array([[1, 2]]))

    def test_empty_rows_invalid(self):
        v = np.zeros((0, 3))
        assert not vec3.is_vec3(v)


class TestZeros:
    def test_single(self):
        v = vec3.zeros()
        assert v.shape == (1, 3)
        np.testing.assert_array_equal(v, np.array([[0.0, 0.0, 0.0]]))

    def test_multiple(self):
        v = vec3.zeros(3)
        assert v.shape == (3, 3)
        assert np.all(v == 0.0)


class TestCount:
    def test_single(self):
        assert vec3.count(vec3.make(1, 2, 3)) == 1

    def test_multiple(self):
        v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert vec3.count(v) == 3


class TestColumnAccessors:
    def test_xcol(self):
        v = vec3.make(1, 2, 3)
        np.testing.assert_array_equal(vec3.xcol(v), np.array([[1]]))

    def test_ycol(self):
        v = vec3.make(1, 2, 3)
        np.testing.assert_array_equal(vec3.ycol(v), np.array([[2]]))

    def test_zcol(self):
        v = vec3.make(1, 2, 3)
        np.testing.assert_array_equal(vec3.zcol(v), np.array([[3]]))

    def test_columns_broadcast(self):
        v = np.array([[1, 2, 3], [4, 5, 6]])
        assert vec3.xcol(v).shape == (2, 1)
        assert vec3.ycol(v).shape == (2, 1)
        assert vec3.zcol(v).shape == (2, 1)


class TestScalarAccessors:
    def test_xval(self):
        assert vec3.xval(vec3.make(10, 20, 30)) == 10

    def test_yval(self):
        assert vec3.yval(vec3.make(10, 20, 30)) == 20

    def test_zval(self):
        assert vec3.zval(vec3.make(10, 20, 30)) == 30

    def test_returns_plain_int(self):
        result = vec3.xval(vec3.make(5, 6, 7))
        assert isinstance(result, int)


class TestDot:
    def test_same_vector(self):
        x = vec3.make(1, 0, 0)
        np.testing.assert_array_equal(vec3.dot(x, x), np.array([[1]]))

    def test_orthogonal_vectors(self):
        x = vec3.make(1, 0, 0)
        y = vec3.make(0, 1, 0)
        np.testing.assert_array_equal(vec3.dot(x, y), np.array([[0]]))

    def test_known_dot_product(self):
        a = vec3.make(1, 2, 3)
        b = vec3.make(4, 5, 6)
        # 1*4 + 2*5 + 3*6 = 32
        np.testing.assert_array_equal(vec3.dot(a, b), np.array([[32]]))

    def test_batch_dot(self):
        xyz = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = vec3.dot(xyz, xyz)
        np.testing.assert_array_equal(result, np.array([[1], [1], [1]]))

    def test_commutative(self):
        a = vec3.make(1, 2, 3)
        b = vec3.make(4, 5, 6)
        np.testing.assert_array_equal(vec3.dot(a, b), vec3.dot(b, a))


class TestMagSq:
    def test_unit_vector(self):
        np.testing.assert_array_equal(
            vec3.mag_sq(vec3.make(1, 0, 0)), np.array([[1]])
        )

    def test_scaled_vector(self):
        np.testing.assert_array_equal(
            vec3.mag_sq(vec3.make(2, 0, 0)), np.array([[4]])
        )

    def test_general_vector(self):
        # 3^2 + 4^2 + 0^2 = 25
        np.testing.assert_array_equal(
            vec3.mag_sq(vec3.make(3, 4, 0)), np.array([[25]])
        )


class TestMag:
    def test_unit_vectors(self):
        for v in [vec3.make(1, 0, 0), vec3.make(0, 1, 0), vec3.make(0, 0, 1)]:
            np.testing.assert_array_almost_equal(vec3.mag(v), np.array([[1.0]]))

    def test_known_magnitude(self):
        # sqrt(9 + 16) = 5
        v = vec3.make(3, 4, 0)
        np.testing.assert_array_almost_equal(vec3.mag(v), np.array([[5.0]]))

    def test_scaled_vector(self):
        v = vec3.make(2, 0, 0)
        np.testing.assert_array_almost_equal(vec3.mag(v), np.array([[2.0]]))


class TestNormalized:
    def test_unit_x(self):
        v = vec3.make(5, 0, 0)
        result = vec3.normalized(v)
        np.testing.assert_array_almost_equal(result, np.array([[1.0, 0.0, 0.0]]))

    def test_unit_y(self):
        v = vec3.make(0, 3, 0)
        result = vec3.normalized(v)
        np.testing.assert_array_almost_equal(result, np.array([[0.0, 1.0, 0.0]]))

    def test_result_has_unit_magnitude(self):
        v = vec3.make(1, 2, 3)
        result = vec3.normalized(v)
        np.testing.assert_array_almost_equal(vec3.mag(result), np.array([[1.0]]))

    def test_negative_vector(self):
        v = vec3.make(-4, 0, 0)
        result = vec3.normalized(v)
        np.testing.assert_array_almost_equal(result, np.array([[-1.0, 0.0, 0.0]]))

    def test_roundtrip_direction_preserved(self):
        v = vec3.make(3, 4, 5)
        n = vec3.normalized(v)
        # Scaling normalized back should give proportional vector.
        m = vec3.mag(v)[0, 0]
        np.testing.assert_array_almost_equal(n * m, v.astype(float))

    def test_zero_vector_produces_nan(self):
        v = np.array([[0, 0, 0]], dtype=float)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value", RuntimeWarning)
            result = vec3.normalized(v)
        assert np.all(np.isnan(result))


class TestMakeFromColumns:
    def test_basic(self):
        result = vec3.make_from_columns(
            np.array([1, 2]), np.array([3, 4]), np.array([5, 6])
        )
        expected = np.array([[1, 3, 5], [2, 4, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_single_element(self):
        result = vec3.make_from_columns(
            np.array([10]), np.array([20]), np.array([30])
        )
        expected = np.array([[10, 20, 30]])
        np.testing.assert_array_equal(result, expected)


class TestBatchOperations:
    """Test operations on arrays containing multiple vectors."""

    def test_mag_batch(self):
        v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = vec3.mag(v)
        np.testing.assert_array_almost_equal(
            result, np.array([[1.0], [1.0], [1.0]])
        )

    def test_dot_single_vs_batch(self):
        # Shape (1,3) dot (3,3) tests numpy broadcasting semantics.
        single = vec3.make(1, 0, 0)
        batch = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = vec3.dot(single, batch)
        np.testing.assert_array_equal(result, np.array([[1], [0], [0]]))

    def test_normalized_batch(self):
        v = np.array([[3, 0, 0], [0, 4, 0], [0, 0, 5]], dtype=float)
        result = vec3.normalized(v)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_array_almost_equal(result, expected)


class TestEdgeCases:
    """Edge cases and numerical precision tests."""

    @pytest.mark.parametrize(
        "v",
        [
            vec3.make(1, 0, 0),
            vec3.make(0, 1, 0),
            vec3.make(0, 0, 1),
            np.array([[1, 1, 1]], dtype=float),
            np.array([[3, 4, 5]], dtype=float),
        ],
    )
    def test_normalize_then_mag_is_one(self, v):
        n = vec3.normalized(v)
        np.testing.assert_array_almost_equal(vec3.mag(n), np.array([[1.0]]))

    def test_dot_product_sign(self):
        a = vec3.make(1, 0, 0)
        b = vec3.make(-1, 0, 0)
        result = vec3.dot(a, b)
        assert result[0, 0] == -1

    def test_large_values(self):
        v = vec3.make(1e6, 1e6, 1e6)
        n = vec3.normalized(v)
        np.testing.assert_array_almost_equal(vec3.mag(n), np.array([[1.0]]))


# -----------------------------------------------------------------------------
# Copyright (C) 2026 Angelos Evripiotis.
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
