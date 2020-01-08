"""Create and operate on numpy arrays of 3d vectors."""

import numpy as np


def normalized(v):
    assert is_vec3(v)
    return v / mag(v)


def mag(v):
    """Return the length of the supplied vector.

    >>> x = make(1, 0, 0)
    >>> mag(x)
    array([[1.]])
    >>> mag(x * 2)
    array([[2.]])

    >>> xyz = np.array([
    ...     [1, 0, 0],
    ...     [0, 1, 0],
    ...     [0, 0, 1],
    ... ])
    >>> mag(xyz)
    array([[1.],
           [1.],
           [1.]])
    >>> mag(xyz * 2)
    array([[2.],
           [2.],
           [2.]])
    """
    assert is_vec3(v)
    return np.sqrt(mag_sq(v))


def mag_sq(v):
    """Return the squared length of the supplied vector.

    >>> x = make(1, 0, 0)
    >>> mag_sq(x)
    array([[1]])
    >>> mag_sq(x * 2)
    array([[4]])

    >>> xyz = np.array([
    ...     [1, 0, 0],
    ...     [0, 1, 0],
    ...     [0, 0, 1],
    ... ])
    >>> mag_sq(xyz)
    array([[1],
           [1],
           [1]])
    >>> mag_sq(xyz * 2)
    array([[4],
           [4],
           [4]])
    """
    assert is_vec3(v)
    return dot(v, v)


def dot(a, b):
    """Return the dot products of all the vectors in x and y.

    >>> x = make(1, 0, 0)
    >>> dot(x, x)
    array([[1]])

    >>> xyz = np.array([
    ...     [1, 0, 0],
    ...     [0, 1, 0],
    ...     [0, 0, 1],
    ... ])

    >>> dot(x, xyz)
    array([[1],
           [0],
           [0]])

    >>> dot(xyz, x)
    array([[1],
           [0],
           [0]])

    >>> dot(xyz, xyz)
    array([[1],
           [1],
           [1]])
    """
    assert is_vec3(a)
    assert is_vec3(b)
    return (xcol(a) * xcol(b)) + (ycol(a) * ycol(b)) + (zcol(a) * zcol(b))


def xcol(v):
    """Return the x column of the supplied vectors.

    Ensure that they still broadcast.

    >>> xcol(make(1, 2, 3))
    array([[1]])
    """
    assert is_vec3(v)
    return v[:, 0:1]


def ycol(v):
    """Return the y column of the supplied vectors.

    Ensure that they still broadcast.

    >>> ycol(make(1, 2, 3))
    array([[2]])
    """
    assert is_vec3(v)
    return v[:, 1:2]


def zcol(v):
    """Return the z column of the supplied vectors.

    Ensure that they still broadcast.

    >>> zcol(make(1, 2, 3))
    array([[3]])
    """
    assert is_vec3(v)
    return v[:, 2:3]


def xval(v):
    """Return the scalar x value of a single vector.

    >>> xval(make(1, 2, 3))
    1
    """
    assert is_vec3(v)
    assert v.shape[0] == 1
    return v[0, 0]


def yval(v):
    """Return the scalar x value of a single vector.

    >>> yval(make(1, 2, 3))
    2
    """
    assert is_vec3(v)
    assert v.shape[0] == 1
    return v[0, 1]


def zval(v):
    """Return the scalar x value of a single vector.

    >>> zval(make(1, 2, 3))
    3
    """
    assert is_vec3(v)
    assert v.shape[0] == 1
    return v[0, 2]


def make_from_columns(x_array, y_array, z_array):
    """Given arrays for x, y, and z values, make them into an array of vectors.

    >>> x_array = np.array([1, 2])
    >>> y_array = np.array([3, 4])
    >>> z_array = np.array([5, 6])
    >>> make_from_columns(x_array, y_array, z_array)
    array([[1, 3, 5],
           [2, 4, 6]])
    """
    return np.column_stack([x_array, y_array, z_array])


def zeros(num_vectors=1):
    """Return an array of vectors set to zero.

    >>> zeros()
    array([[0., 0., 0.]])

    >>> zeros(2)
    array([[0., 0., 0.],
           [0., 0., 0.]])
    """
    return np.zeros((num_vectors, 3))


def count(v):
    """Return the number of vectors in 'v'.

    >>> count(zeros(1))
    1
    >>> count(zeros(2))
    2
    """
    assert is_vec3(v)
    return v.shape[0]


def is_vec3(v):
    return v.ndim == 2 and v.shape[0] > 0 and v.shape[1] == 3


def make(x, y, z):
    return np.array([[x, y, z]])


# -----------------------------------------------------------------------------
# Copyright (C) 2020 Angelos Evripiotis.
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
