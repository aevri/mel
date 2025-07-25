"""Math-related things."""

import math

import numpy

RADS_TO_DEGS = 180 / math.pi


def lerp(origin, target, factor_0_to_1):
    towards = target - origin
    return origin + (towards * factor_0_to_1)


def distance_sq_2d(a, b):
    """Return the squared distance between two points in two dimensions.

    Usage examples:
        >>> distance_sq_2d((1, 1), (1, 1))
        0

        >>> distance_sq_2d((0, 0), (0, 2))
        4
    """
    assert len(a) == 2
    assert len(b) == 2
    x = a[0] - b[0]
    y = a[1] - b[1]
    return (x * x) + (y * y)


def distance_2d(a, b):
    """Return the squared distance between two points in two dimensions.

    Usage examples:
        >>> distance_2d((1, 1), (1, 1))
        0.0

        >>> distance_2d((0, 0), (0, 2))
        2.0
    """
    return math.sqrt(distance_sq_2d(a, b))


def normalized(v):
    """Return vector v normalized to unit length.

    Usage examples:
        >>> normalized((0, 2))
        (0.0, 1.0)
    """
    inv_length = 1 / distance_2d((0, 0), v)
    return (v[0] * inv_length, v[1] * inv_length)


def angle(v):
    """Return the angle between v and 'right'.

    Usage examples:
        >>> angle((1, 0))
        0.0

        >>> angle((-1, 0))
        180.0

        >>> angle((0, 1))
        -90.0

        >>> angle((0, -1))
        90.0
    """
    cos_theta = normalized(v)[0]
    theta = math.acos(cos_theta)
    if v[1] > 0:
        theta = -theta
    return rads_to_degs(theta)


def rads_to_degs(theta):
    return theta * RADS_TO_DEGS


def raise_if_not_int_vector2(v):
    if not isinstance(v, numpy.ndarray):
        raise ValueError(f"{v}:{v!r}:{type(v)} is not a numpy array")
    if not numpy.issubdtype(v.dtype.type, numpy.integer):
        raise ValueError(f"{v}:{v.dtype} is not an int vector2")


# -----------------------------------------------------------------------------
# Copyright (C) 2015-2020 Angelos Evripiotis.
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
