"""Math-related things."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def lerp(origin, target, factor_0_to_1):
    towards = target - origin
    return origin + (towards * factor_0_to_1)


def clamp(x, x_min, x_max):
    """Return x confined between x_min and x_max inclusive.

    Usage examples:
        >>> clamp(-1, 0, 9)
        0

        >>> clamp(1, 0, 9)
        1

        >>> clamp(10, 0, 9)
        9

    """
    return max(min(x, x_max), x_min)


def distance_sq_2d(a, b):
    """Return the squared distance between two points in two dimensions.

    Usage examples:
        >>> distance_sq_2d((1, 1), (1, 1))
        0

        >>> distance_sq_2d((0, 0), (0, 2))
        4

    """
    x = a[0] - b[0]
    y = a[1] - b[1]
    return (x * x) + (y * y)
