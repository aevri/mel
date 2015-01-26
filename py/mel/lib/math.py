"""Math-related things."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def lerp(origin, target, factor_0_to_1):
    towards = target - origin
    return origin + (towards * factor_0_to_1)
