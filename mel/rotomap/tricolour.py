"""Pick colours for moles, based on a tricolour scheme."""

# "9 class set1" from http://colorbrewer2.org/
# This set seems to be the most colour-blind friendly for 9 colours
_NINE_CLASS_SET1 = [
    (228, 26, 28),
    (55, 126, 184),
    (77, 175, 74),
    (152, 78, 163),
    (255, 127, 0),
    (255, 255, 51),
    (166, 86, 40),
    (247, 129, 191),
    (153, 153, 153),
]


def hex3_to_rgb4(hex_string):
    # "12 class paired" from http://colorbrewer2.org/
    scheme = [
        (166, 206, 227),
        (31, 120, 180),
        (178, 223, 138),
        (51, 160, 44),
        (251, 154, 153),
        (227, 26, 28),
        (253, 191, 111),
        (255, 127, 0),
        (202, 178, 214),
        (106, 61, 154),
        (255, 255, 153),
        (177, 89, 40),
    ]

    rgb_list = []

    value = int(hex_string[0:3], 16)
    for _ in range(4):
        index = value % 12
        value //= 12
        rgb_list.append(list(reversed(scheme[index])))

    return rgb_list


def uuid_to_tricolour_first_digits(uuid_):
    return hex3_to_rgb4(uuid_[:3])


def _list_rotated_left(list_, n):
    """Return the input 'list_', rotated left by n places.

    'n' must be between zero and len(list_).

    Usage examples:

        >>> _list_rotated_left([1, 2, 3], 0)
        [1, 2, 3]
        >>> _list_rotated_left([1, 2, 3], 1)
        [2, 3, 1]
        >>> _list_rotated_left([1, 2, 3], 2)
        [3, 1, 2]

    :list_: a list.
    :n: the number of places to rotate left.
    :returns: a new list.
    """
    if n < 0:
        raise ValueError(f"n must be zero or greater, got {n}.")
    if n > len(list_):
        raise ValueError(f"n must be less than list len ({len(list_)}), got {n}.")
    return list_[n:] + list_[:n]


def yield_triband_mapping_in_distinctive_order(num_colours):
    # Assume that pure colours are better than mixed.
    # All bands are the same colour.
    for i in range(num_colours):
        yield (i, i, i)

    # Assume that position is more distinctive than colour difference.
    # Vary band position before colour.
    for colour1 in range(num_colours):
        for colour2 in range(num_colours):
            if colour1 == colour2:
                continue
            for band in range(3):
                yield tuple(_list_rotated_left([colour1, colour2, colour2], band))

    yield from _yield_tricolours_no_repeats(num_colours)


def _yield_tricolours_no_repeats(num_colours):
    for colour1 in range(num_colours):
        for colour2 in range(num_colours):
            if colour1 == colour2:
                continue
            for colour3 in range(num_colours):
                if colour1 == colour3 or colour2 == colour3:
                    continue
                yield (colour1, colour2, colour3)


class UuidTriColourPicker:
    def __init__(self):
        self._uuid_to_colours = {}
        self._palette = _NINE_CLASS_SET1
        self._triband_mapping = yield_triband_mapping_in_distinctive_order(
            len(self._palette)
        )

    def _ensure_uuid(self, uuid_):
        if uuid_ in self._uuid_to_colours:
            return
        try:
            indices = next(self._triband_mapping)
        except StopIteration:
            # We have run out of colours, just paint everything in danger
            # colours.
            red = (0, 0, 255)
            yellow = (0, 255, 255)
            self._uuid_to_colours[uuid_] = (red, yellow, red)
        else:
            self._uuid_to_colours[uuid_] = tuple(self._palette[x] for x in indices)

    def __call__(self, uuid_):
        self._ensure_uuid(uuid_)
        return self._uuid_to_colours[uuid_]


# -----------------------------------------------------------------------------
# Copyright (C) 2016-2018 Angelos Evripiotis.
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
