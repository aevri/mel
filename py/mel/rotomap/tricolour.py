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
        raise ValueError("n must be zero or greater, got {}.".format(n))
    if n > len(list_):
        raise ValueError("n must be less than list len ({}), got {}.".format(
            len(list_), n))
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
                yield tuple(
                    _list_rotated_left(
                        [colour1, colour2, colour2], band))

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
            len(self._palette))

    def _ensure_uuid(self, uuid_):
        if uuid_ in self._uuid_to_colours:
            return
        indices = next(self._triband_mapping)
        self._uuid_to_colours[uuid_] = tuple(
            self._palette[x] for x in indices
        )

    def __call__(self, uuid_):
        self._ensure_uuid(uuid_)
        return self._uuid_to_colours[uuid_]
