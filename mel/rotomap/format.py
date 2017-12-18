"""Format rotomaps as text."""

import math

import mel.lib.math


def make_grid_row(row, max_digits):
    if max_digits > 1:
        out = ' '.join(row)
    else:
        out = ''.join(row)
    return out


def print_grids_wrapped(grid_list, display_width, max_digits):
    row_list_list = []
    for grid in grid_list:
        row_list = []
        for row in grid:
            row_list.append(make_grid_row(row, max_digits))
        row_list_list.append(row_list)

    row_lists_to_display = []
    width = 0
    spacer = '    '
    for row_list in row_list_list:
        row_width = len(row_list[0])
        if row_lists_to_display:
            new_width = width + row_width + len(spacer)
        else:
            new_width = row_width

        if new_width <= display_width:
            width = new_width
            row_lists_to_display.append(row_list)
        else:
            if not row_lists_to_display:
                raise Exception(
                    'Could not fit grid of width {} '
                    'to display of width {}'.format(
                        row_width, display_width))

            print_row_lists_in_columns(
                row_lists_to_display, max_digits, spacer)
            row_lists_to_display = []
            width = 0
            print()

    if row_lists_to_display:
        print_row_lists_in_columns(
            row_lists_to_display, max_digits, spacer)


def print_row_lists_in_columns(row_lists_to_display, max_digits, spacer):
    max_rows = max(len(g) for g in row_lists_to_display)
    for i in range(max_rows):
        row = []
        for row_list in row_lists_to_display:
            if i < len(row_list):
                row.append(row_list[i])
            else:
                width = len(row_list[0])
                row.append(' ' * width)
        print(spacer.join(row))


def mole_uuid_set_from_map_list(mole_map_list):
    uuid_set = set()
    for mole_map in mole_map_list:
        for mole in mole_map:
            uuid_set.add(mole['uuid'])
    return uuid_set


def calc_uuid_display_params(uuid_set):
    max_digits = 1
    uuid_list = sorted(list(uuid_set))
    prev_uuid = uuid_list[0]
    prev_digits = max_digits
    uuid_to_display = {}
    digits = 1
    for this_uuid in uuid_list[1:]:
        digits = 1
        for i, j in zip(prev_uuid, this_uuid):
            if i == j:
                digits += 1
            else:
                break

        unsafe_digits = max(prev_digits, digits)
        safe_digits = len(prev_uuid) - unsafe_digits
        display = prev_uuid[:unsafe_digits] + '.' * safe_digits
        uuid_to_display[prev_uuid] = display

        max_digits = max(digits, max_digits)
        prev_uuid = this_uuid
        prev_digits = digits

    unsafe_digits = max(prev_digits, digits)
    safe_digits = len(prev_uuid) - unsafe_digits
    display = prev_uuid[:unsafe_digits] + '.' * safe_digits
    uuid_to_display[prev_uuid] = display

    return max_digits, uuid_to_display


def map_to_grid(mole_map, num_digits):

    if not mole_map:
        return [['.']]

    minx = min([m['x'] for m in mole_map])
    miny = min([m['y'] for m in mole_map])
    maxx = max([m['x'] for m in mole_map])
    maxy = max([m['y'] for m in mole_map])

    extents_x = max(maxx - minx, 1)
    extents_y = max(maxy - miny, 1)

    scale_x = 1 / extents_x
    scale_y = 1 / extents_y
    initial_scale = min(scale_x, scale_y)
    scale = initial_scale

    any_collisions = True
    while any_collisions:
        any_collisions = False

        try:
            grid = make_grid(
                mole_map, minx, miny, extents_x, extents_y, scale, num_digits)
        except ValueError:
            any_collisions = True
            scale += initial_scale

    return grid


def make_grid(mole_map, left, top, width, height, scale, num_digits):
    grid_width = int(math.ceil(width * scale))
    grid_height = int(math.ceil(height * scale))
    grid = []

    spacer = '.' * num_digits

    for _ in range(grid_height):
        row = []
        for _ in range(grid_width):
            row.append(spacer)
        grid.append(row)

    for mole in mole_map:
        x = mel.lib.math.clamp(
            int((mole['x'] - left) * scale), 0, grid_width - 1)
        y = mel.lib.math.clamp(
            int((mole['y'] - top) * scale), 0, grid_height - 1)
        if grid[y][x] != spacer:
            raise ValueError('Collision when rendering grid')
        grid[y][x] = mole['uuid'][0:num_digits]

    return grid
# -----------------------------------------------------------------------------
# Copyright (C) 2016-2017 Angelos Evripiotis.
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
