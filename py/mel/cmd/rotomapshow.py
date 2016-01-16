"""Render one or more data files as text."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import math

import mel.lib.math


def setup_parser(parser):
    parser.add_argument(
        'FILE',
        type=argparse.FileType(),
        nargs='+',
        help="Path to the rotomap json file.")


def process_args(args):
    mole_map_list = [json.load(x) for x in args.FILE]

    uuid_set = set()
    for mole_map in mole_map_list:
        for mole in mole_map:
            uuid_set.add(mole['uuid'])

    max_digits = 1
    uuid_list = sorted(list(uuid_set))
    prev_uuid = uuid_list[0]
    prev_digits = max_digits
    uuid_to_display = {}
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

    for mole_map in mole_map_list:
        for mole in mole_map:
            mole['uuid'] = uuid_to_display[mole['uuid']]

    for mole_map in mole_map_list:
        grid = map_to_grid(mole_map, max_digits)

        for row in grid:
            for col in row:
                if max_digits > 1:
                    print(col, '', end='')
                else:
                    print(col, end='')
            print()
        print()


def map_to_grid(mole_map, num_digits):

    minx = min([m['x'] for m in mole_map])
    miny = min([m['y'] for m in mole_map])
    maxx = max([m['x'] for m in mole_map])
    maxy = max([m['y'] for m in mole_map])

    extents_x = maxx - minx
    extents_y = maxy - miny

    scale_x = 1 / extents_x
    scale_y = 1 / extents_y
    scale = min(scale_x, scale_y)

    any_collisions = True
    while any_collisions:
        any_collisions = False

        try:
            grid = make_grid(
                mole_map, minx, miny, extents_x, extents_y, scale, num_digits)
        except ValueError:
            any_collisions = True
            scale *= 2

    return grid


def make_grid(mole_map, left, top, width, height, scale, num_digits):
    grid_width = int(math.ceil(width * scale))
    grid_height = int(math.ceil(height * scale))
    grid = []

    spacer = '.' * num_digits

    for j in xrange(grid_height):
        row = []
        for i in xrange(grid_width):
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
