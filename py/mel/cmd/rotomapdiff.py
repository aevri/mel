"""Render the differences between two data files as text."""


import argparse
import copy
import json
import shutil

import mel.rotomap.format


def setup_parser(parser):
    parser.add_argument(
        'FILE',
        type=argparse.FileType(),
        nargs=2,
        help="Path to the rotomap json file.")
    parser.add_argument(
        '--display-width',
        type=int,
        default=shutil.get_terminal_size()[0],
        help="Display maps side-by-side, wrapped to this width.")


def process_args(args):
    mole_map_list = [json.load(x) for x in args.FILE]

    new_map = diff_maps(mole_map_list[0], mole_map_list[1])
    mole_map_list = [new_map]

    uuid_set = mel.rotomap.format.mole_uuid_set_from_map_list(mole_map_list)

    max_digits, uuid_to_display = mel.rotomap.format.calc_uuid_display_params(
        uuid_set)

    for mole_map in mole_map_list:
        for mole in mole_map:
            mole['uuid'] = uuid_to_display[mole['uuid']]

    grid_list = [
        mel.rotomap.format.map_to_grid(m, max_digits) for m in mole_map_list
    ]

    mel.rotomap.format.print_grids_wrapped(
        grid_list, args.display_width, max_digits)


def diff_maps(src, dst):
    src_map = {m['uuid']: m for m in src}
    dst_map = {m['uuid']: m for m in dst}

    src_uuids = {u for u in src_map}
    dst_uuids = {u for u in dst_map}

    added_uuids = dst_uuids - src_uuids
    removed_uuids = src_uuids - dst_uuids
    both_uuids = src_uuids.intersection(dst_uuids)

    new = []

    for u in removed_uuids:
        m = copy.deepcopy(src_map[u])
        m['uuid'] = '-' + m['uuid']
        new.append(m)

    for u in added_uuids:
        m = copy.deepcopy(dst_map[u])
        m['uuid'] = '+' + m['uuid']
        new.append(m)

    for u in both_uuids:
        m1 = src_map[u]
        m2 = dst_map[u]
        if m1 == m2:
            new.append(m1)
        else:
            if m1['x'] == m2['x'] and m1['y'] == m2['y']:
                m = copy.deepcopy(src_map[u])
                m['uuid'] = '*' + m['uuid']
                new.append(m)
            else:
                m1 = copy.deepcopy(src_map[u])
                m2 = copy.deepcopy(dst_map[u])
                m1['uuid'] = '-' + m1['uuid']
                m2['uuid'] = '+' + m2['uuid']
                new.append(m1)
                new.append(m2)

    return new
