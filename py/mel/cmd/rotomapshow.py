"""Render one or more data files as text."""


import argparse
import json
import shutil

import mel.rotomap.format


def setup_parser(parser):
    parser.add_argument(
        'FILE',
        type=argparse.FileType(),
        nargs='+',
        help="Path to the rotomap json file.")
    parser.add_argument(
        '--display-width',
        type=int,
        default=shutil.get_terminal_size()[0],
        help="Display maps side-by-side, wrapped to this width.")


def process_args(args):
    mole_map_list = [json.load(x) for x in args.FILE]

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
