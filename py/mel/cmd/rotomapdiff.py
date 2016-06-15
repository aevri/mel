"""Render the differences between two data files as text."""


import argparse
import copy
import json
import shutil
import sys

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
    parser.add_argument(
        '--color',
        choices=['on', 'off', 'auto'],
        default='auto',
        help='Control the use of terminal colors. Defaults to "auto".')


def process_args(args):
    mole_map_list = [load_mole(x) for x in args.FILE]

    new_map = diff_maps(mole_map_list[0], mole_map_list[1])

    uuid_set = {m['uuid'] for m in new_map}

    max_digits, uuid_to_display = mel.rotomap.format.calc_uuid_display_params(
        uuid_set)

    for mole in new_map:
        mole['uuid'] = uuid_to_display[mole['uuid']]

    use_color = args.color == 'on'
    if args.color == 'auto' and is_stdout_a_tty():
        use_color = True

    grid = mel.rotomap.format.map_to_grid(new_map, max_digits)
    for row in grid:
        if use_color:
            for i, item in enumerate(row):
                if item.startswith('-'):
                    row[i] = "\033[31m{0}\033[00m".format(item)
                elif item.startswith('+'):
                    row[i] = "\033[32m{0}\033[00m".format(item)
                elif item.startswith('*'):
                    row[i] = "\033[34m{0}\033[00m".format(item)

        print(' '.join(row))


def load_mole(file_):
    try:
        return json.load(file_)
    except json.decoder.JSONDecodeError:
        return []


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


def is_stdout_a_tty():
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
