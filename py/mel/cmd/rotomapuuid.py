"""List the uuids of moles that match a prefix, from a list of json files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json


def setup_parser(parser):
    parser.add_argument(
        'PREFIX',
        help="Prefix to find the full id of.")
    parser.add_argument(
        'FILE',
        type=argparse.FileType(),
        nargs='+',
        help="Path to the rotomap json file.")


def process_args(args):
    mole_map_list = [json.load(x) for x in args.FILE]
    uuid_set = mole_uuid_set_from_map_list(mole_map_list)
    results = []

    for mole_uuid in uuid_set:
        if mole_uuid.startswith(args.PREFIX):
            results.append(mole_uuid)

    if results:
        print('\n'.join(results))
        return 0
    else:
        return 1


def mole_uuid_set_from_map_list(mole_map_list):
    uuid_set = set()
    for mole_map in mole_map_list:
        for mole in mole_map:
            uuid_set.add(mole['uuid'])
    return uuid_set
