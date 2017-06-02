"""List the uuids of moles in supplied files."""


import argparse
import json


def setup_parser(parser):
    parser.add_argument(
        'FILE',
        type=argparse.FileType(),
        nargs='+',
        help="Path to the rotomap json file.")


def process_args(args):
    path_data_list = [(x.name, json.load(x)) for x in args.FILE]
    for path, data in path_data_list:
        for mole in data:
            print(mole["uuid"], path)
