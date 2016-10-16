"""Shatter rotomap images into many small fragments, for training networks."""


import argparse
import json
import shutil

import mel.rotomap.format


def setup_parser(parser):
    parser.add_argument(
        'FILE',
        nargs='+',
        help="Path to the rotomap image files.")


def process_args(args):
    pass
