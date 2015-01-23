"""List the moles in a mole catalog."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


def setup_parser(parser):
    pass


def process_args(args):
    for mole_dir in _yield_mole_dirs('.'):
        print(mole_dir)


def _yield_mole_dirs(rootpath):
    for path, dirs, files in os.walk(rootpath):


        # ignore directories with no files
        if not files:
            continue

        # ignore dot-directories in the root, like '.git'
        if os.path.relpath(path, rootpath).startswith('.'):
            continue

        # mole clusters have a picture and all the moles as child dirs, ignore
        if dirs:
            continue

        yield path
