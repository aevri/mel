"""List the moles in a mole catalog."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


def setup_parser(parser):
    pass


def process_args(args):
    for mole in _yield_mole_dirs('.'):
        print(mole.catalog_relative_path)


class _Mole(object):

    def __init__(self, full_path, catalog_relative_path):
        super(_Mole, self).__init__()
        self.full_path = full_path
        self.catalog_relative_path = catalog_relative_path


def _yield_mole_dirs(rootpath):
    for path, dirs, files in os.walk(rootpath):

        this_dirname = os.path.basename(path)

        if this_dirname == '__micro__':
            continue

        catalog_relpath = os.path.relpath(path, rootpath)

        # ignore directories with no files
        if not files:
            continue

        # ignore dot-directories in the root, like '.git'
        if catalog_relpath.startswith('.'):
            continue

        unknown_dirs = set(dirs)
        unknown_dirs.discard('__micro__')

        # mole clusters have a picture and all the moles as child dirs, ignore
        if unknown_dirs:
            continue

        yield _Mole(
            os.path.abspath(path),
            catalog_relpath)
