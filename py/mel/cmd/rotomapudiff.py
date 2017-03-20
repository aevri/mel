"""Print what may be missing from NEW, and what has been introduced in NEW.

When rotomapping, the objective is to identify moles that are new. To this end,
we need to compare the uuids from old and new rotomaps of the same region.

First print the moles which only appear in the supplied NEW rotomap, these are
potentially new moles.

Also print moles that appeared in previous rotomaps but not in the NEW rotomap.
These could indicate that the NEW rotomap is incomplete or mismatched with
previous rotomaps.

Finally print moles that appeared in at least some previous rotomaps and the
NEW rotomap. These are the expected case when comparing the OLD and NEW
rotomaps, and can build confidence that the maps are matched correctly.

Example output:

    New moles:

            d779018034004939823cf7603ad94a80

    Not in NEW map, in all OLD maps:

            dcf5a644398640c6aa76f5e1c925b3f4

    Not in NEW map. Only in 2016_01_17, 2016_03_11:

            b561519f03ba488f9ce23ab508e92067

    Old moles, also seen in NEW map:

            274224c9138c4d82a477402cb9c71490

"""


import argparse
import collections
import itertools
import os

import mel.rotomap.moles


class ArgparseRotomapDirectoryType():

    """Use in the 'type=' parameter to add_argument()."""

    def __init__(self, path):
        self.path = path
        if not os.path.isdir(self.path):
            raise argparse.ArgumentTypeError(
                '"{}" is not a directory, so not a rotomap.'.format(self.path))
        files = os.listdir(self.path)
        self._image_paths = [
            os.path.join(self.path, f)
            for f in files
            if f.lower().endswith('.jpg')
        ]
        if not self._image_paths:
            raise argparse.ArgumentTypeError(
                '"{}" has no images, so not a rotomap.'.format(self.path))

    def yield_mole_lists(self):
        """Yield (image_path, mole_list) for all mole image files."""
        for imagepath in self._image_paths:
            yield imagepath, mel.rotomap.moles.load_image_moles(imagepath)


def setup_parser(parser):
    parser.add_argument(
        'OLD',
        type=ArgparseRotomapDirectoryType,
        nargs='+',
        help="Paths to the rotomap directories to base comparison on.")
    parser.add_argument(
        'NEW',
        type=ArgparseRotomapDirectoryType,
        help="Path to the new rotomap directory to compare with.")


def uuids_from_dir(rotomap_dir):
    uuid_set = set()
    for _, moles in rotomap_dir.yield_mole_lists():
        uuid_set |= set(m['uuid'] for m in moles)
    return uuid_set


def print_category(text, uuids):
    if uuids:
        print(text)
        print()
        for uuid_ in uuids:
            print(' ' * 7, uuid_)
        print()


def process_args(args):

    uuid_to_fromdirs = collections.defaultdict(set)
    for dir_ in args.OLD:
        for _, mole_list in dir_.yield_mole_lists():
            for mole in mole_list:
                uuid_to_fromdirs[mole['uuid']].add(dir_.path)

    from_uuids = set(uuid_to_fromdirs.keys())

    to_uuids = uuids_from_dir(args.NEW)

    print_category('New moles:', to_uuids - from_uuids)

    missing_uuids = sorted(
        list(from_uuids - to_uuids),
        key=uuid_to_fromdirs.get,
        reverse=True)
    for group, ids in itertools.groupby(missing_uuids, uuid_to_fromdirs.get):
        if len(group) == len(args.OLD):
            print_category('Not in NEW map, in all OLD maps:', ids)
        else:
            group_desc = ', '.join(group)
            print_category(
                'Not in NEW map. Only in {}:'.format(group_desc),
                ids)

    print_category('Old moles, also seen in NEW map:', from_uuids & to_uuids)
