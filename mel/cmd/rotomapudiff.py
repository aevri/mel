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

Add UUIDs to a file 'ignore-new', or 'ignore-missing' in the rotomap dir to
suppress messages about new or missing moles respectively. Blank lines and
lines beginning with '#' are ignored in these files.

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


import collections
import itertools

import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        'OLD',
        type=mel.rotomap.moles.make_argparse_rotomap_directory,
        nargs='+',
        help="Paths to the rotomap directories to base comparison on.")
    parser.add_argument(
        'NEW',
        type=mel.rotomap.moles.make_argparse_rotomap_directory,
        help="Path to the new rotomap directory to compare with.")
    parser.add_argument(
        '--show-all', '-a',
        action='store_true',
        help="Show all mole UUIDs, even if ignored.")


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

    ignore_new = mel.rotomap.moles.load_potential_set_file(
        args.NEW.path, mel.rotomap.moles.IGNORE_NEW_FILENAME)
    ignore_missing = mel.rotomap.moles.load_potential_set_file(
        args.NEW.path, mel.rotomap.moles.IGNORE_MISSING_FILENAME)

    to_uuids = uuids_from_dir(args.NEW)

    diff = mel.rotomap.moles.MoleListDiff(
        from_uuids, to_uuids, ignore_new, ignore_missing)

    print_category('New moles:', diff.new)

    missing_uuids = sorted(
        list(diff.missing),
        key=uuid_to_fromdirs.get,
        reverse=True)
    for group, ids in itertools.groupby(missing_uuids, uuid_to_fromdirs.get):
        if len(group) == len(args.OLD):
            print_category('Not in NEW map, in all OLD maps:', ids)
        else:
            group_desc = ', '.join(str(x) for x in group)
            print_category(
                'Not in NEW map. Only in {}:'.format(group_desc),
                ids)

    print_category('Old moles, also seen in NEW map:', diff.matching)

    if args.show_all:
        print_category(
            'Ignored new moles:',
            diff.ignored_new)
        print_category(
            'Would ignore new moles:',
            diff.would_ignore_new)
        print_category(
            'Ignored missing moles:',
            diff.ignored_missing)
        print_category(
            'Would ignore missing moles:',
            diff.would_ignore_missing)


# -----------------------------------------------------------------------------
# Copyright (C) 2017 Angelos Evripiotis.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------ END-OF-FILE ----------------------------------
