"""Show the status of a mel repository.

This idea is to show you active concerns and what you can do about them, with
executable examples.

This is meant to be similar in usage to 'git status', or perhaps 'ls'. Answers
the question 'What's happening here, and what shall I do next?'.

"""

# There are a few things to vary on in the future:
#
#     - Which paths are included
#     - What kinds of things are included (alerts, errors, info, fun facts)
#     - Level of detail, e.g. individual moles, rotomaps, parts, etc.
#
# Potentially we can also try to fit to a certain amount of screen real-estate.


import collections
import datetime
import enum
import pathlib
import sys
import textwrap

import mel.rotomap.moles


class Notification():

    def __init__(self, path):
        self.path = path

    def format(self, detail_level):
        return str(self.path)


class AlertNotification(Notification):
    pass


class RotomapNewMoleAlert(AlertNotification):

    def __init__(self, path):
        super().__init__(path)
        self.uuid_list = []

    def format(self, detail_level):
        output = f'{self.path}'
        if detail_level > 0:
            output += '\n\n'
            output += '\n'.join(' ' * 2 + f'{u}' for u in self.uuid_list)
            output += '\n'

        return output


@enum.unique
class Alert(enum.Enum):
    NEW_MOLE = enum.auto()


@enum.unique
class Error(enum.Enum):
    INVALID_DATE = enum.auto()


@enum.unique
class Info(enum.Enum):
    NO_BASE_DIR = enum.auto()
    UNEXPECTED_FILE = enum.auto()
    MISSING_MOLE = enum.auto()
    UNCONFIRMED_UUID = enum.auto()
    NO_MOLE_FILE = enum.auto()
    NO_MASK = enum.auto()


def setup_parser(parser):
    parser.add_argument('--detail-level', '-d', action='count', default=0)


def process_args(args):
    try:
        melroot = find_melroot()
    except NoMelrootError:
        print('Not in a mel repo, could not find melroot', file=sys.stderr)
        return 1

    print(f'melroot: {melroot}')

    notices = collections.defaultdict(list)

    rotomaps_path = melroot / 'rotomaps'
    if rotomaps_path.exists():
        check_rotomaps(rotomaps_path, notices)
    else:
        notices[Info.NO_BASE_DIR].append('rotomaps')

    for kind, name_list in notices.items():
        print()
        print(kind)
        for name in name_list:
            try:
                print(textwrap.indent(
                    name.format(args.detail_level),
                    '  '))
            except AttributeError:
                print(' ', name)


class NoMelrootError(Exception):
    pass


def find_melroot():
    original_path = pathlib.Path.cwd()

    path = original_path
    while True:
        melroot = path / 'melroot'
        if melroot.exists():
            return path

        parent = path.parent
        if parent == path:
            raise NoMelrootError(original_path)
        path = parent


def check_rotomaps(path, notices):
    # incoming_path = path / 'incoming'
    parts_path = path / 'parts'
    if parts_path.exists():
        # So far I've organised parts like so:
        #
        #   LeftLeg/Upper
        #   LeftLeg/Lower
        #   LeftLeg/Foot
        #   Trunk/Back
        #   Trunk/Waist
        #   etc.
        #
        # So each part is a two-level thing. Each of the parts has leaf
        # directories that are the actual rotomaps. The rotomaps are names
        # after the day they are captured, in ISO 8601 format. Like so:
        #
        #   LeftLeg/Upper/2017_01_20
        #   LeftLeg/Upper/2017_02_14
        #   LeftLeg/Upper/2017_04_01
        #   etc.
        #
        # Might as well assume the same for all, for now. Later we can allow
        # arbitrary nesting.
        #
        for major_part in parts_path.iterdir():
            if major_part.is_dir():
                for minor_part in major_part.iterdir():
                    if minor_part.is_dir():
                        check_rotomap_minor_part(minor_part, notices)
                    else:
                        notices[Info.UNEXPECTED_FILE].append(minor_part)
            else:
                notices[Info.UNEXPECTED_FILE].append(major_part)
    else:
        notices[Info.NO_BASE_DIR].append(parts_path)


def check_rotomap_minor_part(path, notices):
    rotomap_list = []

    for rotomap_path in path.iterdir():
        if rotomap_path.is_dir():
            try:
                datetime.datetime.strptime(
                    rotomap_path.name[:10],
                    '%Y_%m_%d')
            except ValueError:
                notices[Error.INVALID_DATE].append(rotomap_path)
            else:
                rotomap_list.append(
                    mel.rotomap.moles.RotomapDirectory(
                        rotomap_path))
        else:
            notices[Info.UNEXPECTED_FILE].append(rotomap_path)

    rotomap_list.sort(key=lambda x: x.path)
    check_rotomap_list(notices, rotomap_list)

    for rotomap in rotomap_list:
        check_rotomap(notices, rotomap)


def uuids_from_dir(rotomap_dir):
    uuid_set = set()
    for _, moles in rotomap_dir.yield_mole_lists():
        for m in moles:
            if m[mel.rotomap.moles.KEY_IS_CONFIRMED]:
                uuid_set.add(m['uuid'])
    return uuid_set


def check_rotomap_list(notices, rotomap_list):

    if len(rotomap_list) < 2:
        return

    old_ones = rotomap_list[:-1]
    newest = rotomap_list[-1]

    uuid_to_oldmaps = collections.defaultdict(set)
    for dir_ in old_ones:
        for _, mole_list in dir_.yield_mole_lists():
            for mole in mole_list:
                if mole[mel.rotomap.moles.KEY_IS_CONFIRMED]:
                    uuid_to_oldmaps[mole['uuid']].add(dir_.path)

    old_uuids = set(uuid_to_oldmaps.keys())
    new_uuids = uuids_from_dir(newest)

    ignore_new = mel.rotomap.moles.load_potential_set_file(
        newest.path, mel.rotomap.moles.IGNORE_NEW_FILENAME)
    ignore_missing = mel.rotomap.moles.load_potential_set_file(
        newest.path, mel.rotomap.moles.IGNORE_MISSING_FILENAME)

    diff = mel.rotomap.moles.MoleListDiff(
        old_uuids, new_uuids, ignore_new, ignore_missing)

    if diff.new:
        new_mole_alert = RotomapNewMoleAlert(newest.path)
        new_mole_alert.uuid_list.extend(diff.new)
        notices[Alert.NEW_MOLE].append(new_mole_alert)

    for uuid_ in diff.missing:
        notices[Info.MISSING_MOLE].append(f'{newest.path} {uuid_}')


def check_rotomap(notices, rotomap):

    for imagepath, mole_list in rotomap.yield_mole_lists():
        for mole in mole_list:
            if not mole[mel.rotomap.moles.KEY_IS_CONFIRMED]:
                notices[Info.UNCONFIRMED_UUID].append(
                    f'{imagepath} {mole["uuid"]}')

    for frame in rotomap.yield_frames():
        if not frame.has_mole_file():
            notices[Info.NO_MOLE_FILE].append(f'{frame.path}')
        if not frame.has_mask():
            notices[Info.NO_MASK].append(f'{frame.path}')


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