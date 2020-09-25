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
import os
import sys
import textwrap

import colorama

import mel.lib.fs
import mel.rotomap.moles


class Notification:
    def __init__(self, path):
        self.path = path

    def format(self, detail_level):
        return str(self.path)


class AlertNotification(Notification):
    pass


class ErrorNotification(Notification):
    pass


class InfoNotification(Notification):
    pass


class RotomapNewMoleAlert(AlertNotification):
    def __init__(self, path):
        super().__init__(path)
        self.uuid_list = []

    def format(self, detail_level):
        output = f"{self.path}"
        if detail_level > 0:
            output += "\n\n"
            output += "\n".join(" " * 2 + f"{u}" for u in self.uuid_list)
            output += "\n"

        return output


class RotomapLesionChangedAlert(AlertNotification):
    def __init__(self, path):
        super().__init__(path)
        self.uuid_list = []

    def format(self, detail_level):
        output = f"{self.path}"
        if detail_level > 0:
            output += "\n\n"
            output += "\n".join(" " * 2 + f"{u}" for u in self.uuid_list)
            output += "\n"

        return output


class MicroLesionChangedAlert(AlertNotification):
    def __init__(self, path, id_):
        super().__init__(path)
        self.id_ = id_

    def format(self, detail_level):
        output = f"{self.path}"
        if detail_level > 0:
            output += "\n\n"
            output += " " * 2 + f"{self.id_}"
            output += "\n"

        return output


class InvalidDateError(ErrorNotification):
    pass


class RotomapDuplicateUuidError(ErrorNotification):
    def __init__(self, rotomap_path):
        super().__init__(rotomap_path)
        self.frame_to_uuid_list = collections.defaultdict(list)

    def format(self, detail_level):
        output = f"{self.path}"
        if detail_level > 0:
            if detail_level == 1:
                output += "\n\n"
                output += "\n".join(
                    " " * 2 + f"{f}" for f in sorted(self.frame_to_uuid_list)
                )
                output += "\n"
            else:
                f_to_ul = self.frame_to_uuid_list
                for frame, uuid_list in sorted(f_to_ul.items()):
                    output += "\n\n"
                    output += f"  {frame}:\n"
                    output += "\n"
                    output += "\n".join(" " * 4 + f"{u}" for u in uuid_list)

        return output


class RotomapNotLoadable(ErrorNotification):
    def __init__(self, path, error=None):
        super().__init__(path)
        self.error = error

    def format(self, detail_level):
        output = f"{self.path}"

        if detail_level > 0 and self.error is not None:
            output += ":\n\n"
            output += f"  {self.error}"
            if isinstance(self.error, Exception):
                error = self.error
                while True:
                    if error.__cause__:
                        error = error.__cause__
                        output += f"\n  caused by '{error}'."
                    elif error.__context__ and not error.__suppress_context__:
                        error = error.__context__
                        output += f"\n  during '{error}'."
                    else:
                        break
            output += "\n"

        return output


class NoBaseDirInfo(InfoNotification):
    pass


class UnexpectedFileInfo(InfoNotification):
    pass


class UnexpectedDirInfo(InfoNotification):
    pass


class MicroMissingIdInfo(InfoNotification):
    def __init__(self, path):
        super().__init__(path)

    def format(self, detail_level):
        return f"{self.path}"


class RotomapMissingMoleInfo(InfoNotification):
    def __init__(self, path):
        super().__init__(path)
        self.uuid_list = []

    def format(self, detail_level):
        output = f"{self.path}"
        if detail_level > 0:
            output += "\n\n"
            output += "\n".join(" " * 2 + f"{u}" for u in self.uuid_list)
            output += "\n"

        return output


class RotomapMissingLesionUnchangedStatus(InfoNotification):
    def __init__(self, path):
        super().__init__(path)
        self.uuid_list = []

    def format(self, detail_level):
        output = f"{self.path}"
        if detail_level > 0:
            output += "\n\n"
            output += "\n".join(" " * 2 + f"{u}" for u in self.uuid_list)
            output += "\n"

        return output


class RotomapUnconfirmedMoleInfo(InfoNotification):
    def __init__(self, rotomap_path):
        super().__init__(rotomap_path)
        self.frame_to_uuid_list = collections.defaultdict(list)

    def format(self, detail_level):
        output = f"{self.path}"
        if detail_level > 0:
            if detail_level == 1:
                output += "\n\n"
                output += "\n".join(
                    " " * 2 + f"{f}" for f in sorted(self.frame_to_uuid_list)
                )
                output += "\n"
            else:
                f_to_ul = self.frame_to_uuid_list
                for frame, uuid_list in sorted(f_to_ul.items()):
                    output += "\n\n"
                    output += f"  {frame}:\n"
                    output += "\n"
                    output += "\n".join(" " * 4 + f"{u}" for u in uuid_list)

        return output


class RotomapMissingMoleFileInfo(InfoNotification):
    def __init__(self, path):
        super().__init__(path)
        self.frame_list = []

    def format(self, detail_level):
        output = f"{self.path}"
        if detail_level > 0:
            output += "\n\n"
            output += "\n".join(" " * 2 + f"{u}" for u in self.frame_list)
            output += "\n"

        return output


class RotomapMissingMaskInfo(InfoNotification):
    def __init__(self, path):
        super().__init__(path)
        self.frame_list = []

    def format(self, detail_level):
        output = f"{self.path}"
        if detail_level > 0:
            output += "\n\n"
            output += "\n".join(" " * 2 + f"{u}" for u in self.frame_list)
            output += "\n"

        return output


class RotomapMissingSpaceInfo(InfoNotification):
    def __init__(self, path):
        super().__init__(path)
        self.frame_list = []

    def format(self, detail_level):
        output = f"{self.path}"
        if detail_level > 0:
            output += "\n\n"
            output += "\n".join(" " * 2 + f"{u}" for u in self.frame_list)
            output += "\n"

        return output


def setup_parser(parser):
    parser.add_argument("PATH", nargs="?")
    parser.add_argument("--detail", "-d", action="count", default=0)
    parser.add_argument("--trivia", "-t", action="count", default=0)


def process_args(args):
    colorama.init()
    try:
        melroot = mel.lib.fs.find_melroot()
    except mel.lib.fs.NoMelrootError:
        print("Not in a mel repo, could not find melroot", file=sys.stderr)
        return 1

    if args.detail > 2:
        print(f"melroot: {melroot}")

    notice_list = []

    rotomaps_path = melroot / mel.lib.fs.ROTOMAPS_PATH
    if rotomaps_path.exists():
        check_rotomaps(rotomaps_path, notice_list)
    else:
        notice_list.append(NoBaseDirInfo(mel.lib.fs.ROTOMAPS_PATH))

    micro_path = melroot / mel.lib.fs.MICRO_PATH
    if micro_path.exists():
        check_micro(micro_path, notice_list)
    else:
        notice_list.append(NoBaseDirInfo(mel.lib.fs.MICRO_PATH))

    alerts_to_notices = collections.defaultdict(list)
    errors_to_notices = collections.defaultdict(list)
    info_to_notices = collections.defaultdict(list)

    abspath = os.path.abspath(args.PATH) if args.PATH is not None else None

    for notice in notice_list:

        if abspath is not None:
            if not str(notice.path).startswith(abspath):
                continue

        klass = notice.__class__
        if issubclass(klass, AlertNotification):
            alerts_to_notices[klass].append(notice)
        elif issubclass(klass, ErrorNotification):
            errors_to_notices[klass].append(notice)
        elif issubclass(klass, InfoNotification):
            info_to_notices[klass].append(notice)
        else:
            raise RuntimeError(f"Unexpected notice type: {klass}")

    any_notices = bool(alerts_to_notices or errors_to_notices)

    print_klass_to_notices(alerts_to_notices, args.detail, colorama.Fore.RED)
    print_klass_to_notices(
        errors_to_notices, args.detail, colorama.Fore.MAGENTA
    )
    if args.trivia > 0:
        print_klass_to_notices(
            info_to_notices, args.detail, colorama.Fore.BLUE
        )
        if not any_notices and info_to_notices:
            any_notices = True

    return any_notices


def print_klass_to_notices(klass_to_notices, detail_level, fore):

    for klass, notice_list in klass_to_notices.items():
        print()
        print(fore, klass.__name__, colorama.Fore.RESET)
        for notice in notice_list:
            print(textwrap.indent(notice.format(detail_level), "  "))


def check_rotomaps(path, notices):
    # incoming_path = path / 'incoming'
    parts_path = path / "parts"
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
                        notices.append(UnexpectedFileInfo(minor_part))
            else:
                notices.append(UnexpectedFileInfo(major_part))
    else:
        notices.append(NoBaseDirInfo(parts_path))


def check_rotomap_minor_part(path, notices):
    rotomap_list = []

    for rotomap_path in path.iterdir():
        if rotomap_path.is_dir():
            try:
                datetime.datetime.strptime(rotomap_path.name[:10], "%Y_%m_%d")
            except ValueError:
                notices.append(InvalidDateError(rotomap_path))
            else:
                rotomap_list.append(
                    mel.rotomap.moles.RotomapDirectory(rotomap_path)
                )
        else:
            notices.append(UnexpectedFileInfo(rotomap_path))

    rotomap_list.sort(key=lambda x: x.path)
    check_rotomap_list(notices, rotomap_list)

    for rotomap in rotomap_list:
        check_rotomap(notices, rotomap)

    if rotomap_list:
        check_newest_rotomap(notices, rotomap_list[-1])


def uuids_from_dir(rotomap_dir):
    uuid_set = set()
    for _, moles in rotomap_dir.yield_mole_lists():
        for m in moles:
            if m[mel.rotomap.moles.KEY_IS_CONFIRMED]:
                uuid_set.add(m["uuid"])
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
                    uuid_to_oldmaps[mole["uuid"]].add(dir_.path)

    old_uuids = set(uuid_to_oldmaps.keys())
    new_uuids = uuids_from_dir(newest)

    ignore_new = mel.rotomap.moles.load_potential_set_file(
        newest.path, mel.rotomap.moles.IGNORE_NEW_FILENAME
    )
    ignore_missing = mel.rotomap.moles.load_potential_set_file(
        newest.path, mel.rotomap.moles.IGNORE_MISSING_FILENAME
    )

    diff = mel.rotomap.moles.MoleListDiff(
        old_uuids, new_uuids, ignore_new, ignore_missing
    )

    if diff.new:
        new_mole_alert = RotomapNewMoleAlert(newest.path)
        new_mole_alert.uuid_list.extend(diff.new)
        notices.append(new_mole_alert)

    if diff.missing:
        missing_notification = RotomapMissingMoleInfo(newest.path)
        missing_notification.uuid_list.extend(diff.missing)
        notices.append(missing_notification)


def check_rotomap(notices, rotomap):

    unconfirmed = RotomapUnconfirmedMoleInfo(rotomap.path)
    duplicates = RotomapDuplicateUuidError(rotomap.path)

    for imagepath, mole_list in rotomap.yield_mole_lists():
        current_uuid_set = set()
        for mole in mole_list:
            uuid_ = mole["uuid"]

            if uuid_ in current_uuid_set:
                duplicates.frame_to_uuid_list[imagepath].append(uuid_)
            current_uuid_set.add(uuid_)

            if not mole[mel.rotomap.moles.KEY_IS_CONFIRMED]:
                unconfirmed.frame_to_uuid_list[imagepath].append(uuid_)

    if duplicates.frame_to_uuid_list:
        notices.append(duplicates)
    if unconfirmed.frame_to_uuid_list:
        notices.append(unconfirmed)

    missing_mole_file_info = RotomapMissingMoleFileInfo(rotomap.path)
    missing_mask_info = RotomapMissingMaskInfo(rotomap.path)
    missing_space_info = RotomapMissingSpaceInfo(rotomap.path)

    try:
        for frame in rotomap.yield_frames():
            if not frame.has_mole_file():
                missing_mole_file_info.frame_list.append(frame.path)
            if not frame.has_mask():
                missing_mask_info.frame_list.append(frame.path)
            if "ellipse" not in frame.metadata:
                missing_space_info.frame_list.append(frame.path)
    except Exception as e:
        notices.append(RotomapNotLoadable(rotomap.path, e))

    for i in rotomap.path.iterdir():
        if i.is_dir():
            notices.append(UnexpectedDirInfo(i))

    if missing_mole_file_info.frame_list:
        notices.append(missing_mole_file_info)
    if missing_mask_info.frame_list:
        notices.append(missing_mask_info)
    if missing_space_info.frame_list:
        notices.append(missing_space_info)


def check_newest_rotomap(notices, rotomap):

    missing_unchanged_status = RotomapMissingLesionUnchangedStatus(
        rotomap.path
    )

    changed = RotomapLesionChangedAlert(rotomap.path)

    ignore_new = mel.rotomap.moles.load_potential_set_file(
        rotomap.path, mel.rotomap.moles.IGNORE_NEW_FILENAME
    )

    uuids = rotomap.calc_uuids()

    uuid_to_unchanged_status = {
        lesion["uuid"]: lesion[mel.rotomap.moles.KEY_IS_UNCHANGED]
        for lesion in rotomap.lesions
    }

    for u in uuids:
        if u not in uuid_to_unchanged_status:
            if u in ignore_new:
                continue
            missing_unchanged_status.uuid_list.append(u)
            continue
        unchanged_status = uuid_to_unchanged_status[u]
        if unchanged_status is None:
            if u in ignore_new:
                continue
            missing_unchanged_status.uuid_list.append(u)
            continue
        elif not unchanged_status:
            changed.uuid_list.append(u)
            continue

    if missing_unchanged_status.uuid_list:
        notices.append(missing_unchanged_status)
    if changed.uuid_list:
        notices.append(changed)


def check_micro(path, notices):
    parts_path = path / "data"
    if parts_path.exists():
        # So far I've organised parts like so:
        #
        #   LeftArm/Hand
        #   LeftArm/Upper
        #   LeftLeg/Foot
        #   LeftLeg/LowerLeg
        #   LeftLeg/UpperLeg
        #   RightArm/Armpit
        #   RightArm/Forearm
        #   RightArm/Hand
        #   RightArm/Upper
        #   etc.
        #
        # So each part is a two-level thing. Each of the parts has leaf
        # directories that are the actual moles or mole groups.
        #
        for major_part in parts_path.iterdir():
            if major_part.is_dir():
                for minor_part in major_part.iterdir():
                    if minor_part.is_dir():
                        for mole in mel.micro.fs.yield_moles(minor_part):
                            _validate_mole_dir(mole.path, notices)
                            changed_path = mel.micro.fs.Names.CHANGED
                            if (mole.path / changed_path).exists():
                                notices.append(
                                    MicroLesionChangedAlert(mole.path, mole.id)
                                )
                            if mole.id is None:
                                notices.append(
                                    MicroMissingIdInfo(
                                        mole.path / mel.micro.fs.Names.ID
                                    )
                                )
                    else:
                        notices.append(UnexpectedFileInfo(minor_part))
            else:
                notices.append(UnexpectedFileInfo(major_part))
    else:
        notices.append(NoBaseDirInfo(parts_path))


def _validate_mole_dir(path, notices):
    for sub in path.iterdir():
        if sub.name.lower() not in mel.micro.fs.MOLE_DIR_ENTRIES:

            if sub.suffix.lower() in mel.micro.fs.IMAGE_SUFFIXES:
                continue

            if sub.name in mel.micro.fs.FILES_TO_IGNORE and sub.is_file():
                continue

            if sub.name in mel.micro.fs.DIRS_TO_IGNORE and sub.is_dir():
                continue

            if sub.is_dir():
                notices.append(UnexpectedDirInfo(sub))
            else:
                notices.append(UnexpectedFileInfo(sub))


# -----------------------------------------------------------------------------
# Copyright (C) 2018-2019 Angelos Evripiotis.
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
