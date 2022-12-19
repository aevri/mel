"""Summarize the timelog of a repository.

Answers the question 'How long do things take?'.
"""

import sys

import pandas as pd

import mel.lib.fs
import mel.rotomap.moles


def setup_parser(parser):
    pass


def process_args(args):
    try:
        melroot = mel.lib.fs.find_melroot()
    except mel.lib.fs.NoMelrootError:
        print("Not in a mel repo, could not find melroot", file=sys.stderr)
        return 1

    timelog_path = melroot / mel.lib.fs.TIMELOG_NAME

    if not timelog_path.exists():
        print(f"Not found, aborting: {timelog_path}", file=sys.stderr)
        return 2

    timelog = pd.read_csv(
        timelog_path,
        dtype={
            "command": str,
            "mode": str,
            "path": str,
            "elapsed_secs": "float",
        },
        parse_dates=["start"],
    )

    print()

    print("Time spent, per command:")
    print(
        timelog[["command", "elapsed_secs"]]
        .groupby("command")
        .sum()["elapsed_secs"]
        .sort_values(ascending=False)
    )

    print()

    print("Events logged, per command:")
    print(
        timelog[["command", "elapsed_secs"]]
        .groupby("command")
        .size()
        .sort_values(ascending=False)
    )

    print()


# -----------------------------------------------------------------------------
# Copyright (C) 2022 Angelos Evripiotis.
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
