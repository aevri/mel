#! /usr/bin/env python3
"""Test mel from a user's perspective."""

import argparse
import contextlib
import os
import pathlib
import subprocess
import sys
import tempfile

import mel.cmd.mel


class ExpectationError(Exception):
    def __init__(self, message, completed_process):
        super(ExpectationError, self).__init__(message)
        self.completed_process = completed_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    # cd to the root of the repository, so all the paths are relative to that
    rootdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    os.chdir(rootdir)

    try:
        run_tests()
    except ExpectationError as e:
        print()
        print(e)
        print("-- stdout:")
        print(e.completed_process.stdout)
        print("-- stderr:")
        print(e.completed_process.stderr)
        print("--")
        return 1

    print("OK")


def run_tests():
    run_mel_help_tests()
    run_mel_debug_help_tests()
    run_smoke_test()


def run_mel_help_tests():

    mel_cmd = "mel"

    expect_returncode(2, mel_cmd)
    expect_ok(mel_cmd, "-h")

    for subcommand in mel.cmd.mel.COMMANDS["root"].keys():
        expect_ok(mel_cmd, subcommand, "-h")

    for subcommand in mel.cmd.mel.COMMANDS["micro"].keys():
        expect_ok(mel_cmd, "micro", subcommand, "-h")

    for subcommand in mel.cmd.mel.COMMANDS["rotomap"].keys():
        expect_ok(mel_cmd, "rotomap", subcommand, "-h")


def run_mel_debug_help_tests():

    mel_cmd = "mel-debug"

    expect_returncode(2, mel_cmd)
    expect_ok(mel_cmd, "-h")

    subcommands = [
        "bench-automark",
        "gen-repo",
        "render-valuefield",
    ]

    for s in subcommands:
        expect_ok(mel_cmd, s, "-h")


def run_smoke_test():
    with chtempdir_context():
        expect_ok("mel-debug", "gen-repo", ".")
        target_part = pathlib.Path("rotomaps/parts/LeftLeg/Lower")

        target_rotomap_0 = target_part / "2016_01_01"
        target_image_files = list(target_rotomap_0.glob("*.jpg"))
        target_json_files = list(target_rotomap_0.glob("*.jpg.json"))
        expect_ok("mel", "rotomap", "automask", *target_image_files)
        expect_ok("mel", "rotomap", "calc-space", *target_image_files)

        target_rotomap_1 = target_part / "2017_01_01"
        target_image_files = list(target_rotomap_1.glob("*.jpg"))
        target_json_files = list(target_rotomap_1.glob("*.jpg.json"))
        expect_ok("mel", "rotomap", "automask", *target_image_files)
        expect_ok("mel", "rotomap", "calc-space", *target_image_files)

        expect_ok("mel", "rotomap", "identify2-train")

        target_rotomap_2 = target_part / "2018_01_01"
        target_image_files = list(target_rotomap_2.glob("*.jpg"))
        target_json_files = list(target_rotomap_2.glob("*.jpg.json"))

        for json_file in target_json_files:
            json_file.rename(json_file.with_suffix(".json.bak"))

        expect_ok("mel", "rotomap", "automask", *target_image_files)
        expect_ok("mel", "rotomap", "calc-space", *target_image_files)
        expect_ok("mel", "rotomap", "automark", *target_image_files)
        expect_ok("mel", "rotomap", "identify2", *target_image_files)

        for json_file in target_json_files:
            json_file.with_suffix(".json.bak").rename(json_file)

        expect_ok("mel", "rotomap", "confirm", *target_json_files)
        expect_ok("mel", "rotomap", "mark-unchanged", target_rotomap_2)
        expect_ok("mel", "rotomap", "list", *target_json_files)
        expect_ok("mel", "rotomap", "loadsave", *target_json_files)
        expect_ok("mel", "status", "-ttdd")
        expect_ok("mel", "micro", "list")


@contextlib.contextmanager
def chtempdir_context():
    with tempfile.TemporaryDirectory() as tempdir:
        saved_path = os.getcwd()
        os.chdir(tempdir)
        try:
            yield
        finally:
            os.chdir(saved_path)


def expect_ok(*args):
    return expect_returncode(0, args)


def expect_returncode(expected_code, command):
    print(".", end="", flush=True)

    result = subprocess.run(
        command,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != expected_code:
        raise ExpectationError(
            "'{cmd}' returned {rc}, expected {erc}".format(
                cmd=command, rc=result.returncode, erc=expected_code
            ),
            result,
        )


if __name__ == "__main__":
    sys.exit(main())


# -----------------------------------------------------------------------------
# Copyright (C) 2015-2019 Angelos Evripiotis.
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
