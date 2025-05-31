#! /usr/bin/env python3
"""Smoke-test mel from the CLI, make sure nothing errors out."""

import contextlib
import json
import os
import pathlib
import subprocess
import tempfile

import mel.cmd.mel


def test_mel_help():

    mel_cmd = "mel"

    expect_returncode(2, mel_cmd)
    expect_ok(mel_cmd, "-h")

    for subcommand in mel.cmd.mel.COMMANDS["root"].keys():
        expect_ok(mel_cmd, subcommand, "-h")

    for subcommand in mel.cmd.mel.COMMANDS["micro"].keys():
        expect_ok(mel_cmd, "micro", subcommand, "-h")

    for subcommand in mel.cmd.mel.COMMANDS["rotomap"].keys():
        expect_ok(mel_cmd, "rotomap", subcommand, "-h")


def test_mel_debug_help():

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


def test_smoke():
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

        expect_ok("mel", "rotomap", "identify-train")

        target_rotomap_2 = target_part / "2018_01_01"
        target_image_files = list(target_rotomap_2.glob("*.jpg"))
        target_json_files = list(target_rotomap_2.glob("*.jpg.json"))

        expect_ok("mel", "rotomap", "automask", *target_image_files)
        expect_ok("mel", "rotomap", "filter-marks-pretrain", *target_image_files)
        expect_ok("mel", "rotomap", "filter-marks-train", "--train-proportion", "1")
        expect_ok(
            "mel",
            "rotomap",
            "automark2-train",
            "-e",
            "1",
            "-b",
            "1",
            "--limit-train-batches",
            "1",
            "--limit-valid-batches",
            "1",
            "--no-post-validate",
        )

        for json_file in target_json_files:
            json_file.rename(json_file.with_suffix(".json.bak"))

        expect_ok("mel", "rotomap", "automark", *target_image_files)
        expect_ok("mel", "rotomap", "filter-marks", *target_image_files)
        expect_ok("mel", "rotomap", "calc-space", *target_image_files)
        expect_ok("mel", "rotomap", "identify", *target_image_files)

        for json_file in target_json_files:
            json_file.with_suffix(".json.bak").rename(json_file)

        expect_ok(
            "mel", "rotomap", "automark", "--extra-stem", "smoke", *target_image_files
        )
        expect_ok(
            "mel",
            "rotomap",
            "filter-marks",
            "--extra-stem",
            "smoke",
            *target_image_files
        )
        expect_ok(
            "mel", "rotomap", "compare-extra-stem", "smoke", *target_image_files
        )
        expect_ok(
            "mel", "rotomap", "compare-extra-stem", "smoke", *target_image_files
        )
        expect_ok(
            "mel", "rotomap", "identify", "--extra-stem", "smoke", *target_image_files
        )
        expect_ok(
            "mel",
            "rotomap",
            "compare-extra-stem",
            "--compare-uuids",
            "smoke",
            *target_image_files
        )
        expect_ok("mel", "rotomap", "merge-extra-stem", "smoke", *target_image_files)
        expect_ok("mel", "rotomap", "identify-train", "--extra-stem", "smoke")

        expect_ok("mel", "rotomap", "confirm", *target_json_files)
        expect_ok("mel", "rotomap", "mark-unchanged", target_rotomap_2)
        expect_ok("mel", "rotomap", "list", *target_json_files)
        expect_ok("mel", "rotomap", "loadsave", *target_json_files)

        # Test resize functionality with smaller dimensions
        assert target_image_files, "No target image files found for resize test"
        expect_ok(
            "mel", "rotomap", "resize", "--width", "100", "--height", "100",
            str(target_image_files[0])
        )

        expect_ok("mel", "status", "-ttdd")
        
        # Create empty timelog for timelog command test
        timelog_path = pathlib.Path("timelog.csv")
        if not timelog_path.exists():
            timelog_path.write_text(
                "command,mode,path,start,elapsed_secs\n"
                "test-command,test,rotomaps/parts/TestPart/Lower,2020-01-01T00:00:00,1.0\n"
            )
        
        expect_ok("mel", "timelog")
        expect_ok("mel", "micro", "list")
        
        # Test additional non-interactive rotomap commands
        # uuid command returns 1 when no matches found, so expect that
        expect_returncode(1, "mel", "rotomap", "uuid", "nonexistent-prefix", *target_json_files)
        expect_ok(
            "mel", "rotomap", "rm", "--uuids", "nonexistent-uuid", "--files",
            *target_json_files
        )
        expect_ok("mel", "rotomap", "guess-missing", 
                  str(target_image_files[0]), str(target_image_files[1]))
        expect_ok("mel", "rotomap", "guess-refine", 
                  str(target_image_files[0]), str(target_image_files[1]))
        # For montage-single, we need a UUID, so let's get one from the first JSON file
        # and use the corresponding image file
        json_file = target_json_files[0]
        corresponding_image = str(json_file).replace('.jpg.json', '.jpg')
        
        with open(json_file) as f:
            moles_data = json.load(f)
        if moles_data:
            test_uuid = moles_data[0]['uuid']
            expect_ok(
                "mel", "rotomap", "montage-single", 
                corresponding_image, test_uuid, "test_montage.jpg"
            )
        else:
            # Skip montage-single test if no moles found
            print("Skipping montage-single test - no moles found in test data")


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
    subprocess.check_call(args)


def expect_returncode(expected_code, *args):
    return_code = subprocess.call(args)
    assert return_code == expected_code


# -----------------------------------------------------------------------------
# Copyright (C) 2015-2025 Angelos Evripiotis.
# Generated with assistance from Claude Code.
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
