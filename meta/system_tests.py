#! /usr/bin/env python3
"""Test mel from a user's perspective."""

import argparse
import contextlib
import os
import subprocess
import sys
import tempfile


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

    mel_cmd = 'mel'

    expect_returncode(2, mel_cmd)
    expect_ok(mel_cmd, '-h')

    subcommands = [
        'add-cluster',
        'add-single',
        'list',
        'micro-add',
        'micro-compare',
        'micro-view',
        'rotomap-automark',
        'rotomap-automask',
        'rotomap-calc-space',
        'rotomap-compare',
        'rotomap-confirm',
        'rotomap-edit',
        'rotomap-filter-marks',
        'rotomap-identify',
        'rotomap-list',
        'rotomap-montage-single',
        'rotomap-organise',
        'rotomap-rm',
        'rotomap-udiff',
        'rotomap-uuid',
        'status',
    ]

    for s in subcommands:
        expect_ok(mel_cmd, s, '-h')


def run_mel_debug_help_tests():

    mel_cmd = 'mel-debug'

    expect_returncode(2, mel_cmd)
    expect_ok(mel_cmd, '-h')

    subcommands = [
        'bench-automark',
        'gen-repo',
        'render-valuefield',
    ]

    for s in subcommands:
        expect_ok(mel_cmd, s, '-h')


def run_smoke_test():
    with chtempdir_context():
        expect_ok('mel-debug', 'gen-repo', '.')
        target_rotomap = 'rotomaps/parts/LeftLeg/Lower/2018_01_01'
        target_image = target_rotomap + '/0.jpg'
        target_json = target_image + '.json'
        expect_ok('mel', 'rotomap-automask', target_image)
        expect_ok('mel', 'rotomap-calc-space', target_image)
        expect_ok('mel', 'rotomap-automark', target_image)
        expect_ok('mel', 'rotomap-confirm', target_json)
        expect_ok('mel', 'rotomap-list', target_json)
        expect_ok('mel', 'rotomap-loadsave', target_json)
        expect_ok('mel', 'status', '-ttdd')
        expect_ok('mel', 'list')


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
    print('.', end='', flush=True)

    result = subprocess.run(
        command,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    if result.returncode != expected_code:
        raise ExpectationError("'{cmd}' returned {rc}, expected {erc}".format(
            cmd=command, rc=result.returncode, erc=expected_code),
            result)


if __name__ == '__main__':
    sys.exit(main())
