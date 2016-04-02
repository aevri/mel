#! /usr/bin/env python3
"""Test mel from a user's perspective."""

import argparse
import os
import subprocess
import sys


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

    mel_cmd = './bin/mel'

    expect_returncode(2, mel_cmd)
    expect_ok(mel_cmd, '-h')

    subcommands = [
        'add-cluster',
        'add-single',
        'list',
        'micro-add',
        'micro-view',
        'rotomap-edit',
        'rotomap-molepicker',
        'rotomap-relate',
        'rotomap-show',
        'rotomap-uuid',
    ]

    for s in subcommands:
        expect_ok(mel_cmd, s, '-h')


def expect_ok(*args):
    return expect_returncode(0, args)


def expect_returncode(expected_code, command):
    print('.', end='', flush=True)
    # print('Running "{}", expect return code {}'.format(
    #     command, expected_code))

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
