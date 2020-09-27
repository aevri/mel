#! /bin/bash

die() {
    echo $@ >&2
    exit 1
}

# cd to the root of the repository, so all the paths are relative to that
cd "$(dirname "$0")"/.. || die 'Could not cd to repo root'

# check that the working copy is clean
git diff --exit-code
if [ "$?" -ne 0 ]; then
    echo
    echo 'The working copy is dirty (see diff above)'
    echo
    echo 'We need a clean working copy before automatically running "autofix"'
    echo 'otherwise it may be confusing as to where changes came from. Also '
    echo 'we want to check for a clean working copy afterwards to make sure '
    echo 'that autofix did not make any changes.'
    exit 1
fi

printf "autofix: "
./meta/autofix.sh || die "Could not run autofix."
git diff --exit-code
if [ "$?" -ne 0 ]; then
    echo
    echo The working copy is dirty after automatic fixes, please review and
    echo add the changes before continuing.
    exit 1
fi

printf "static tests: "
./meta/static_tests.sh || die 'Static tests failed.'

printf "unit tests: "
./meta/unit_tests.sh || die 'Unit tests failed.'

printf "system tests: "
pytest -v || die 'System tests failed.'
