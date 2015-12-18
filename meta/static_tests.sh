#! /bin/bash

trap 'echo Failed.' EXIT
set -e # exit immediately on error

# cd to the root of the repository, so all the paths are relative to that
cd "$(dirname "$0")"/..

allscripts=$(find py/ -iname '*.py' |  tr '\n' ' ')

printf '.'
flake8 $allscripts
printf '.'
./meta/import_all.py py/mel

echo OK
trap - EXIT
