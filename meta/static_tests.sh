#! /bin/bash

trap 'echo Failed.' EXIT
set -e # exit immediately on error

# cd to the root of the repository, so all the paths are relative to that
cd "$(dirname "$0")"/..

allscripts=$(find py/ -iname '*.py' |  tr '\n' ' ')

printf '.'
python3 -m pylint --errors-only py/mel/

printf '.'
python3 -m flake8 $allscripts

printf '.'
python3 -m vulture bin/mel --exclude '*__t.py' py/

echo OK
trap - EXIT
