#! /bin/bash

trap 'echo Failed.' EXIT
set -e # exit immediately on error

# cd to the root of the repository, so all the paths are relative to that
cd "$(dirname "$0")"/..

allscripts=$(find mel/ -iname '*.py' |  tr '\n' ' ')

black --quiet --line-length 79 $allscripts
printf "."

docformatter -i $allscripts
printf "."

isort --quiet --apply $allscripts
printf "."

echo
trap - EXIT
