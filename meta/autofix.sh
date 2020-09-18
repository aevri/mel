#! /bin/bash

trap 'echo Failed.' EXIT
set -e # exit immediately on error

# cd to the root of the repository, so all the paths are relative to that
cd "$(dirname "$0")"/..

allscripts=$(find mel/ -iname '*.py' |  tr '\n' ' ')

black --line-length 80 $allscripts
printf "."

docformatter -i $allscripts
printf "."

echo
trap - EXIT
