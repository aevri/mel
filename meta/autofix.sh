#! /bin/bash

# cd to the root of the repository, so all the paths are relative to that
cd "$(dirname "$0")"/..

allscripts=$(find mel/ -iname '*.py' |  tr '\n' ' ')

docformatter -i $allscripts
printf "."

black --quiet --line-length 79 $allscripts
printf "."

isort --quiet --apply $allscripts
printf "."

echo
