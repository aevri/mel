#! /bin/bash

# cd to the root of the repository, so all the paths are relative to that
cd "$(dirname "$0")"/..

allscripts=$(find mel/ -iname '*.py' |  tr '\n' ' ')

docformatter -i $allscripts
printf "."

black --quiet --line-length 79 $allscripts
printf "."

# Use ruff to fix import sorting and other auto-fixable issues
python3 -m ruff check --fix mel/
printf "."

# Keep isort as fallback for any remaining import issues
isort --quiet --apply $allscripts
printf "."

echo
