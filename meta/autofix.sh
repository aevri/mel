#! /bin/bash

# cd to the root of the repository, so all the paths are relative to that
cd "$(dirname "$0")"/..

allscripts=$(find mel/ -iname '*.py' |  tr '\n' ' ')

docformatter -i $allscripts
printf "."

# Use ruff for formatting and import sorting (replaces black and isort)
python3 -m ruff format --line-length 79 mel/
printf "."

python3 -m ruff check --fix mel/
printf "."

echo
