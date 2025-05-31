#! /bin/bash

# cd to the root of the repository, so all the paths are relative to that
cd "$(dirname "$0")"/..

allscripts=$(find mel/ -iname '*.py' |  tr '\n' ' ')

uv run docformatter -i $allscripts
printf "."

uv run black --quiet --line-length 79 $allscripts
printf "."

uv run isort --quiet --apply $allscripts
printf "."

echo

# -----------------------------------------------------------------------------
# Copyright (C) 2025 Angelos Evripiotis.
# Generated with assistance from Claude Code.
