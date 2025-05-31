#! /bin/bash

# cd to the root of the repository, so all the paths are relative to that
cd "$(dirname "$0")"/..

allscripts=$(find mel/ -iname '*.py' |  tr '\n' ' ')

time uv run ruff check --fix mel/
printf "."

time uv run ruff format mel/
printf "."

time uv run docformatter -i $allscripts
printf "."

echo

# -----------------------------------------------------------------------------
# Copyright (C) 2025 Angelos Evripiotis.
# Generated with assistance from Claude Code.
