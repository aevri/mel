#! /bin/bash

# cd to the root of the repository, so all the paths are relative to that
cd "$(dirname "$0")"/..

allscripts=$(find mel/ -iname '*.py' |  tr '\n' ' ')

# Use ruff for fast auto-fixing if available
if uv run python -c "import ruff" 2>/dev/null; then
    time uv run ruff check --fix mel/
    printf "."
fi

time uv run docformatter -i $allscripts
printf "."

time uv run black --quiet --line-length 79 $allscripts
printf "."

time uv run isort --quiet --apply $allscripts
printf "."

echo

# -----------------------------------------------------------------------------
# Copyright (C) 2025 Angelos Evripiotis.
# Generated with assistance from Claude Code.
