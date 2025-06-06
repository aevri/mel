#! /bin/bash

trap 'echo Failed.' EXIT
set -e # exit immediately on error

# cd to the root of the repository, so all the paths are relative to that
cd "$(dirname "$0")"/..

allscripts=$(find mel/ -iname '*.py' |  tr '\n' ' ')

printf '.'
# flake8-builtins (A)
# refurb (FURB)
# pep8 naming (N)
# Pylint Error (PLE)
uv run ruff check --quiet --extend-select A,FURB,N,PLE mel/

printf '.'
uv run python -m vulture \
    --ignore-names training_step,validation_step,configure_optimizers \
    --exclude '*__t.py,mel/rotomap/detectmoles.py,mel/rotomap/identifynn.py' \
    mel/

echo OK
trap - EXIT

# -----------------------------------------------------------------------------
# Copyright (C) 2025 Angelos Evripiotis.
# Generated with assistance from Claude Code.
