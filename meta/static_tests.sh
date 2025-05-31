#! /bin/bash

trap 'echo Failed.' EXIT
set -e # exit immediately on error

# cd to the root of the repository, so all the paths are relative to that
cd "$(dirname "$0")"/..

printf '.'
python3 -m pylint --errors-only mel/

printf '.'
python3 -m ruff check mel/

printf '.'
python3 -m vulture \
    --ignore-names training_step,validation_step,configure_optimizers \
    --exclude '*__t.py,mel/rotomap/detectmoles.py,mel/rotomap/identifynn.py' \
    mel/

echo OK
trap - EXIT
