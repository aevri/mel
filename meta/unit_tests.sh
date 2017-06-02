#! /bin/bash

# cd to the root of the repository, so all the paths are relative to that
cd "$(dirname "$0")"/..

python3 -m nose mel/ --with-doc --doctest-tests --all-modules $@
