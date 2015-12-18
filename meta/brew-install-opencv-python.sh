#! /bin/bash
trap 'echo Failed.' EXIT
set -e
brew install python
brew tap homebrew/science
brew install opencv
echo 'Try "import cv" ..'
python -c 'import cv'
echo OK
trap - EXIT
