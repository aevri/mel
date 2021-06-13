#! /bin/bash
trap 'echo Failed.' EXIT
set -e
brew install python3
brew tap homebrew/science
brew install opencv3 --with-python3
echo /usr/local/opt/opencv3/lib/python2.7/site-packages >> /usr/local/lib/python2.7/site-packages/opencv3.pth
echo /usr/local/opt/opencv3/lib/python3.6/site-packages >> /usr/local/lib/python3.6/site-packages/opencv3.pth
echo 'Try "import cv2" ..'
python3 -c 'import cv2'
echo OK
trap - EXIT

# For using opencv within a Pipenv this way, see this issue raised on pipenv:
# https://github.com/pypa/pipenv/issues/1313
#
# $ brew install opencv
# $ pipenv --python 3.6
# $ pipenv install numpy
# $ ln -s "$(brew --prefix)"/lib/python3.6/site-packages/cv2*.so "$(pipenv --venv)"/lib/python3.6/site-packages
# $ pipenv run python -c "import cv2; print(cv2.__version__)"
# 3.4.0

# When invoking from a virtual env:
# ln -s "$(brew --prefix)"/lib/python3.7/site-packages/cv2 "${VIRTUAL_ENV}"/lib/python3.7/site-packages/

# Also probably want to do this when developing:
# $ pipenv install -e .[dev]

# To get mel to install under Conda, it appears we need to install the Mac SDK
# headers, or "limits.h" won't be found:
#
#     conda install gcc opencv
#     sudo installer -pkg /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg -target /
#

# Pipenv experiment:
#
#   pipenv install --site-packages -e .[dev]
#

# Pipx experiment
#
#   pipx install --system-site-packages -e .[dev]
#
