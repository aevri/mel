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
