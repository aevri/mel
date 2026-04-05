"""Show a live view through an attached microscope."""

import argparse

import cv2

import mel.lib.moleimaging


def setup_parser(parser: argparse.ArgumentParser) -> None:
    pass


def process_args(_args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        msg = "Could not open video capture device."
        raise RuntimeError(msg)

    # create an 800x600 window
    window_name = "output"
    cv2.namedWindow(window_name)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_width = 800
    window_height = 600
    cv2.resizeWindow(window_name, window_width, window_height)

    # loop until the user presses a key
    print("Press any key to exit.")
    is_finished = False
    mole_acquirer = mel.lib.moleimaging.MoleAcquirer()
    while not is_finished:
        key = cv2.waitKey(50)
        if key != -1:
            msg = "User aborted."
            raise RuntimeError(msg)

        ret, frame = cap.read()
        if not ret:
            msg = "Could not read frame."
            raise RuntimeError(msg)

        ringed, stats = mel.lib.moleimaging.find_mole(frame)

        mole_acquirer.update(stats)

        if mole_acquirer.is_locked:
            # show the image with mole encircled
            cv2.imshow(window_name, ringed)
        else:
            # show the output from the microscope
            cv2.imshow(window_name, frame)


# -----------------------------------------------------------------------------
# Copyright (C) 2015-2018, 2026 Angelos Evripiotis.
# Generated with assistance from Claude Code.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------ END-OF-FILE ----------------------------------
