"""Replay recorded videos of moles and analyse."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2

import mel.lib.moleimaging


def setup_parser(parser):
    parser.add_argument(
        'path',
        type=str,
        nargs='+',
        help="Paths to movies to replay.")


def process_args(args):

    for path in args.path:

        print(path)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise Exception("Could not open {}.".format(path))

        is_finished = False
        mole_acquirer = mel.lib.moleimaging.MoleAcquirer()
        while not is_finished:
            ret, frame = cap.read()
            if not ret:
                is_finished = True
                continue

            _, stats = mel.lib.moleimaging.find_mole(frame)
            mole_acquirer.update(stats)
