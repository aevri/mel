"""Replay recorded videos of moles and analyse."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import cv2

import mel.lib.moleimaging


def setup_parser(parser):
    parser.add_argument(
        'path',
        type=str,
        nargs='+',
        help="Paths to movies to replay.")


def process_args(args):

    pool = multiprocessing.Pool()

    filename_stats_iter = pool.imap_unordered(get_stats, args.path)
    for filename, stats in filename_stats_iter:
        print(filename)
        if stats:
            print(map(lambda x: int(x), stats))


def get_stats(filename):

    final_stats = None

    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise Exception("Could not open {}.".format(filename))

    is_finished = False
    mole_acquirer = mel.lib.moleimaging.MoleAcquirer()
    while not is_finished:
        ret, frame = cap.read()
        if not ret:
            is_finished = True
            continue

        _, stats = mel.lib.moleimaging.find_mole(frame)
        mole_acquirer.update(stats)
        if mole_acquirer.just_locked():
            final_stats = stats

    return filename, final_stats
