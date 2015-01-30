"""Replay recorded videos of moles and record stats."""

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

    results_iter = pool.imap_unordered(get_stats, args.path)
    for filename, images in results_iter:
        print(filename, images)


def get_stats(filename):

    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise Exception("Could not open {}.".format(filename))

    count = 0
    is_finished = False
    mole_acquirer = mel.lib.moleimaging.MoleAcquirer()
    image_list = []
    while not is_finished:
        ret, frame = cap.read()
        if not ret:
            is_finished = True
            continue

        _, stats = mel.lib.moleimaging.find_mole(frame)
        mole_acquirer.update(stats)
        if mole_acquirer.just_locked():
            count += 1
            outname = '{}.{}.jpg'.format(filename, count)
            cv2.imwrite(outname, frame)
            image_list.append(outname)

    return filename, image_list
