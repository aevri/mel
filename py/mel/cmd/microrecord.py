"""Record videos of individual moles live from a microscope."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2

import mel.lib.moleimaging


def setup_parser(parser):
    parser.add_argument(
        '--filename-prefix',
        type=str,
        default=None,
        help="A prefix to add to each video recorded.")


def process_args(args):

    filename_format = 'mole_{number}.avi'
    if args.filename_prefix:
        filename_format = args.filename_prefix + '_' + filename_format

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open video capture device.")

    # read first frame to get dimensions
    ret, frame = cap.read()
    if not ret:
        raise Exception("Could not read frame.")
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    # create an 800x600 window
    window_name = "output"
    cv2.namedWindow(window_name)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_width = frame_width
    window_height = frame_height
    cv2.resizeWindow(window_name, window_width, window_height)

    video_writer = None

    is_finished = False
    mole_acquirer = mel.lib.moleimaging.MoleAcquirer()
    mole_counter = 0
    while not is_finished:
        key = cv2.waitKey(50)
        if key != -1:
            raise Exception('User aborted.')

        ret, frame = cap.read()
        if not ret:
            raise Exception("Could not read frame.")

        ringed, stats = mel.lib.moleimaging.find_mole(frame)
        mole_acquirer.update(stats)

        if mole_acquirer.just_locked():
            mole_counter += 1
            filename = filename_format.format(number=mole_counter)
            print(filename)
            video_writer = make_recorder(filename, frame_width, frame_height)
        elif mole_acquirer.just_unlocked():
            video_writer.release()
            video_writer = None

        if mole_acquirer.is_locked:
            # show the image with mole encircled
            cv2.imshow(window_name, ringed)
            video_writer.write(frame)
        else:
            # show the output from the microscope
            cv2.imshow(window_name, frame)


def make_recorder(name, width, height):
    fourcc = cv2.cv.CV_FOURCC(*'MJPG')
    return cv2.VideoWriter(
        name,
        fourcc,
        25.0,
        (width, height))
