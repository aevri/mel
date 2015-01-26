"""Show a live view through an attached microscope."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2

import mel.lib.moleimaging


def setup_parser(parser):
    pass


def process_args(args):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open video capture device.")

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
    mole_acquirer = MoleAcquirer()
    while not is_finished:
        key = cv2.waitKey(50)
        if key != -1:
            raise Exception('User aborted.')

        ret, frame = cap.read()
        if not ret:
            raise Exception("Could not read frame.")

        # highlight areas of high saturation, they are likely moles
        img = frame.copy()
        img = cv2.blur(img, (40, 40))
        img = cv2.cvtColor(img, cv2.cv.CV_BGR2HSV)
        img = cv2.split(img)[1]
        _, img = cv2.threshold(img, 30, 255, cv2.cv.CV_THRESH_BINARY)
        ringed, stats = mel.lib.moleimaging.process_contours(img, frame)

        mole_acquirer.update(stats)

        if mole_acquirer.is_locked:
            # show the image with mole encircled
            cv2.imshow(window_name, ringed)
        else:
            # show the output from the microscope
            cv2.imshow(window_name, frame)


class MoleAcquirer(object):

    def __init__(self):
        super(MoleAcquirer, self).__init__()
        self._is_locked = False
        self._was_locked = False
        self._last_stats = None
        self._last_stats_diff = None

    def update(self, stats):
        self._was_locked = self._is_locked
        if stats and self._last_stats:
            stats_diff = map(lambda x, y: x - y, self._last_stats, stats)
            self._last_stats = map(lambda x, y: _lerp(x, y, 0.1), self._last_stats, stats)

            if self._last_stats_diff:
                self._last_stats_diff = map(
                    lambda x, y: _lerp(x, y, 0.1),
                    self._last_stats_diff,
                    stats_diff)

                should_lock = all(
                    map(lambda x: int(x) == 0, self._last_stats_diff))

                should_unlock = any(
                    map(lambda x: abs(int(x)) > 1, self._last_stats_diff))

                if not self._is_locked and should_lock:
                    self._is_locked = True
                    print(
                        'lock',
                        map(lambda x: int(x), self._last_stats),
                        map(lambda x: int(x), self._last_stats_diff)
                    )
                elif self._is_locked and should_unlock:
                    self._is_locked = False
                    # print(
                    #     "unlock",
                    #     map(lambda x: int(x), self._last_stats),
                    #     map(lambda x: int(x), self._last_stats_diff)
                    # )

            else:
                self._last_stats_diff = stats_diff
                self._is_locked = False
        elif stats:
            self._last_stats = stats
            self._is_locked = False
        else:
            self._is_locked = False

    def just_unlocked(self):
        return self._was_locked and not self._is_locked

    def just_locked(self):
        return not self._was_locked and self._is_locked

    @property
    def is_locked(self):
        return self._is_locked

    @property
    def last_stats(self):
        return self._last_stats


def _lerp(origin, target, factor_0_to_1):
    towards = target - origin
    return origin + (towards * factor_0_to_1)
