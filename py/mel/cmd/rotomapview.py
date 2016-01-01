"""View a 'rotomap' series of images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os


def setup_parser(parser):
    parser.add_argument(
        'PATH',
        type=str,
        help="Path to the mole to add new microscope images to.")
    parser.add_argument(
        '--display-width',
        type=int,
        default=800,
        help="Width of the preview display window.")
    parser.add_argument(
        '--display-height',
        type=int,
        default=600,
        help="Width of the preview display window.")
    parser.add_argument(
        '--rot90',
        type=int,
        default=None,
        help="Rotate images 90 degrees clockwise this number of times.")


def process_args(args):
    display = Display(
        args.PATH, args.display_width, args.display_height, args.rot90)

    left = 63234
    right = 63235

    print("Press any other key to save and quit.")
    is_finished = False
    while not is_finished:
        key = cv2.waitKey(50)
        if key != -1:
            if key == left:
                display.show_prev()
            elif key == right:
                display.show_next()
            else:
                is_finished = True


class Display:

    def __init__(self, path, width, height, rot90):
        self._name = path
        self._width = width
        self._height = height
        self._rot90 = rot90

        # list all images
        self._path_list = [
            os.path.join(path, x)
            for x in os.listdir(path)
        ]

        cv2.namedWindow(self._name)
        cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._name, self._width, self._height)

        self._list_index = 0
        self._num_images = len(self._path_list)

        self.show_current()

    def load_current_image(self):
        image = cv2.imread(self._path_list[self._list_index])
        if self._rot90:
            image = mel.lib.common.rotated90(image, self._rot90)
        return image

    def show_current(self):
        image = self.load_current_image()
        image = mel.lib.image.letterbox(
            image, self._width, self._height)
        cv2.imshow(self._name, image)

    def show_prev(self):
        new_index = self._list_index + self._num_images - 1
        self._list_index = new_index % self._num_images
        self.show_current()

    def show_next(self):
        self._list_index = (self._list_index + 1) % self._num_images
        self.show_current()
