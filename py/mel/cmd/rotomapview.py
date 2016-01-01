"""View a 'rotomap' series of images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os

import mel.lib.common
import mel.lib.image
import mel.lib.math


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

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                image_x, image_y = display.windowxy_to_imagexy(x, y)
                display.show_zoomed(image_x, image_y)

    display.set_mouse_callback(mouse_callback)

    print("Press left for previous image, right for next image.")
    print("Ctrl-click on a point to zoom in on it.")
    print("Press space to restore original zoom.")
    print("Press any other key to quit.")
    is_finished = False
    while not is_finished:
        key = cv2.waitKey(50)
        if key != -1:
            if key == left:
                display.show_prev()
            elif key == right:
                display.show_next()
            elif key == ord(' '):
                display.show_current()
            else:
                is_finished = True

    display.clear_mouse_callback()


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

        self._image_width = image.shape[1]
        self._image_height = image.shape[0]
        letterbox = mel.lib.image.calc_letterbox(
            self._image_width,
            self._image_height,
            self._width,
            self._height)

        self._image_left = letterbox[0]
        self._image_top = letterbox[1]
        self._image_scale = image.shape[1] / letterbox[2]

        image = mel.lib.image.letterbox(
            image, self._width, self._height)
        cv2.imshow(self._name, image)

    def windowxy_to_imagexy(self, window_x, window_y):
        image_x = mel.lib.math.clamp(
            window_x - self._image_left,
            0,
            self._image_width)
        image_y = mel.lib.math.clamp(
            window_y - self._image_top,
            0,
            self._image_height)
        return (
            int(image_x * self._image_scale),
            int(image_y * self._image_scale)
        )

    def show_prev(self):
        new_index = self._list_index + self._num_images - 1
        self._list_index = new_index % self._num_images
        self.show_current()

    def show_next(self):
        self._list_index = (self._list_index + 1) % self._num_images
        self.show_current()

    def show_zoomed(self, x, y):
        image = self.load_current_image()
        nx, ny = mel.lib.image.calc_centering_offset(
            (x, y),
            (image.shape[1], image.shape[0]),
            (self._width, self._height))
        image = mel.lib.image.translated_and_clipped(
            image, nx, ny, self._width, self._height)
        self._image_left = -nx
        self._image_top = -ny
        self._image_width = image.shape[1] + nx
        self._image_height = image.shape[0] + ny
        self._image_scale = 1
        cv2.imshow(self._name, image)

    def set_mouse_callback(self, callback):
        cv2.setMouseCallback(self._name, callback)

    def clear_mouse_callback(self):

        def null_handler(event, x, y, flags, param):
            pass

        cv2.setMouseCallback(self._name, null_handler)
