"""Edit a 'rotomap' series of images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import uuid

import cv2

import mel.lib.common
import mel.lib.image
import mel.lib.math


def setup_parser(parser):
    parser.add_argument(
        'PATH',
        type=str,
        help="Path to the rotomap image directory.")
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
            elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                image_x, image_y = display.windowxy_to_imagexy(x, y)
                display.remove_mole(image_x, image_y)
            else:
                image_x, image_y = display.windowxy_to_imagexy(x, y)
                display.add_mole(image_x, image_y)

    display.set_mouse_callback(mouse_callback)

    print("Press left for previous image, right for next image.")
    print("Click on a point to add a mole there and save.")
    print("Ctrl-click on a point to zoom in on it.")
    print("Press space to restore original zoom.")
    print("Press any other key to quit.")

    is_finished = False
    while not is_finished:
        key = cv2.waitKey(50)
        if key != -1:
            if key == left:
                display.show_prev()
                print(display.current_image_path())
            elif key == right:
                display.show_next()
                print(display.current_image_path())
            elif key == ord(' '):
                display.show_fitted()
            else:
                is_finished = True

    display.clear_mouse_callback()


def load_image_moles(image_path):
    moles_path = image_path + '.json'
    moles = []
    if os.path.exists(moles_path):
        with open(moles_path) as moles_file:
            moles = json.load(moles_file)

    converted = []
    for m in moles:
        if type(m) is list:
            m = {'x': m[0], 'y': m[1]}
        if 'uuid' not in m:
            m['uuid'] = uuid.uuid4().hex
        converted.append(m)

    return converted


def save_image_moles(moles, image_path):
    moles_path = image_path + '.json'
    with open(moles_path, 'w') as moles_file:
        json.dump(
            moles,
            moles_file,
            indent=4,
            separators=(',', ': '),
            sort_keys=True)

        # There's no newline after dump(), add one here for happier viewing
        print(file=moles_file)


def draw_mole(image, x, y, mole):
    r = int(mole['uuid'][0:2], 16)
    g = int(mole['uuid'][2:4], 16)
    b = int(mole['uuid'][4:6], 16)
    cv2.circle(image, (x, y), 10, (r, g, b), -1)


class Display:

    def __init__(self, path, width, height, rot90):
        self._name = path
        self._width = width
        self._height = height
        self._rot90 = rot90

        self._moles = []

        # list all images
        self._path_list = [
            os.path.join(path, x)
            for x in os.listdir(path)
            if x.endswith('.jpg')
        ]

        cv2.namedWindow(self._name)
        cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._name, self._width, self._height)

        self._list_index = 0
        self._num_images = len(self._path_list)

        self._is_zoomed = False

        self._cached_image = None
        self._cached_image_index = None

        self.show_current()

    def load_current_image(self):

        if self._cached_image_index == self._list_index:
            return self._cached_image

        image_path = self._path_list[self._list_index]
        image = cv2.imread(image_path)

        self._moles = load_image_moles(image_path)

        if self._rot90:
            image = mel.lib.common.rotated90(image, self._rot90)

        self._cached_image_index = self._list_index
        self._cached_image = image

        return image

    def show_current(self):
        if not self._is_zoomed:
            self.show_fitted()
        else:
            self.show_zoomed(self._zoom_x, self._zoom_y)

    def show_fitted(self):
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

        for mole in self._moles:
            x = int(mole['x'] / self._image_scale + self._image_left)
            y = int(mole['y'] / self._image_scale + self._image_top)
            draw_mole(image, x, y, mole)

        cv2.imshow(self._name, image)
        self._is_zoomed = False

    def show_zoomed(self, x, y):
        image = self.load_current_image()
        nx, ny = mel.lib.image.calc_centering_offset(
            (x, y),
            (image.shape[1], image.shape[0]),
            (self._width, self._height))
        image = mel.lib.image.translated_and_clipped(
            image, nx, ny, self._width, self._height)
        self._zoom_x = x
        self._zoom_y = y
        self._image_left = -nx
        self._image_top = -ny
        self._image_width = image.shape[1] + nx
        self._image_height = image.shape[0] + ny
        self._image_scale = 1
        for mole in self._moles:
            x = mole['x'] + self._image_left
            y = mole['y'] + self._image_top
            if x >= 0 and y >= 0:
                if x < self._image_width and y < self._image_height:
                    draw_mole(image, x, y, mole)
        cv2.imshow(self._name, image)
        self._is_zoomed = True

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

    def add_mole(self, x, y):
        self._moles.append({
            'x': x,
            'y': y,
            'uuid': uuid.uuid4().hex,
        })
        image_path = self._path_list[self._list_index]
        save_image_moles(self._moles, image_path)
        self.show_current()

    def current_image_path(self):
        return self._path_list[self._list_index]

    def remove_mole(self, x, y):
        closest_index = None
        closest_distance = None
        for i, mole in enumerate(self._moles):
            dx = x - mole['x']
            dy = y - mole['y']
            distance = math.sqrt(dx * dx + dy * dy)
            if closest_distance is None or distance < closest_distance:
                closest_index = i
                closest_distance = distance

        if closest_index is not None:
            del self._moles[closest_index]

        image_path = self._path_list[self._list_index]
        save_image_moles(self._moles, image_path)

        self.show_current()

    def set_mouse_callback(self, callback):
        cv2.setMouseCallback(self._name, callback)

    def clear_mouse_callback(self):

        def null_handler(event, x, y, flags, param):
            pass

        cv2.setMouseCallback(self._name, null_handler)
