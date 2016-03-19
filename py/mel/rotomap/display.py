"""Display a rotomap."""

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


def load_image(path, rot90):
    image = cv2.imread(path)

    if rot90:
        image = mel.lib.common.rotated90(image, rot90)

    return image


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


def hex3_to_rgb4(hex_string):

    # "12 class paired" from http://colorbrewer2.org/
    scheme = [
        (166, 206, 227),
        (31, 120, 180),
        (178, 223, 138),
        (51, 160, 44),

        (251, 154, 153),
        (227, 26, 28),
        (253, 191, 111),
        (255, 127, 0),

        (202, 178, 214),
        (106, 61, 154),
        (255, 255, 153),
        (177, 89, 40),
    ]

    rgb_list = []

    value = int(hex_string[0:3], 16)
    for x in xrange(4):
        index = value % 12
        value //= 12
        rgb_list.append(scheme[index])

    return rgb_list


def draw_mole(image, x, y, mole):
    draw_target(image, x, y, mole)


def draw_target(image, x, y, mole):

    radius = 16
    colors = hex3_to_rgb4(mole['uuid'][:3])
    for index in xrange(3):
        cv2.circle(image, (x, y), radius, colors[index], -1)
        radius -= 4


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

        self._is_showing_markers = True

        self.show_current()

    def get_moles(self):
        return self._moles

    def set_moles(self, moles):
        self._moles = moles
        # self._save_image_moles()
        self.show_current()

    def get_image(self):
        return self._cached_image

    def toggle_markers(self):
        self._is_showing_markers = not self._is_showing_markers
        self.show_current()

    def load_current_image(self):

        if self._cached_image_index == self._list_index:
            return self._cached_image

        image_path = self._path_list[self._list_index]
        image = load_image(image_path, self._rot90)

        self._moles = load_image_moles(image_path)

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

        if self._is_showing_markers:
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
        if self._is_showing_markers:
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

    def _save_image_moles(self):
        image_path = self._path_list[self._list_index]
        save_image_moles(self._moles, image_path)

    def add_mole(self, x, y):
        self._moles.append({
            'x': x,
            'y': y,
            'uuid': uuid.uuid4().hex,
        })
        self._save_image_moles()
        self.show_current()

    def current_image_path(self):
        return self._path_list[self._list_index]

    def _closest_mole_index(self, x, y):
        closest_index = None
        closest_distance = None
        for i, mole in enumerate(self._moles):
            dx = x - mole['x']
            dy = y - mole['y']
            distance = math.sqrt(dx * dx + dy * dy)
            if closest_distance is None or distance < closest_distance:
                closest_index = i
                closest_distance = distance

        return closest_index

    def set_mole_uuid(self, x, y, mole_uuid):
        closest_index = self._closest_mole_index(x, y)
        if closest_index is not None:
            self._moles[closest_index]['uuid'] = mole_uuid
        self._save_image_moles()
        self.show_current()

    def get_mole_uuid(self, x, y):
        closest_index = self._closest_mole_index(x, y)
        if closest_index is not None:
            return self._moles[closest_index]['uuid']

        return None

    def remove_mole(self, x, y):
        closest_index = self._closest_mole_index(x, y)

        if closest_index is not None:
            del self._moles[closest_index]

        self._save_image_moles()
        self.show_current()

    def set_mouse_callback(self, callback):
        cv2.setMouseCallback(self._name, callback)

    def clear_mouse_callback(self):

        def null_handler(event, x, y, flags, param):
            pass

        cv2.setMouseCallback(self._name, null_handler)
