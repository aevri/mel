"""Display a rotomap."""


import cv2

import mel.lib.common
import mel.lib.image
import mel.lib.math
import mel.lib.ui
import mel.rotomap.moles


def load_image(path):
    image = cv2.imread(path)

    return image


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
    for x in range(4):
        index = value % 12
        value //= 12
        rgb_list.append(scheme[index])

    return rgb_list


def draw_mole(image, x, y, mole):
    draw_target(image, x, y, mole)


def draw_target(image, x, y, mole):

    radius = 16
    colors = hex3_to_rgb4(mole['uuid'][:3])
    for index in range(3):
        cv2.circle(image, (x, y), radius, colors[index], -1)
        radius -= 4


class Display:

    def __init__(self, width, height):
        self._name = str(id(self))

        if width is None or height is None:
            full_width_height = mel.lib.ui.guess_fullscreen_width_height()
            if width is None:
                width = full_width_height[0]
            if height is None:
                height = full_width_height[1]

        self._width = width
        self._height = height

        cv2.namedWindow(self._name)
        cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._name, self._width, self._height)

        self._is_zoomed = False
        self._is_showing_markers = True
        self._is_faded_markers = True

    def toggle_markers(self):
        self._is_showing_markers = not self._is_showing_markers

    def toggle_faded_markers(self):
        self._is_faded_markers = not self._is_faded_markers

    def show_current(self, image, mole_list):
        if not self._is_zoomed:
            image = self._render_fitted_image(image)
        else:
            image = self._render_zoomed_image(
                image, self._zoom_x, self._zoom_y)

        if self._is_showing_markers:
            image = self._overlay_mole_markers(image, mole_list)

        cv2.imshow(self._name, image)

    def show_fitted(self, image, mole_list):
        self.set_fitted()
        self.show_current(image, mole_list)

    def show_zoomed(self, image, mole_list, x, y):
        self.set_zoomed(x, y)
        self.show_current(image, mole_list)

    def set_fitted(self):
        self._is_zoomed = False

    def set_zoomed(self, x, y):
        self._zoom_x = x
        self._zoom_y = y
        self._is_zoomed = True

    def _render_fitted_image(self, image):
        letterbox = mel.lib.image.calc_letterbox(
            image.shape[1],
            image.shape[0],
            self._width,
            self._height)

        self._image_left = letterbox[0]
        self._image_top = letterbox[1]
        self._image_scale = image.shape[1] / letterbox[2]

        image = mel.lib.image.letterbox(
            image, self._width, self._height)

        return image

    def _render_zoomed_image(self, image, x, y):
        left, top = mel.lib.image.calc_centering_offset(
            (x, y),
            (self._width, self._height))

        self._image_left = left
        self._image_top = top
        self._image_scale = 1

        image = mel.lib.image.centered_at(
            image, x, y, self._width, self._height)

        return image

    def _overlay_mole_markers(self, image, mole_list):
        marker_image = image
        if self._is_faded_markers:
            marker_image = image.copy()
        for mole in mole_list:
            x = int(mole['x'] / self._image_scale + self._image_left)
            y = int(mole['y'] / self._image_scale + self._image_top)
            draw_mole(marker_image, x, y, mole)
        if self._is_faded_markers:
            image = cv2.addWeighted(image, 0.75, marker_image, 0.25, 0.0)

        return image

    def set_mouse_callback(self, callback):
        cv2.setMouseCallback(self._name, callback)

    def clear_mouse_callback(self):

        def null_handler(event, x, y, flags, param):
            pass

        cv2.setMouseCallback(self._name, null_handler)

    def windowxy_to_imagexy(self, window_x, window_y):
        image_x = window_x - self._image_left
        image_y = window_y - self._image_top
        return (
            int(image_x * self._image_scale),
            int(image_y * self._image_scale)
        )


class Editor:

    def __init__(self, path_list_list, width, height):
        self.display = Display(width, height)
        self.moledata_list = [MoleData(x) for x in path_list_list]
        self.moledata_index = 0
        self.moledata = self.moledata_list[self.moledata_index]
        self.show_current()

    def set_moles(self, moles):
        self.moledata.moles = moles
        self.show_current()

    def toggle_markers(self):
        self.display.toggle_markers()
        self.show_current()

    def toggle_faded_markers(self):
        self.display.toggle_faded_markers()
        self.show_current()

    def show_current(self):
        image = self.moledata.get_image()
        self.display.show_current(image, self.moledata.moles)

    def show_fitted(self):
        image = self.moledata.get_image()
        self.display.show_fitted(image, self.moledata.moles)

    def show_zoomed(self, mouse_x, mouse_y):
        image_x, image_y = self.display.windowxy_to_imagexy(mouse_x, mouse_y)
        image = self.moledata.get_image()
        self.display.show_zoomed(image, self.moledata.moles, image_x, image_y)

    def show_prev_map(self):
        self.moledata_index -= 1
        self.moledata_index %= len(self.moledata_list)
        self.moledata = self.moledata_list[self.moledata_index]
        self.show_current()

    def show_next_map(self):
        self.moledata_index += 1
        self.moledata_index %= len(self.moledata_list)
        self.moledata = self.moledata_list[self.moledata_index]
        self.show_current()

    def show_prev(self):
        self.moledata.decrement()
        self.show_current()

    def show_next(self):
        self.moledata.increment()
        self.show_current()

    def add_mole(self, mouse_x, mouse_y):
        image_x, image_y = self.display.windowxy_to_imagexy(mouse_x, mouse_y)
        mel.rotomap.moles.add_mole(self.moledata.moles, image_x, image_y)
        self.moledata.save_moles()
        self.show_current()

    def set_mole_uuid(self, mouse_x, mouse_y, mole_uuid):
        image_x, image_y = self.display.windowxy_to_imagexy(mouse_x, mouse_y)
        mel.rotomap.moles.set_nearest_mole_uuid(
            self.moledata.moles, image_x, image_y, mole_uuid)
        self.moledata.save_moles()
        self.show_current()

    def get_mole_uuid(self, mouse_x, mouse_y):
        image_x, image_y = self.display.windowxy_to_imagexy(mouse_x, mouse_y)
        return mel.rotomap.moles.get_nearest_mole_uuid(
            self.moledata.moles, image_x, image_y)

    def move_nearest_mole(self, mouse_x, mouse_y):
        image_x, image_y = self.display.windowxy_to_imagexy(mouse_x, mouse_y)
        mel.rotomap.moles.move_nearest_mole(
            self.moledata.moles, image_x, image_y)
        self.moledata.save_moles()
        self.show_current()

    def remove_mole(self, mouse_x, mouse_y):
        image_x, image_y = self.display.windowxy_to_imagexy(mouse_x, mouse_y)
        mel.rotomap.moles.remove_nearest_mole(
            self.moledata.moles, image_x, image_y)
        self.moledata.save_moles()
        self.show_current()


class MoleData:

    def __init__(self, path_list):
        self.moles = []
        self._path_list = path_list
        self._list_index = 0
        self._num_images = len(self._path_list)
        self._cached_image = None
        self._cached_image_index = None

    def get_image(self):
        return self._load()

    def _load(self):

        if self._cached_image_index == self._list_index:
            return self._cached_image

        image_path = self._path_list[self._list_index]
        image = load_image(image_path)

        self.moles = mel.rotomap.moles.load_image_moles(image_path)

        self._cached_image_index = self._list_index
        self._cached_image = image

        return image

    def decrement(self):
        new_index = self._list_index + self._num_images - 1
        self._list_index = new_index % self._num_images

    def increment(self):
        self._list_index = (self._list_index + 1) % self._num_images

    def save_moles(self):
        image_path = self._path_list[self._list_index]
        mel.rotomap.moles.save_image_moles(self.moles, image_path)

    def current_image_path(self):
        return self._path_list[self._list_index]
