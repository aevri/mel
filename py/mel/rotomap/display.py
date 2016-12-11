"""Display a rotomap."""


import cv2
import numpy

import mel.lib.common
import mel.lib.image
import mel.lib.math
import mel.lib.ui
import mel.rotomap.moles
import mel.rotomap.tricolour


def load_image(path):
    image = cv2.imread(path)

    return image


def draw_mole(image, x, y, colours):

    radius = 16
    for index in range(3):
        cv2.circle(image, (x, y), radius, colours[index], -1)
        radius -= 4


def draw_non_canonical_mole(image, x, y, colours):
    radius = 16
    thickness = 4
    for index in range(3):
        top_left = (x - radius, y - radius)
        bottom_right = (x + radius, y + radius)
        cv2.rectangle(image, top_left, bottom_right, colours[index], thickness)
        radius -= thickness


def draw_crosshair(image, x, y):
    inner_radius = 16
    outer_radius = 24
    white = (255, 255, 255)
    black = (0, 0, 0)

    directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # Right, down, left, up

    size_color_list = [(3, white), (2, black)]

    for size, color in size_color_list:
        for d in directions:
            cv2.line(
                image,
                (x + (inner_radius * d[0]), y + (inner_radius * d[1])),
                (x + (outer_radius * d[0]), y + (outer_radius * d[1])),
                color,
                size)


class Display:

    def __init__(self, width, height, uuid_to_tricolour=None):
        self._name = str(id(self))

        if width is None or height is None:
            full_width_height = mel.lib.ui.guess_fullscreen_width_height()
            if width is None:
                width = full_width_height[0]
            if height is None:
                height = full_width_height[1]

        self._rect = numpy.array((width, height))

        cv2.namedWindow(self._name)
        cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._name, *self._rect)

        self._uuid_to_tricolour = uuid_to_tricolour
        if self._uuid_to_tricolour is None:
            self._uuid_to_tricolour = (
                mel.rotomap.tricolour.uuid_to_tricolour_first_digits)

        self._transform = None

        self._zoom_pos = None
        self._is_zoomed = False

        self._is_showing_markers = True
        self._is_faded_markers = True
        self._highlight_uuid = None

    def toggle_markers(self):
        self._is_showing_markers = not self._is_showing_markers

    def set_highlight_uuid(self, highlight_uuid):
        self._highlight_uuid = highlight_uuid

    def toggle_faded_markers(self):
        self._is_faded_markers = not self._is_faded_markers

    def show_current(self, image, mole_list):

        if not self._is_zoomed:
            self._transform = FittedImageTransform(
                image, self._rect)
        else:
            self._transform = ZoomedImageTransform(
                image, self._zoom_pos, self._rect)

        image = self._transform.render()

        highlight_mole = None
        if self._highlight_uuid is not None:
            for m in mole_list:
                if m['uuid'] == self._highlight_uuid:
                    highlight_mole = m
                    break

        if self._is_showing_markers:
            image = self._overlay_mole_markers(
                image, mole_list, highlight_mole)

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
        self._zoom_pos = numpy.array((x, y))
        self._is_zoomed = True

    def _overlay_mole_markers(self, image, mole_list, highlight_mole):
        marker_image = image
        if self._is_faded_markers:
            marker_image = image.copy()
        for mole in mole_list:
            x, y = self._transform.imagexy_to_transformedxy(
                mole['x'], mole['y'])
            if mole is highlight_mole:
                draw_crosshair(marker_image, x, y)
            colours = self._uuid_to_tricolour(mole['uuid'])
            if mole['is_uuid_canonical']:
                draw_mole(marker_image, x, y, colours)
            else:
                draw_non_canonical_mole(marker_image, x, y, colours)
        if self._is_faded_markers:
            image = cv2.addWeighted(image, 0.75, marker_image, 0.25, 0.0)

        return image

    def set_mouse_callback(self, callback):
        cv2.setMouseCallback(self._name, callback)

    def clear_mouse_callback(self):
        cv2.setMouseCallback(
            self._name,
            mel.lib.common.make_null_mouse_callback())

    def windowxy_to_imagexy(self, window_x, window_y):
        return self._transform.transformedxy_to_imagexy(window_x, window_y)

    def set_title(self, title):
        cv2.setWindowTitle(self._name, title)


class ZoomedImageTransform():

    def __init__(self, image, pos, rect):
        self._pos = pos
        self._rect = rect
        self._offset = mel.lib.image.calc_centering_offset(pos, rect)

        self._image = image

    def render(self):

        return mel.lib.image.centered_at(
            self._image,
            self._pos,
            self._rect)

    def imagexy_to_transformedxy(self, x, y):
        return numpy.array((x, y)) + self._offset

    def transformedxy_to_imagexy(self, x, y):
        return numpy.array((x, y)) - self._offset


class FittedImageTransform():

    def __init__(self, image, fit_rect):
        self._fit_rect = fit_rect
        image_rect = mel.lib.image.get_image_rect(image)

        letterbox = mel.lib.image.calc_letterbox(
            *image_rect, *self._fit_rect)

        self._offset = numpy.array(letterbox[:2])
        self._scale = image.shape[1] / letterbox[2]

        self._image = image

    def render(self):
        return mel.lib.image.letterbox(
            self._image, *self._fit_rect)

    def imagexy_to_transformedxy(self, x, y):
        return (numpy.array((x, y)) / self._scale + self._offset).astype(int)

    def transformedxy_to_imagexy(self, x, y):
        return ((numpy.array((x, y)) - self._offset) * self._scale).astype(int)


class Editor:

    def __init__(self, path_list_list, width, height):
        self._uuid_to_tricolour = mel.rotomap.tricolour.UuidTriColourPicker()
        self.display = Display(width, height, self._uuid_to_tricolour)
        self.moledata_list = [MoleData(x) for x in path_list_list]

        self.moledata_index = 0
        self.moledata = self.moledata_list[self.moledata_index]
        self._follow = None
        self.show_current()

    def set_moles(self, moles):
        self.moledata.moles = moles
        self.show_current()

    def follow(self, uuid_to_follow):
        self._follow = uuid_to_follow
        self.display.set_highlight_uuid(self._follow)

        follow_mole = None
        for m in self.moledata.moles:
            if m['uuid'] == self._follow:
                follow_mole = m
                break

        if follow_mole is not None:
            image = self.moledata.get_image()
            self.display.show_zoomed(
                image, self.moledata.moles, follow_mole['x'], follow_mole['y'])
        else:
            self.show_fitted()

    def toggle_markers(self):
        self.display.toggle_markers()
        self.show_current()

    def toggle_faded_markers(self):
        self.display.toggle_faded_markers()
        self.show_current()

    def show_current(self):
        image = self.moledata.get_image()
        self.display.show_current(image, self.moledata.moles)
        self.display.set_title(self.moledata.current_image_path())

    def show_fitted(self):
        image = self.moledata.get_image()
        self.display.show_fitted(image, self.moledata.moles)

    def show_zoomed(self, mouse_x, mouse_y):
        image_x, image_y = self.display.windowxy_to_imagexy(mouse_x, mouse_y)
        image = self.moledata.get_image()
        self.display.show_zoomed(image, self.moledata.moles, image_x, image_y)

    def show_zoomed_display(self, image_x, image_y):
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

    def add_mole_display(self, image_x, image_y, mole_uuid=None):
        mel.rotomap.moles.add_mole(
            self.moledata.moles, image_x, image_y, mole_uuid)
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

    def crud_mole(self, mole_uuid, mouse_x, mouse_y):
        image_x, image_y = self.display.windowxy_to_imagexy(mouse_x, mouse_y)

        i = mel.rotomap.moles.uuid_mole_index(self.moledata.moles, mole_uuid)
        if i is not None:
            self.moledata.moles[i]['x'] = image_x
            self.moledata.moles[i]['y'] = image_y
        else:
            mel.rotomap.moles.add_mole(
                self.moledata.moles, image_x, image_y, mole_uuid)

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
        self._load()

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
