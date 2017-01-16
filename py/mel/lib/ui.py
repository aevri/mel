"""User interface related things."""


import logging
import os
import subprocess
import tkinter

import cv2

import mel.lib.common
import mel.lib.image


# Codes as returned by waitkey().
# There may be some platform dependence, these codes were observed on Mac OSX.
WAITKEY_LEFT_ARROW = 63234
WAITKEY_RIGHT_ARROW = 63235
WAITKEY_UP_ARROW = 63232
WAITKEY_DOWN_ARROW = 63233

WAITKEY_BACKSPACE = 127

WAITKEY_ARROWS = [
    WAITKEY_LEFT_ARROW,
    WAITKEY_RIGHT_ARROW,
    WAITKEY_UP_ARROW,
    WAITKEY_DOWN_ARROW,
]


def bring_python_to_front():
    """Bring Python windows to the front.

    Behavior of this function is undefined if there are multiple Python
    processes running.

    :returns: None

    """
    osascript = "/usr/bin/osascript"

    if not os.path.isfile(osascript):
        # This may be an unsupported platform
        logging.warning("Could not find 'osascript', unsupported platform?")
        return

    subprocess.call([
        osascript,
        '-e',
        'tell app "Finder" to set frontmost of process "Python" to true',
    ])


def set_clipboard_contents(text):
    """Set the contents of the clipbaord, only works on Mac OSX.

    :returns: None

    """
    pbcopy = "/usr/bin/pbcopy"

    if not os.path.isfile(pbcopy):
        raise NotImplementedError(
            '{} was not found, cannot write clipboard'.format(pbcopy))

    p = subprocess.Popen(
        [pbcopy], stdin=subprocess.PIPE, universal_newlines=True)
    p.communicate(input=text)


def guess_fullscreen_width_height():
    tkroot = tkinter.Tk()
    width = tkroot.winfo_screenwidth() - 50
    height = tkroot.winfo_screenheight() - 150
    return width, height


class MultiImageDisplay():

    def __init__(self, name, width=None, height=None):
        self._display = ImageDisplay(name, width, height)
        self._images_names = []

        self._border_width = 50

        self._layout = [[]]

    def add_image(self, image, name=None):
        self._images_names.append((image, name))
        index = len(self._images_names) - 1
        self._layout[-1].append(index)
        self.refresh()
        return index

    def new_row(self):
        assert self._layout[-1]
        self._layout.append([])

    def update_image(self, image, index):
        name = self._images_names[index][1]
        self._images_names[index] = (image, name)
        self.refresh()

    def refresh(self):
        row_image_list = []

        for row in self._layout:
            row_image = None
            for index in row:
                image, _ = self._images_names[index]
                if row_image is None:
                    row_image = image
                else:
                    row_image = mel.lib.image.montage_horizontal(
                        self._border_width, row_image, image)
            row_image_list.append(row_image)

        if len(row_image_list) == 1:
            montage_image = row_image_list[0]
        else:
            montage_image = mel.lib.image.montage_vertical(
                0, *row_image_list)

        self._display.show_image(montage_image)


class ImageDisplay():
    """Display an image, centered in a new window."""

    def __init__(self, name, width=None, height=None):
        self.name = name

        if width is None or height is None:
            full_width_height = guess_fullscreen_width_height()
            if width is None:
                width = full_width_height[0]
            if height is None:
                height = full_width_height[1]

        self.width = width
        self.height = height
        self.original_width = None
        self.original_height = None
        self.image = None

        cv2.namedWindow(name)

        # If we don't do this apparently useless re-creation and resize then
        # the window appears under the dock in Mac OSX.
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, self.width, self.height)

        # This doesn't seem to work, fails with:
        #   error: (-27) NULL window in function cvSetModeWindow_COCOA
        #
        # cv2.setWindowProperty(
        #     "Name", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.show_image(
            mel.lib.common.new_image(
                self.width, self.height))

    def show_image(self, image):
        self.original_height, self.original_width = image.shape[:2]
        self.image = mel.lib.image.letterbox(image, self.width, self.height)
        cv2.imshow(self.name, self.image)

    def image_to_screen(self, x, y):
        return mel.lib.image.calc_original_to_letterboxed(
            x, y,
            self.width, self.height,
            self.original_width, self.original_height)

    def screen_to_image(self, x, y):
        return mel.lib.image.calc_letterboxed_to_original(
            x, y,
            self.width, self.height,
            self.original_width, self.original_height)

    def set_title(self, title):
        cv2.setWindowTitle(self.name, title)


class LeftRightDisplay():
    """Display images in a window, supply controls for navigating."""

    def __init__(self, name, image_list, width=None, height=None):
        if not image_list:
            raise ValueError(
                "image_list must be a list with at least one image.")

        self._original_title = name
        self.display = ImageDisplay(name, width, height)
        self._image_list = image_list
        self._index = 0
        self.show()

    def next_image(self):
        if self._image_list:
            self._index = (self._index + 1) % len(self._image_list)
        self.show()

    def prev_image(self):
        if self._image_list:
            num_images = len(self._image_list)
            self._index = (self._index + num_images -
                           1) % len(self._image_list)
        self.show()

    def _get_image(self, path):
        return cv2.imread(path)

    def show(self):
        if self._image_list:
            path = self._image_list[self._index]
            self.display.show_image(
                self._get_image(path))
            self.display.set_title(path)
        else:
            self.display.show_image(
                mel.lib.common.new_image(
                    self.display.height,
                    self.display.width))
            self.display.set_title(self._original_title)

    def set_mouse_callback(self, callback):
        cv2.setMouseCallback(self.display.name, callback)

    def clear_mouse_callback(self):
        cv2.setMouseCallback(
            self.display.name,
            mel.lib.common.make_null_mouse_callback())
