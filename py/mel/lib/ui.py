"""User interface related things."""


import logging
import os
import subprocess
import tkinter

import cv2

import mel.lib.image


# Codes as returned by waitkey().
# There may be some platform dependence, these codes were observed on Mac OSX.
WAITKEY_LEFT_ARROW = 63234
WAITKEY_RIGHT_ARROW = 63235
WAITKEY_UP_ARROW = 63232
WAITKEY_DOWN_ARROW = 63233


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
            full_width_height = mel.lib.ui.guess_fullscreen_width_height()
            if width is None:
                width = full_width_height[0]
            if height is None:
                height = full_width_height[1]

        self.width = width
        self.height = height
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
        self.image = mel.lib.image.letterbox(image, self.width, self.height)
        cv2.imshow(self.name, self.image)
