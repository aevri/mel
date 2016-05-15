"""User interface related things."""


import tkinter

import cv2

import mel.lib.image


def guess_fullscreen_width_height():
    tkroot = tkinter.Tk()
    width = tkroot.winfo_screenwidth() - 50
    height = tkroot.winfo_screenheight() - 150
    return width, height


class MultiImageDisplay():

    def __init__(self, name, width=None, height=None):
        self._name = name
        self._images_names = []

        self._border_width = 50

        self._layout = [[]]

        if width is None or height is None:
            full_width_height = mel.lib.ui.guess_fullscreen_width_height()
            if width is None:
                width = full_width_height[0]
            if height is None:
                height = full_width_height[1]

        self._width = width
        self._height = height

        cv2.namedWindow(name)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, self._width, self._height)

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
                image, name = self._images_names[index]
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

        montage_image = mel.lib.image.letterbox(
            montage_image, self._width, self._height)

        cv2.imshow(self._name, montage_image)
