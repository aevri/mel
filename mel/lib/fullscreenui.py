"""Provide a full-screen UI."""


import contextlib

import cv2

import mel.lib.common
import mel.lib.image
import mel.lib.ui

_PYGAME_HAD_EXCLUSIVE_INIT = False


def yield_frames_keys(video_capture, display, error_key):
    # Import pygame as late as possible, to avoid displaying its
    # startup-text where it is not actually used.
    import pygame

    while True:
        ret, frame = video_capture.read()
        if not ret:
            raise Exception("Could not read frame.")

        keys = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                keys.append(event.key)

        display.update_screen_if_needed()

        if keys:
            for key in keys:
                if key == error_key:
                    raise mel.lib.ui.AbortKeyInterruptError()
                else:
                    yield frame, key
        else:
            yield frame, None


def yield_events_until_quit(
    display, *, quit_key=None, quit_func=None, error_key=None
):
    # Import pygame as late as possible, to avoid displaying its
    # startup-text where it is not actually used.
    import pygame

    if quit_key is None:
        quit_key = pygame.K_q

    display.update_screen_if_needed()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == quit_key:
                    return
                elif error_key is not None and event.key == error_key:
                    raise mel.lib.ui.AbortKeyInterruptError()
            yield event
            display.update_screen_if_needed()

        if quit_func is not None and quit_func():
            return


@contextlib.contextmanager
def fullscreen_context():
    """Initialise and return a fullscreen surface for drawing onto.

    Shutdown the underlying `pygame` library when the context expires.

    Only one context can exist per run of the application. Once `pygame` has
    been shutdown then it cannot be initialised again.
    """
    # Import pygame as late as possible, to avoid displaying its
    # startup-text where it is not actually used.
    import pygame

    global _PYGAME_HAD_EXCLUSIVE_INIT
    if _PYGAME_HAD_EXCLUSIVE_INIT:
        raise Exception(
            "An exclusive context was already started, only 1 per run."
        )
    _PYGAME_HAD_EXCLUSIVE_INIT = True

    pygame.init()
    pygame.display.set_caption("mel")
    surface = pygame.display.set_mode([0, 0], pygame.FULLSCREEN)
    try:
        yield Display(surface)
    finally:
        pygame.quit()


class Display:
    """Display an opencv image, centered in a surface."""

    def __init__(self, surface):
        # Import pygame as late as possible, to avoid displaying its
        # startup-text where it is not actually used.
        import pygame

        self._pygame = pygame
        self.surface = surface
        self.width = surface.get_width()
        self.height = surface.get_height()
        self.surface.fill((0, 0, 0))
        self.is_dirty = True

    def show_opencv_image(self, image):
        image = mel.lib.image.letterbox(image, self.width, self.height)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.swapaxes(0, 1)
        image = self._pygame.surfarray.make_surface(image)
        self.surface.blit(image, [0, 0])
        self.is_dirty = True

    def update_screen_if_needed(self):
        # Import pygame as late as possible, to avoid displaying its
        # startup-text where it is not actually used.
        import pygame

        if self.is_dirty:
            pygame.display.update()
        self.is_dirty = False


class LeftRightDisplay:
    """Display images in a window, supply controls for navigating."""

    def __init__(self, screen, image_list):
        if not image_list:
            raise ValueError(
                "image_list must be a list with at least one image."
            )

        self.display = screen
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
            self._index = (self._index + num_images - 1) % len(
                self._image_list
            )
        self.show()

    def _get_image(self, path):
        return mel.lib.image.load_image(path)

    def show(self):
        if self._image_list:
            path = self._image_list[self._index]
            self.display.show_opencv_image(self._get_image(path))
        else:
            self.display.show_opencv_image(
                mel.lib.common.new_image(
                    self.display.height, self.display.width
                )
            )


class MultiImageDisplay:
    def __init__(self, display):
        self._display = display
        self.reset()

    def reset(self):
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
                        self._border_width, row_image, image
                    )
            row_image_list.append(row_image)

        if len(row_image_list) == 1:
            montage_image = row_image_list[0]
        else:
            montage_image = mel.lib.image.montage_vertical(0, *row_image_list)

        self._display.show_opencv_image(montage_image)


# -----------------------------------------------------------------------------
# Copyright (C) 2020-2021 Angelos Evripiotis.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------ END-OF-FILE ----------------------------------
