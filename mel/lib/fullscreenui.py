"""Provide a full-screen UI."""


import contextlib

import cv2

import mel.lib.common
import mel.lib.image

_PYGAME_HAD_EXCLUSIVE_INIT = False


class AbortKeyInterruptError(Exception):
    pass


def yield_events_until_quit(quit_key="q", error_key=None, quit_func=None):
    # Import pygame as late as possible, to avoid displaying its
    # startup-text where it is not actually used.
    import pygame

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return
            yield event

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

    def show_opencv_image(self, image):
        image = mel.lib.image.letterbox(image, self.width, self.height)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.swapaxes(0, 1)
        image = self._pygame.surfarray.make_surface(image)
        self.surface.blit(image, [0, 0])


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


# -----------------------------------------------------------------------------
# Copyright (C) 2020 Angelos Evripiotis.
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
