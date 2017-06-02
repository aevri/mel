"""A global object for debug rendering into images without around."""

import contextlib

import cv2


class GlobalContext():

    def __init__(self):
        self._image = None

    @contextlib.contextmanager
    def image_context(self, image):
        if self._image is not None:
            raise Exception('Nested image context not yet supported')
        try:
            self._image = image
            yield
        finally:
            self._image = None

    def arrow(self, from_, to):
        if self._image is None:
            return
        cv2.arrowedLine(
            self._image,
            tuple(from_.astype(int)),
            tuple(to.astype(int)),
            (255, 255, 255),
            2,
            cv2.LINE_AA)

    def circle(self, point, radius):
        if self._image is None:
            return
        cv2.circle(
            self._image,
            tuple(point.astype(int)),
            int(radius),
            (255, 255, 255),
            2)
