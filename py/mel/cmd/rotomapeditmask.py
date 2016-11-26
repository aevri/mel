"""Edit mask images for rotomap images."""

import os

import cv2
import numpy

import mel.lib.common
import mel.lib.fs
import mel.lib.ui


def setup_parser(parser):
    parser.add_argument(
        'IMAGES',
        nargs='+',
        help="A list of paths to images sets or images.")
    parser.add_argument(
        '--display-width',
        type=int,
        default=None,
        help="Width of the preview display window.")
    parser.add_argument(
        '--display-height',
        type=int,
        default=None,
        help="Width of the preview display window.")


def process_args(args):

    imager = MultiMaskImager()

    display = MaskDisplay(
        "rotomap-editmask",
        mel.lib.fs.expand_dirs_to_jpegs(args.IMAGES),
        imager,
        args.display_width,
        args.display_height)

    mel.lib.ui.bring_python_to_front()

    def mouse_callback(event, mouse_x, mouse_y, flags, _param):
        del _param
        if event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                enable = not(flags & cv2.EVENT_FLAG_CTRLKEY)
                image_x, image_y = display.display.screen_to_image(
                    mouse_x, mouse_y)
                screen_x, screen_y = display.display.image_to_screen(0, 0)
                imager.imager.draw_mask_circle(image_x, image_y, 100, enable)
                display.show()

    display.set_mouse_callback(mouse_callback)

    print("Press left arrow or right arrow to change image.")
    print("Press any other key to exit.")

    is_finished = False
    while not is_finished:
        key = cv2.waitKey(50)
        if key != -1:
            if key == mel.lib.ui.WAITKEY_RIGHT_ARROW:
                display.next_image()
            elif key == mel.lib.ui.WAITKEY_LEFT_ARROW:
                display.prev_image()
            else:
                is_finished = True

    display.clear_mouse_callback()

    imager.flush()


class MaskDisplay(mel.lib.ui.LeftRightDisplay):
    """Display images in a window, supply controls for masking."""

    def __init__(self, name, image_list, imager, width=None, height=None):
        self._imager = imager
        super().__init__(name, image_list, width, height)

    def _get_image(self, path):
        return self._imager.get_image(path)


class MultiMaskImager():
    """Manage multiple images."""

    def __init__(self):
        self.imager = None
        self._path = None

    def get_image(self, path):
        if self._path != path:
            if self.imager is not None:
                self.imager.save_mask()
            self.imager = MaskImager(path)
            self._path = path
        return self.imager.get_image()

    def flush(self):
        if self.imager is not None:
            self.imager.save_mask()


class MaskImager():
    """Composite mask and image, allow for saving and editing."""

    def __init__(self, path):
        self.path = path
        self.half_image = cv2.imread(self.path) // 2
        height, width = self.half_image.shape[:2]

        self.mask_path = self.path + '.mask.png'
        if os.path.isfile(self.mask_path):
            self.mask = cv2.imread(self.mask_path, -1)
            print(self.mask.shape)
        else:
            self.mask = numpy.zeros((height, width, 1), numpy.uint8)

    def draw_mask_circle(self, x, y, radius, enable=True):
        value = 255 if enable else 0
        cv2.circle(self.mask, (x, y), radius, value, -1)

    def save_mask(self):
        cv2.imwrite(self.mask_path, self.mask)

    def get_image(self):
        new_image = self.half_image[:]
        masked_image = cv2.bitwise_and(
            self.half_image, self.half_image, mask=self.mask)
        return cv2.add(masked_image, new_image)
