"""Edit mask images for rotomap images."""

import cv2

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
        if event == cv2.EVENT_LBUTTONDOWN:
            print(mouse_x, mouse_y)

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
        pass

    def get_image(self, path):
        pass

    # def get_current_imager(self):
    #     pass

    def flush(self):
        pass


# class MaskImager():
    # """Composite mask and image, allow for saving and editing."""

    # def __init__(self, path):
    #     pass

    # def draw_mask_circle(self, x, y, radius, enable=True):
    #     pass

    # def save_mask(self):
    #     pass

    # def get_image(self):
    #     pass
