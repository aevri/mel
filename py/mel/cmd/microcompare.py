"""Compare existing microscope images of a mole."""

import cv2
import os

import mel.lib.common
import mel.lib.datetime
import mel.lib.image
import mel.lib.moleimaging
import mel.lib.ui


def setup_parser(parser):
    parser.add_argument(
        'PATH',
        type=str,
        help="Path to the mole to compare images from.")
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


def get_comparison_images(path):

    micro_path = os.path.join(path, '__micro__')

    # List all the 'jpg' files in the micro dir
    # TODO: support more than just '.jpg'
    names = [x for x in os.listdir(micro_path) if x.lower().endswith('.jpg')]
    paths = [os.path.join(micro_path, x) for x in names]
    images = [cv2.imread(x) for x in paths]

    for path, img in zip(paths, images):
        if img is None:
            raise ValueError("Failed to load file: {}".format(path))

    return images


def process_args(args):
    width = args.display_width
    height = args.display_height

    images = get_comparison_images(args.PATH)

    display = mel.lib.ui.ImageDisplay(args.PATH, width, height)

    index = 0

    if images:
        display.show_image(images[index])
        pass

    print("Press left arrow or right arrow to display another image.")
    print("Press any other key to exit.")

    is_finished = False
    while not is_finished:
        key = cv2.waitKey(50)
        if key != -1:
            if key == mel.lib.ui.WAITKEY_RIGHT_ARROW:
                index = (index + 1) % len(images)
                display.show_image(images[index])
            elif key == mel.lib.ui.WAITKEY_LEFT_ARROW:
                index = (index + len(images) - 1) % len(images)
                display.show_image(images[index])
            else:
                is_finished = True
