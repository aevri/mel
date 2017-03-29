"""Provide a debugging view of triangulation in rotomaps."""


import cv2
import numpy

import mel.lib.common
import mel.lib.image
import mel.lib.math
import mel.lib.ui

import mel.rotomap.display


def setup_parser(parser):
    parser.add_argument(
        '--images',
        nargs='+',
        action='append',
        required=True,
        help="A list of paths to images, specify multiple times for multiple "
             "sets.")
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

    editor = mel.rotomap.display.Editor(
        args.images, args.display_width, args.display_height)

    def mouse_callback(event, mouse_x, mouse_y, flags, _param):
        del _param
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                editor.show_zoomed(mouse_x, mouse_y)
            else:
                show_best_moles(editor, mouse_x, mouse_y)

    editor.display.set_mouse_callback(mouse_callback)

    print("Press left for previous image, right for next image.")
    print("Press up for previous map, down for next map.")
    print("Ctrl-click on a point to zoom in on it.")
    print("Press space to restore original zoom.")
    print("Press enter to toggle mole markers.")
    print("Press any other key to quit.")

    for key in mel.lib.ui.yield_keys_until_quitkey():
        if key == mel.lib.ui.WAITKEY_LEFT_ARROW:
            editor.show_prev()
            print(editor.moledata.current_image_path())
        elif key == mel.lib.ui.WAITKEY_RIGHT_ARROW:
            editor.show_next()
            print(editor.moledata.current_image_path())
        elif key == mel.lib.ui.WAITKEY_UP_ARROW:
            editor.show_prev_map()
            print(editor.moledata.current_image_path())
        elif key == mel.lib.ui.WAITKEY_DOWN_ARROW:
            editor.show_next_map()
            print(editor.moledata.current_image_path())
        elif key == ord(' '):
            editor.show_fitted()
        elif key == 13:
            editor.toggle_markers()

    editor.display.clear_mouse_callback()


def show_best_moles(editor, x, y):
    display = editor.display
    moledata = editor.moledata
    image = moledata.get_image().copy()

    molepoint = numpy.array(
        display.windowxy_to_imagexy(x, y))

    image_rect = (0, 0, image.shape[1], image.shape[0])
    moles_for_mapping = mel.rotomap.moles.get_best_moles_for_mapping(
        molepoint, moledata.moles, image_rect)

    for mole in moles_for_mapping:
        drawpoint = (mole['x'], mole['y'])
        cv2.circle(image, drawpoint, 64, (255, 255, 255), -1)

    mel.rotomap.moles.mapped_pos(
        molepoint, moles_for_mapping, moledata.moles)

    display.show_current(image, moledata.moles)
