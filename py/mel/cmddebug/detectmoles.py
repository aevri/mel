"""Provide a debugging view of mole detection in rotomaps."""


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

    mel.lib.ui.bring_python_to_front()

    def mouse_callback(event, mouse_x, mouse_y, flags, _param):
        del _param
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                editor.show_zoomed(mouse_x, mouse_y)

    editor.display.set_mouse_callback(mouse_callback)

    print("Press left for previous image, right for next image.")
    print("Ctrl-click on a point to zoom in on it.")
    print("Press space to restore original zoom.")
    print("Press enter to toggle mole markers.")
    print("Press 'm' to display mole detection results.")
    print("Press any other key to quit.")

    is_finished = False

    while not is_finished:
        key = cv2.waitKey(50)
        if key != -1:
            if key == mel.lib.ui.WAITKEY_LEFT_ARROW:
                editor.show_prev()
            elif key == mel.lib.ui.WAITKEY_RIGHT_ARROW:
                editor.show_next()
            elif key == ord(' '):
                editor.show_fitted()
            elif key == ord('m'):
                detect_moles(editor)
            elif key == 13:
                editor.toggle_markers()
            else:
                is_finished = True

    editor.display.clear_mouse_callback()


def detect_moles(editor):
    display = editor.display
    moledata = editor.moledata

    path = editor.moledata.current_image_path()
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    im = im[:, :, 1]

    mask = cv2.imread(path + '.mask.png', cv2.IMREAD_UNCHANGED)

    im = cv2.bitwise_and(im, im, mask=mask)
    im = cv2.bitwise_not(im, im)
    im = cv2.blur(im, (10, 10))
    # im = cv2.equalizeHist(im)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    params.minThreshold = 0
    params.maxThreshold = 256

    params.filterByArea = True
    params.minArea = 1500
    params.minArea = 50

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(im)
    im2 = cv2.drawKeypoints(
        im,
        keypoints,
        numpy.array([]),
        (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    display.show_current(im2, moledata.moles)
