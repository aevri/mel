"""A tool for adding a new cluster / constellation from photographs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy


def setup_parser(parser):

    parser.add_argument(
        'context',
        type=str,
        default=None,
        help="Path to the context image to add.")

    parser.add_argument(
        'detail',
        type=str,
        default=None,
        help="Path to the detail image to add.")

    parser.add_argument(
        'destination',
        type=str,
        default=None,
        help="New path to create and store the constellation to.")

    parser.add_argument(
        'moles',
        type=str,
        default=None,
        nargs='+',
        help="Names of the moles to store.")


def process_args(args):
    context_image = cv2.imread(args.context)
    detail_image = cv2.imread(args.detail)

    # print out the dimensions of the images
    print('{}: {}'.format(args.context, context_image.shape))
    print('{}: {}'.format(args.detail, detail_image.shape))

    # display the context image in a reasonably sized window
    cv2.namedWindow('display', cv2.WINDOW_NORMAL)
    window_width = 800
    window_height = 600
    cv2.resizeWindow('display', window_width, window_height)

    # get the user to mark the mole positions
    context_mole_positions, detail_mole_positions = _user_mark_moles(
        'display', context_image, detail_image, args.moles)

    # Put a box around moles on context image
    _box_moles(context_image, context_mole_positions, thickness=50)
    cv2.imshow('display', context_image)
    print("Press any key to continue.")
    cv2.waitKey()

    # TODO: connect moles on cluster detail image
    # TODO: combine context image with cluster detail image to make montage

    # TODO: point to moles on individual detail images

    cv2.destroyAllWindows()
    raise NotImplementedError()


def _user_mark_moles(window_name, context_image, detail_image, moles):

    display_image = numpy.copy(context_image)
    cv2.imshow(window_name, display_image)

    circle_radius = 50

    context_mole_positions = []
    detail_mole_positions = []
    current_mole_positions = context_mole_positions

    cv2.setMouseCallback(
        window_name,
        _make_mole_capture_callback(
            window_name,
            display_image,
            circle_radius,
            context_mole_positions))

    # main loop
    print('Please mark all specified moles, double-click to mark.')
    print('Press any key to abort.')

    is_finished = False
    while not is_finished:
        key = cv2.waitKey(50)

        if key != -1:
            raise Exception('User aborted.')

        if len(current_mole_positions) == len(moles):
            if not detail_mole_positions:
                current_mole_positions = detail_mole_positions
                display_image = numpy.copy(detail_image)
                cv2.setMouseCallback(
                    window_name,
                    _make_mole_capture_callback(
                        window_name,
                        display_image,
                        circle_radius,
                        detail_mole_positions))
                cv2.imshow(window_name, display_image)
            else:
                print("context positions:")
                print(context_mole_positions)
                print("detail positions:")
                print(detail_mole_positions)
                is_finished = True

    return context_mole_positions, detail_mole_positions


def _make_mole_capture_callback(window_name, image, radius, mole_positions):

    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(image, (x, y), radius, (255, 0, 0), -1)
            mole_positions.append((x, y, radius))
            cv2.imshow(window_name, image)

    return draw_circle


def _box_moles(image, mole_positions, thickness):
    left = min((m[0] - m[2] for m in mole_positions))
    top = min((m[1] - m[2] for m in mole_positions))
    right = max((m[0] + m[2] for m in mole_positions))
    bottom = max((m[1] + m[2] for m in mole_positions))

    left -= 2 * thickness
    top -= 2 * thickness
    right += 2 * thickness
    bottom += 2 * thickness

    left_top = (left, top)
    right_bottom = (right, bottom)

    blue = (255, 0, 0)
    cv2.rectangle(image, left_top, right_bottom, blue, thickness)
