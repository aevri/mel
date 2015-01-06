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

    # Connect moles on cluster detail image
    cluster_detail_image = numpy.copy(detail_image)
    _connect_moles(cluster_detail_image, detail_mole_positions)
    cv2.imshow('display', cluster_detail_image)
    print("Press any key to continue.")
    cv2.waitKey()

    # Combine context image with cluster detail image to make montage
    cluster_monatage_image = _montage_horizontal(
        context_image, cluster_detail_image)
    cv2.imshow('display', cluster_monatage_image)
    print("Press any key to continue.")
    cv2.waitKey()

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


def _connect_moles(image, mole_positions):
    for mole_a, mole_b in _yield_neighbors(mole_positions):
        thickness = max(mole_a[2], mole_b[2])

        # draw connection
        a = numpy.array(mole_a[:2])
        b = numpy.array(mole_b[:2])
        a_to_b = numpy.linalg.norm(b - a)
        a_to_b = a_to_b / numpy.linalg.norm(a_to_b)
        padding = a_to_b * thickness
        a += padding
        b -= padding
        a = tuple(a.tolist())
        b = tuple(b.tolist())
        blue = (255, 0, 0)
        print(a_to_b, a, b, thickness)
        cv2.line(image, a, b, blue, thickness)


def _yield_neighbors(node_list):
    is_first = True
    prev_node = None
    for node in node_list:
        if is_first:
            is_first = False
        else:
            yield (prev_node, node)
        prev_node = node


def _montage_horizontal(left_image, right_image):
    if left_image.shape != right_image.shape:
        raise ValueError('image shapes must be identical')

    # calculate the bounds of the montage
    border_size = 50
    total_border_size = numpy.array([border_size * 2, border_size * 3])
    image_shape = left_image.shape
    total_image_size = numpy.array([image_shape[0], image_shape[1] * 2])
    total_montage_size = total_border_size + total_image_size

    montage_image = _new_image(total_montage_size[0], total_montage_size[1])

    # write the images into the montage image
    x = border_size
    _copy_image_into_image(left_image, montage_image, border_size, x)
    x += border_size
    x += image_shape[1]
    _copy_image_into_image(right_image, montage_image, border_size, x)

    return montage_image


def _new_image(height, width):
    return numpy.zeros((height, width, 3), numpy.uint8)


def _copy_image_into_image(source, dest, y, x):
    shape = source.shape
    dest[y:(y + shape[0]), x:(x + shape[1])] = source
