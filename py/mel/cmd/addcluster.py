"""A tool for adding a new cluster / constellation from photographs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy
import os


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
    # TODO: validate destination path up-front
    # TODO: validate mole names up-front

    context_image = cv2.imread(args.context)
    detail_image = cv2.imread(args.detail)

    montage_size = 1024
    mole_size = 512

    # print out the dimensions of the images
    print('{}: {}'.format(args.context, context_image.shape))
    print('{}: {}'.format(args.detail, detail_image.shape))

    # display the context image in a reasonably sized window
    window_name = 'display'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_width = 800
    window_height = 600
    cv2.resizeWindow(window_name, window_width, window_height)

    # get the user to mark the mole positions
    context_mole_positions, detail_mole_positions = _user_mark_moles(
        window_name, context_image, detail_image, args.moles)

    # Put a box around moles on context image
    _box_moles(context_image, context_mole_positions, thickness=50)

    # Connect moles on cluster detail image
    cluster_detail_image = numpy.copy(detail_image)
    _connect_moles(cluster_detail_image, detail_mole_positions)

    # Combine context image with cluster detail image to make montage
    cluster_monatage_image = _montage_horizontal(
        context_image, cluster_detail_image)
    cluster_monatage_image = _shrink_to_max_dimension(
        cluster_monatage_image, montage_size)

    # Let user review montage
    _user_review_image(window_name, cluster_monatage_image)

    # Point to moles on individual detail images
    mole_images = []
    for index, mole in enumerate(detail_mole_positions):
        indicated_image = numpy.copy(detail_image)
        _indicate_mole(indicated_image, mole)
        indicated_image = _shrink_to_max_dimension(
            indicated_image, mole_size)
        _user_review_image(window_name, indicated_image)
        mole_images.append(indicated_image)

    # No more interaction, close all windows
    cv2.destroyAllWindows()

    # Write the images
    #
    # TODO: try to determine the date from the original filename if possible
    #       and use that in ISO 8601 format.
    #
    _overwrite_image(
        args.destination,
        'ident.jpg',
        cluster_monatage_image)
    for index, mole in enumerate(args.moles):
        mole_dir = os.path.join(args.destination, mole)
        _overwrite_image(
            mole_dir,
            'ident.jpg',
            mole_images[index])

    # TODO: optionally remove the original images


def _overwrite_image(directory, filename, image):
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = os.path.join(directory, filename)
    if os.path.exists(path):
        os.remove(path)

    cv2.imwrite(path, image)


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

    # stop handling events, or there could be nasty side-effects
    cv2.setMouseCallback(window_name, _make_null_mouse_callback())

    return context_mole_positions, detail_mole_positions


def _make_mole_capture_callback(window_name, image, radius, mole_positions):

    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(image, (x, y), radius, (255, 0, 0), -1)
            mole_positions.append((x, y, radius))
            cv2.imshow(window_name, image)

    return draw_circle


def _make_null_mouse_callback():

    def null_callback(_event, _x, _y, _flags, _param):
        pass

    return null_callback


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


def _shrink_to_max_dimension(image, max_dimension):
    """May or may not return the original image."""

    shape = image.shape
    height = shape[0]
    width = shape[1]

    scaling_factor = max_dimension / max(width, height)
    if scaling_factor >= 1:
        return image
    else:
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        return cv2.resize(image, (new_width, new_height))


def _indicate_mole(image, mole):
    pos = mole[:2]
    radius = mole[2]

    _draw_radial_line(
        image, pos, radius * 4, radius * 6, (-1, 0), radius)
    _draw_radial_line(
        image, pos, radius * 4, radius * 6, (1, 0), radius)
    _draw_radial_line(
        image, pos, radius * 4, radius * 6, (0, 1), radius)
    _draw_radial_line(
        image, pos, radius * 4, radius * 6, (0, -1), radius)


def _draw_radial_line(
        image, origin, inner_radius, outer_radius, direction, thickness):
    origin = numpy.array(origin)
    direction = numpy.array(direction)
    line_start = origin + direction * inner_radius
    line_end = origin + direction * outer_radius

    blue = (255, 0, 0)

    line_start = tuple(line_start.tolist())
    line_end = tuple(line_end.tolist())

    cv2.line(image, line_start, line_end, blue, thickness)


def _user_review_image(window_name, image):
    cv2.imshow(window_name, image)
    print("Press 'a' abort, any other key to continue.")
    key = cv2.waitKey()
    if key == ord('q'):
        raise Exception('User aborted.')
