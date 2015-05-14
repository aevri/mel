"""A big ball of mud to hold common functionality pending a re-org."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import datetime
import numpy
import os


def determine_filename_for_ident(*source_filenames):
    if not source_filenames:
        raise ValueError(
            '{} is not a valid list of filenames'.format(
                source_filenames))

    dates = [guess_date_from_path(x) for x in source_filenames]
    valid_dates = [x for x in dates if x is not None]
    if valid_dates:
        latest_date = max(valid_dates)
        return '{}.jpg'.format(latest_date.isoformat())
    else:
        return "ident.jpg"


def guess_date_from_path(path):
    """Return None if no date could be guessed, date otherwise.

    Usage examples:

        >>> guess_date_from_path('inbox/Photo 05-01-2015 23 25 40.jpg')
        datetime.date(2015, 1, 5)

        >>> guess_date_from_path('blah')

    :path: path string to be converted
    :returns: datetime.date if successful, None otherwise

    """
    # TODO: try the file date if unable to determine from name
    filename = os.path.basename(path)
    name = os.path.splitext(filename)[0]
    return guess_date_from_string(name)


def guess_date_from_string(date_str):
    """Return None if no date could be guessed, date otherwise.

    Usage examples:

        >>> guess_date_from_string('Photo 05-01-2015 23 25 40')
        datetime.date(2015, 1, 5)

        >>> guess_date_from_string('blah')

    :date_str: string to be converted
    :returns: datetime.date if successful, None otherwise

    """
    try:
        dt = datetime.datetime.strptime(date_str, 'Photo %d-%m-%Y %H %M %S')
        date = datetime.date(dt.year, dt.month, dt.day)
        return date
    except ValueError:
        return None


def make_now_datetime_string():
    return make_datetime_string(datetime.datetime.utcnow())


def make_datetime_string(datetime_):
    return datetime_.strftime("%Y%m%dT%H%M%S")


def overwrite_image(directory, filename, image):
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = os.path.join(directory, filename)
    if os.path.exists(path):
        os.remove(path)

    cv2.imwrite(path, image)


def user_mark_moles(window_name, context_image, detail_image, num_moles):

    display_image = numpy.copy(context_image)
    cv2.imshow(window_name, display_image)

    circle_radius = 50

    context_mole_positions = []
    detail_mole_positions = []
    current_mole_positions = context_mole_positions

    cv2.setMouseCallback(
        window_name,
        make_mole_capture_callback(
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

        if len(current_mole_positions) == num_moles:
            if not detail_mole_positions:
                current_mole_positions = detail_mole_positions
                display_image = numpy.copy(detail_image)
                cv2.setMouseCallback(
                    window_name,
                    make_mole_capture_callback(
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
    cv2.setMouseCallback(window_name, make_null_mouse_callback())

    return context_mole_positions, detail_mole_positions


def make_mole_capture_callback(window_name, image, radius, mole_positions):

    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(image, (x, y), radius, (255, 0, 0), -1)
            mole_positions.append((x, y, radius))
            cv2.imshow(window_name, image)

    return draw_circle


def make_null_mouse_callback():

    def null_callback(_event, _x, _y, _flags, _param):
        pass

    return null_callback


def box_moles(image, mole_positions, thickness):
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


def connect_moles(image, mole_positions):
    for mole_a, mole_b in yield_neighbors(mole_positions):
        thickness = max(mole_a[2], mole_b[2])

        # draw connection
        a = numpy.array(mole_a[:2])
        b = numpy.array(mole_b[:2])
        a_to_b = b - a
        a_to_b = a_to_b / numpy.linalg.norm(a_to_b)
        padding = a_to_b * (thickness * 2)
        a += padding
        b -= padding
        a = tuple(a.tolist())
        b = tuple(b.tolist())
        blue = (255, 0, 0)
        print(a_to_b, a, b, thickness)
        cv2.line(image, a, b, blue, thickness)


def yield_neighbors(node_list):
    is_first = True
    prev_node = None
    for node in node_list:
        if is_first:
            is_first = False
        else:
            yield (prev_node, node)
        prev_node = node


def montage_horizontal(left_image, right_image):
    if left_image.shape != right_image.shape:
        raise ValueError('image shapes must be identical')

    # calculate the bounds of the montage
    border_size = 50
    total_border_size = numpy.array([border_size * 2, border_size * 3])
    image_shape = left_image.shape
    total_image_size = numpy.array([image_shape[0], image_shape[1] * 2])
    total_montage_size = total_border_size + total_image_size

    montage_image = new_image(total_montage_size[0], total_montage_size[1])

    # write the images into the montage image
    x = border_size
    copy_image_into_image(left_image, montage_image, border_size, x)
    x += border_size
    x += image_shape[1]
    copy_image_into_image(right_image, montage_image, border_size, x)

    return montage_image


def new_image(height, width):
    return numpy.zeros((height, width, 3), numpy.uint8)


def copy_image_into_image(source, dest, y, x):
    shape = source.shape
    dest[y:(y + shape[0]), x:(x + shape[1])] = source


def shrink_to_max_dimension(image, max_dimension):
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


def indicate_mole(image, mole):
    pos = mole[:2]
    radius = mole[2]

    draw_radial_line(
        image, pos, radius * 4, radius * 6, (-1, 0), radius)
    draw_radial_line(
        image, pos, radius * 4, radius * 6, (1, 0), radius)
    draw_radial_line(
        image, pos, radius * 4, radius * 6, (0, 1), radius)
    draw_radial_line(
        image, pos, radius * 4, radius * 6, (0, -1), radius)


def draw_radial_line(
        image, origin, inner_radius, outer_radius, direction, thickness):
    origin = numpy.array(origin)
    direction = numpy.array(direction)
    line_start = origin + direction * inner_radius
    line_end = origin + direction * outer_radius

    blue = (255, 0, 0)

    line_start = tuple(line_start.tolist())
    line_end = tuple(line_end.tolist())

    cv2.line(image, line_start, line_end, blue, thickness)


def user_review_image(window_name, image):
    cv2.imshow(window_name, image)
    print("Press 'q' quit, any other key to continue.")
    key = cv2.waitKey()
    if key == ord('q'):
        raise Exception('User aborted.')


def rotated90(image, times):
    for _ in xrange(times % 4):
        image = cv2.transpose(image)
        image = cv2.flip(image, 1)
    return image

def add_context_detail_arguments(parser):

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
        '--rot90',
        type=int,
        default=None,
        help="Rotate images 90 degrees clockwise this number of times.")

    parser.add_argument(
        '--rot90-context',
        type=int,
        default=None,
        help="Rotate context image 90 degrees clockwise this number of times.")

    parser.add_argument(
        '--rot90-detail',
        type=int,
        default=None,
        help="Rotate detail image 90 degrees clockwise this number of times.")

    parser.add_argument(
        '--h-mirror',
        action="store_true",
        help="Mirror both images horizontally.")

    parser.add_argument(
        '--h-mirror-context',
        action="store_true",
        help="Mirror context image horizontally.")

    parser.add_argument(
        '--h-mirror-detail',
        action="store_true",
        help="Mirror detail image horizontally.")


def process_context_detail_args(args):
    # TODO: validate destination path up-front
    # TODO: validate mole names up-front

    context_image = cv2.imread(args.context)
    detail_image = cv2.imread(args.detail)

    if args.rot90:
        context_image = rotated90(context_image, args.rot90)
        detail_image = rotated90(detail_image, args.rot90)

    if args.rot90_context:
        context_image = rotated90(
            context_image, args.rot90_context)

    if args.rot90_detail:
        context_image = rotated90(
            detail_image, args.rot90_detail)

    if args.h_mirror:
        context_image = cv2.flip(context_image, 1)
        detail_image = cv2.flip(detail_image, 1)

    if args.h_mirror_context:
        context_image = cv2.flip(context_image, 1)

    if args.h_mirror_detail:
        detail_image = cv2.flip(detail_image, 1)

    return context_image, detail_image
