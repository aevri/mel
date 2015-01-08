"""A tool for adding a new cluster / constellation from photographs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy
import os

import mel.lib.common


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


def process_args(args):
    # TODO: validate destination path up-front
    # TODO: validate mole names up-front

    context_image = cv2.imread(args.context)
    detail_image = cv2.imread(args.detail)

    if args.rot90:
        context_image = mel.lib.common.rotated90(context_image, args.rot90)
        detail_image = mel.lib.common.rotated90(detail_image, args.rot90)

    if args.rot90_context:
        context_image = mel.lib.common.rotated90(
            context_image, args.rot90_context)

    if args.rot90_detail:
        context_image = mel.lib.common.rotated90(
            detail_image, args.rot90_detail)

    if args.h_mirror:
        context_image = cv2.flip(context_image, 1)
        detail_image = cv2.flip(detail_image, 1)

    if args.h_mirror_context:
        context_image = cv2.flip(context_image, 1)

    if args.h_mirror_detail:
        detail_image = cv2.flip(detail_image, 1)

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
    context_mole_pos, detail_mole_pos = mel.lib.common.user_mark_moles(
        window_name, context_image, detail_image, len(args.moles))

    # Put a box around moles on context image
    mel.lib.common.box_moles(
        context_image,
        context_mole_pos,
        thickness=50)

    # Connect moles on cluster detail image
    cluster_detail_image = numpy.copy(detail_image)
    mel.lib.common.connect_moles(cluster_detail_image, detail_mole_pos)

    # Combine context image with cluster detail image to make montage
    cluster_monatage_image = mel.lib.common.montage_horizontal(
        context_image, cluster_detail_image)
    cluster_monatage_image = mel.lib.common.shrink_to_max_dimension(
        cluster_monatage_image, montage_size)

    # Let user review montage
    mel.lib.common.user_review_image(window_name, cluster_monatage_image)

    # Point to moles on individual detail images
    mole_images = []
    for index, mole in enumerate(detail_mole_pos):
        indicated_image = numpy.copy(detail_image)
        mel.lib.common.indicate_mole(indicated_image, mole)
        indicated_image = mel.lib.common.shrink_to_max_dimension(
            indicated_image, mole_size)
        mel.lib.common.user_review_image(window_name, indicated_image)
        mole_images.append(indicated_image)

    # No more interaction, close all windows
    cv2.destroyAllWindows()

    # Write the images
    #
    # TODO: try to determine the date from the original filename if possible
    #       and use that in ISO 8601 format.
    #
    mel.lib.common.overwrite_image(
        args.destination,
        mel.lib.common.determine_filename_for_ident(args.context, args.detail),
        cluster_monatage_image)
    for index, mole in enumerate(args.moles):
        mole_dir = os.path.join(args.destination, mole)
        mel.lib.common.overwrite_image(
            mole_dir,
            mel.lib.common.determine_filename_for_ident(args.detail),
            mole_images[index])

    # TODO: optionally remove the original images
