"""A tool for adding a single mole from photographs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2

import mel.lib.common
import mel.lib.image


def setup_parser(parser):

    mel.lib.common.add_context_detail_arguments(parser)

    parser.add_argument(
        'destination',
        type=str,
        default=None,
        help="New path to create and store the mole to.")


def process_args(args):
    # TODO: validate destination path up-front
    # TODO: validate mole names up-front

    context_image, detail_image = mel.lib.common.process_context_detail_args(
        args)

    # TODO: extract this choice to a common place
    montage_size = 1024

    # display the context image in a reasonably sized window
    # TODO: extract this choice to a common place
    window_name = 'display'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_width = 800
    window_height = 600
    cv2.resizeWindow(window_name, window_width, window_height)

    # get the user to mark the mole positions
    context_mole_pos, detail_mole_pos = mel.lib.common.user_mark_moles(
        window_name, context_image, detail_image, 1)

    # Put a box around mole on context image
    # TODO: extract the thickness choice to a common place
    mel.lib.common.box_moles(
        context_image,
        context_mole_pos,
        thickness=50)

    # Point to mole on detail image
    mel.lib.common.indicate_mole(detail_image, detail_mole_pos[0])

    # Combine context image with detail image to make montage
    monatage_image = mel.lib.image.montage_horizontal(
        context_image, detail_image)
    monatage_image = mel.lib.common.shrink_to_max_dimension(
        monatage_image, montage_size)

    # Let user review montage
    mel.lib.common.user_review_image(window_name, monatage_image)

    # No more interaction, close all windows
    cv2.destroyAllWindows()

    # Write the images
    mel.lib.common.overwrite_image(
        args.destination,
        mel.lib.common.determine_filename_for_ident(args.context, args.detail),
        monatage_image)

    # TODO: optionally remove the original images
