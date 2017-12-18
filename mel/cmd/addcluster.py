"""A tool for adding a new cluster / constellation from photographs."""

import os

import cv2
import numpy

import mel.lib.common
import mel.lib.image


def setup_parser(parser):

    mel.lib.common.add_context_detail_arguments(parser)

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

    context_image, detail_image = mel.lib.common.process_context_detail_args(
        args)

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
    cluster_monatage_image = mel.lib.image.montage_horizontal(
        50, context_image, cluster_detail_image)
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
# -----------------------------------------------------------------------------
# Copyright (C) 2015-2017 Angelos Evripiotis.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------ END-OF-FILE ----------------------------------
