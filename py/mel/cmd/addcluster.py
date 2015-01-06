"""A tool for adding a new cluster / constellation from photographs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2


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
    raise NotImplementedError()
