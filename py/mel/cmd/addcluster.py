"""A tool for adding a new cluster / constellation from photographs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def setup_parser(parser):

    parser.add_argument(
        'context-filename',
        type=str,
        default=None,
        help="Path to the context image to add.")

    parser.add_argument(
        'detail-filename',
        type=str,
        default=None,
        help="Path to the detail image to add.")


def process_args(args):
    raise NotImplementedError()
