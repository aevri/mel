"""Determine the space described by the mask of a rotomap image."""

import sys

import cv2

import mel.lib.ellipsespace
import mel.lib.moleimaging

import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        'TARGET',
        nargs='+',
        help="Paths to images to calculate the space of.")
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help="Print information about the processing.")


def process_args(args):
    for path in args.TARGET:
        if args.verbose:
            print('Target:', path)

        mask = mel.rotomap.mask.load_or_none(path)
        if mask is None:
            print(f'{path} has no mask.', file=sys.stderr)
            return 1
        contour = mel.lib.moleimaging.biggest_contour(mask)
        ellipse = cv2.minAreaRect(contour)
        metadata = {
            'ellipse': ellipse
        }
        mel.rotomap.moles.save_image_metadata(
            metadata, path)


# -----------------------------------------------------------------------------
# Copyright (C) 2018 Angelos Evripiotis.
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
