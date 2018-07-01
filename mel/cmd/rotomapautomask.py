"""Automatically mask rotomap images."""

import mel.lib.common
import mel.lib.fs
import mel.lib.image
import mel.lib.ui
import mel.rotomap.mask


def setup_parser(parser):
    parser.add_argument(
        'TARGET',
        nargs='+',
        help="Paths to images to automask.")
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help="Print information about the processing.")


def process_args(args):
    for path in args.TARGET:
        if args.verbose:
            print('Target:', path)
        image = mel.lib.image.load_image(path)
        mask = mel.rotomap.mask.guess_mask_otsu(image)
        mel.lib.common.write_image(path + '.mask.png', mask)
# -----------------------------------------------------------------------------
# Copyright (C) 2017 Angelos Evripiotis.
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
