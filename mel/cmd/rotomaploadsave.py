"""Load and save the specified moles, to apply any format changes.

Note that if a mole file did not exist for the image before, it will be
created.
"""

import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        'IMAGES',
        nargs='+',
        help="A list of paths to images to automark.")
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help="Print information about the processing.",
    )


def process_args(args):
    for path in args.IMAGES:
        if args.verbose:
            print(path)
        loaded_moles = mel.rotomap.moles.load_image_moles(path)
        mel.rotomap.moles.save_image_moles(loaded_moles, path)


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
