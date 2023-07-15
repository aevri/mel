"""Automatically look at moles, and record features of them.

Note that these features are not human-interpretable, and are intended to help
machines identify which mole is which.

"""

import mel.lib.common
import mel.lib.fs
import mel.lib.image
import mel.rotomap.mask


def setup_parser(parser):
    parser.add_argument(
        "TARGET", nargs="+", help="Paths to images to autolook."
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print information about the processing.",
    )


def process_args(args):
    for path in args.TARGET:
        if args.verbose:
            print("Target:", path)
        image = mel.lib.image.load_image(path)
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Copyright (C) 2023 Angelos Evripiotis.
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
