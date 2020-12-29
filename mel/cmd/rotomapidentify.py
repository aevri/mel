"""Guess which mole is which in a rotomap image."""

import os


def setup_parser(parser):
    parser.add_argument(
        "TARGET",
        nargs="+",
        help="Paths to images to identify.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print information about the processing.",
    )


def process_args(args):
    # This is an expensive import, so only do it when necessary.
    import mel.rotomap.identifynn

    identifier = mel.rotomap.identifynn.make_identifier()
    for target in args.TARGET:
        if args.verbose:
            print("Processing", target, "..")

        # part = mel.lib.fs.get_rotomap_part_from_path(melroot, target)
        frame = mel.rotomap.moles.RotomapFrame(os.path.abspath(target))

        new_moles = identifier.get_new_moles(frame)

        mel.rotomap.moles.save_image_moles(new_moles, str(frame.path))


# -----------------------------------------------------------------------------
# Copyright (C) 2019 Angelos Evripiotis.
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
