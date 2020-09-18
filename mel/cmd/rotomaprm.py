"""Remove moles matching specified UUIDs from selected JSONs."""

import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        "--uuids",
        "-u",
        metavar="UUID",
        nargs="+",
        required=True,
        help="A list of UUIDs to remove.",
    )
    parser.add_argument(
        "--files",
        "-f",
        metavar="JSON_FILE",
        nargs="+",
        required=True,
        help="A list of paths to image json files.",
    )


def process_args(args):
    changed_count = 0

    for path in args.files:
        moles = mel.rotomap.moles.load_json(path)
        new_moles = [m for m in moles if m["uuid"] not in args.uuids]
        mel.rotomap.moles.save_json(path, new_moles)
        changed_count += len(moles) - len(new_moles)

    print(f"Removed {changed_count} moles.")


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
