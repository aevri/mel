"""Set moles to be manually confirmed to have the correct UUID."""

import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        'JSON_FILE',
        nargs='+',
        help="A list of paths to image json files.")


def process_args(args):
    changed_count = 0

    for path in args.JSON_FILE:
        moles = mel.rotomap.moles.load_json(path)
        for m in moles:
            if not m.get(mel.rotomap.moles.KEY_IS_CONFIRMED, False):
                changed_count += 1
                m[mel.rotomap.moles.KEY_IS_CONFIRMED] = True
        mel.rotomap.moles.save_json(path, moles)

    print(f'Confirmed {changed_count} moles.')


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
