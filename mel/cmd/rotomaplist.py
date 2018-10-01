"""List the uuids of moles in supplied files."""


import argparse
import json


def setup_parser(parser):
    parser.add_argument(
        'FILE',
        type=argparse.FileType(),
        nargs='+',
        help="Path to the rotomap json file.",
    )


def process_args(args):
    path_data_list = [(x.name, json.load(x)) for x in args.FILE]
    for path, data in path_data_list:
        for mole in data:
            print(mole["uuid"], path)


# -----------------------------------------------------------------------------
# Copyright (C) 2016-2017 Angelos Evripiotis.
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
