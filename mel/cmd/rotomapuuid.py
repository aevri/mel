"""List the uuids of moles that match a prefix, from a list of json files."""


import argparse
import json


def setup_parser(parser):
    parser.add_argument("PREFIX", help="Prefix to find the full id of.")
    parser.add_argument(
        "FILE",
        type=argparse.FileType(),
        nargs="+",
        help="Path to the rotomap json file.",
    )


def process_args(args):
    mole_map_list = [json.load(x) for x in args.FILE]
    uuid_set = mole_uuid_set_from_map_list(mole_map_list)
    results = []

    for mole_uuid in uuid_set:
        if mole_uuid.startswith(args.PREFIX):
            results.append(mole_uuid)

    if results:
        print("\n".join(results))
        return 0
    else:
        return 1


def mole_uuid_set_from_map_list(mole_map_list):
    uuid_set = set()
    for mole_map in mole_map_list:
        for mole in mole_map:
            uuid_set.add(mole["uuid"])
    return uuid_set


# -----------------------------------------------------------------------------
# Copyright (C) 2016-2018 Angelos Evripiotis.
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
