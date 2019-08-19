"""Set moles in this rotomap to be marked unchanged."""

import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        'ROTOMAP',
        type=mel.rotomap.moles.make_argparse_rotomap_directory,
        nargs='+',
        help="A list of paths to rotomaps to mark unchanged.",
    )


def process_args(args):
    changed_count = 0

    for rotomap in args.ROTOMAP:
        uuids = rotomap.calc_uuids()
        for uuid_ in uuids:
            lesion = None
            for l in rotomap.lesions:
                if l["uuid"] == uuid_:
                    lesion = l
            if lesion is None:
                lesion = {"uuid": uuid_}
                rotomap.lesions.append(lesion)
            if not lesion.get(mel.rotomap.moles.KEY_IS_UNCHANGED, False):
                changed_count += 1
            lesion[mel.rotomap.moles.KEY_IS_UNCHANGED] = True
        mel.rotomap.moles.save_rotomap_dir_lesions_file(
            rotomap.path, rotomap.lesions
        )

    print(f'Marked {changed_count} moles as unchanged.')


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
