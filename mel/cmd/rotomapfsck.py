"""Detect and fix problems with a rotomap filesystem."""


import logging
import pathlib

import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        '--path',
        type=pathlib.PosixPath,
        default=pathlib.Path('.'),
        help="Path to check.",
    )

    parser.add_argument(
        '--fix',
        action='store_true',
        help='Automatically fix encountered problems, where possible.',
    )

    parser.add_argument('--verbose', '-v', action='count', default=0)


def process_args(args):
    setup_logging(args.verbose)
    return fsck(args.path, args.fix)


def fsck(path, do_fix):
    if not path.is_dir():
        print(f'"{path}" is not a directory, so not a rotomap.')

    any_problems = False

    uuid_path = path / '__id__'
    has_uuid = uuid_path.exists()
    has_images = len(tuple(path.glob('*.jpg'))) > 0
    has_subdirs = any(i.is_dir() for i in path.iterdir())

    logging.debug(
        f'{path}: has uuid: {has_uuid}, has images: {has_images}, '
        f'has subdirs: {has_subdirs}'
    )

    if has_subdirs and (has_uuid or has_images):
        print(f'"{path}" looks like a rotomap, but it has subdirs.')
        any_problems = True
    elif has_images and not has_uuid:
        print(f'"{path}" looks like a rotomap, but has no uuid.')
        if do_fix:
            uuid_ = mel.rotomap.moles.make_new_uuid()
            uuid_path.write_text(uuid_)
            print(f'Wrote {uuid_} to {uuid_path}')
        else:
            any_problems = True
    elif has_subdirs:
        for p in path.iterdir():
            if p.is_dir():
                new_problems = fsck(p, do_fix)
                any_problems = new_problems or any_problems

    return any_problems


def setup_logging(verbosity):
    logtypes = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = min(len(logtypes) - 1, verbosity)
    logging.basicConfig(
        level=logtypes[level], format='%(levelname)s: %(message)s'
    )


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
