"""Write images of individual moles and non-moles, for training AIs."""

import numpy
import os
import pathlib
import sys

import cv2

import mel.lib.common
import mel.lib.image
import mel.rotomap.detectmoles
import mel.rotomap.mask
import mel.rotomap.moles
import mel.cmd.error


SIZE = 224

# TODO: Check for masking errors, where the mole is obscured by the mask.
# TODO: Make the mask green.


def setup_parser(parser):

    parser.add_argument(
        'OUT_PATH',
        type=pathlib.Path,
        help="Path to write non-mole images to.")

    parser.add_argument(
        'FRAMES',
        type=mel.rotomap.moles.make_argparse_image_moles,
        nargs='+',
        help="Path to rotomap or image to copy from.")

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help="Print information about the processing.")

    parser.add_argument(
        '--size-half',
        '-s',
        default=SIZE,
        type=int,
        help="The 'radius' around the items, pixels. i.e. x2 to calc width.")


def process_args(args):
    marks_out = pathlib.Path(args.OUT_PATH) / 'marks'
    moles_out = pathlib.Path(args.OUT_PATH) / 'moles'
    marks_out.mkdir(exist_ok=True)
    moles_out.mkdir(exist_ok=True)
    if args.verbose:
        print('Write non-moles to:', marks_out)
        print('Write moles to:', moles_out)

    written_moles = 0
    written_marks = 0
    for image_mole_iter in args.FRAMES:
        for image_path, moles in image_mole_iter:
            try:
                original_image = mel.lib.image.load_image(image_path)

                mask = mel.rotomap.mask.load(image_path)
                green = numpy.zeros(original_image.shape, numpy.uint8)
                green[:, :, 1] = 255
                image = cv2.bitwise_and(
                    original_image, original_image, mask=mask)
                not_mask = cv2.bitwise_not(mask)
                green = cv2.bitwise_and(green, green, mask=not_mask)
                image = cv2.bitwise_or(image, green)

                moles_and_marks = mel.rotomap.moles.load_image_moles(
                    image_path)
                moles = filter_moles(moles_and_marks)
                marks = filter_marks(moles_and_marks)
                written_marks += write_images(
                    marks_out, image_path, image, mask, marks, args.size_half)
                written_moles += write_images(
                    moles_out, image_path, image, mask, moles, args.size_half)
            except Exception:
                raise Exception('Error while processing {}'.format(image_path))

    if args.verbose:
        print('Written moles:', written_moles)
        print('Written marks:', written_marks)


def filter_moles(moles_and_marks):
    moles = []

    for item in moles_and_marks:
        kind = item.get('kind', None)
        looks_like = item.get('looks_like', None)
        if looks_like == 'unsure':
            continue
        elif kind == 'mole' and looks_like == 'non-mole':
            continue
        elif kind == 'non-mole' and looks_like == 'mole':
            pass
        elif not item[mel.rotomap.moles.KEY_IS_CONFIRMED]:
            continue
        moles.append(item)

    return moles


def filter_marks(moles_and_marks):
    marks = []

    for item in moles_and_marks:
        kind = item.get('kind', None)
        looks_like = item.get('looks_like', None)
        if looks_like == 'unsure':
            continue
        elif kind == 'non-mole' and looks_like == 'mole':
            continue
        elif kind == 'mole':
            # Even if this looks_like a non-mole, exclude it from the dataset.
            # Even humans may find those cases ambiguous, perhaps it's better
            # to stick to unambiguous cases for training and evaluating.
            # There seem to be plenty of real marks to consider instead.
            continue
        elif item[mel.rotomap.moles.KEY_IS_CONFIRMED]:
            continue
        marks.append(item)

    return marks


def write_images(dir_out, original_path, image, mask, items, size):
    count = 0
    for i in items:
        if write_item_image(
                dir_out, original_path, image, mask, i, size):
            count += 1
    return count


def write_item_image(dir_out, original_path, image, mask, item, size):
    name = str(original_path).replace('/', '_').replace('.', '_')
    name += '_' + item['uuid']
    # name += f'_{item["x"]}_{item["y"]}'
    name += '.jpg'

    x = item['x']
    y = item['y']

    if not mask[y, x]:
        print(f"Occuluded by mask: {name}", file=sys.stderr)
        return None

    item_image = image[y - size:y + size, x - size:x + size]

    if not all(item_image.shape):
        print(f"Invalid size {item_image.shape} for {name}", file=sys.stderr)
        # raise Exception('Invalid item shape', item_image.shape)
        return None

    target_path = os.path.join(dir_out, name)
    mel.lib.common.write_image(target_path, item_image)

    return target_path


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
