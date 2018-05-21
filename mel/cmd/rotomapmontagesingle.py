"""Create a montage image for a single mole from a rotomap."""

import cv2

import mel.lib.common
import mel.lib.image
import mel.rotomap.moles
import mel.cmd.error


def setup_parser(parser):

    parser.add_argument(
        'ROTOMAP',
        type=mel.rotomap.moles.make_argparse_rotomap_directory,
        help="Path to the rotomap to copy from.")

    parser.add_argument(
        'UUID',
        type=str,
        help="Unique id of the mole to copy.")

    parser.add_argument(
        'OUTPUT',
        type=str,
        help="Name of the image to write.")

    parser.add_argument(
        '--rot90',
        type=int,
        default=0,
        help="Rotate images 90 degrees clockwise this number of times.")


def process_args(args):
    mel.lib.common.write_image(
        args.OUTPUT,
        make_montage_image(args.ROTOMAP, args.UUID, args.rot90))


def make_montage_image(rotomap, uuid_, rot90=0):

    path_moles_list = []

    radius = 10
    montage_height = 1024

    for imagepath, moles in rotomap.yield_mole_lists():
        for m in moles:
            if m['uuid'] == uuid_:
                path_moles_list.append((imagepath, moles))

    if not path_moles_list:
        raise mel.cmd.error.UsageError(
            'UUID "{}" not found in rotomap "{}".'.format(
                uuid_, rotomap.path))

    # Pick 'best' image for this particular mole, assuming that the middle
    # image is where the mole is most prominent. This assumption is based on
    # the idea that the images represent a rotation around the subject, the
    # 'middle' image should be where the mole is most centered.
    #
    # TODO: Cater for the case where the image set represents a complete
    # rotation around the subject, and therefore the ends meet. e.g. if there
    # are 10 images, and the target mole appears in images 0, 1, 7, 8, 9 then
    # this rule will pick image 7 instead of 9.
    #
    path, mole_list = path_moles_list[len(path_moles_list) // 2]

    mole_dict = {m['uuid']: m for m in mole_list}
    mole = mole_dict[uuid_]

    context_image = mel.rotomap.moles.load_image(path)
    x = mole['x']
    y = mole['y']

    # Draw a faded mark to indicate the mole. Make it faded in case we
    # accidentally cover moles or other distinguishing marks.
    unmarked_context_image = context_image.copy()
    mel.lib.common.indicate_mole(context_image, (x, y, radius))
    context_image = cv2.addWeighted(
        unmarked_context_image, 0.75, context_image, 0.25, 0.0)

    context_image = mel.lib.common.rotated90(context_image, rot90)
    for _ in range(rot90 % 4):
        x, y = -y, x
    if x < 0:
        x = x + context_image.shape[1]
    if y < 0:
        y = y + context_image.shape[0]

    detail_image = make_detail_image(context_image, x, y, montage_height)

    context_scale = montage_height / context_image.shape[0]
    context_scaled_width = int(context_image.shape[1] * context_scale)
    context_image = cv2.resize(
        context_image,
        (context_scaled_width, montage_height))

    return mel.lib.image.montage_horizontal_inner_border(
        25, context_image, detail_image)


def make_detail_image(context_image, x, y, size):
    half_size = size // 2
    left = max(x - half_size, 0)
    top = max(y - half_size, 0)
    right = left + half_size * 2
    bottom = top + half_size * 2
    return context_image[top:bottom, left:right]
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
