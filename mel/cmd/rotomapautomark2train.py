"""Train to automatically mark moles on rotomap images."""

import contextlib

import cv2
import numpy
import torch


def setup_parser(parser):
    parser.add_argument(
        'IMAGES',
        nargs='+',
        help="A list of paths to images to automark.")
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help="Print information about the processing.",
    )


def process_args(args):
    for path in args.IMAGES:
        if args.verbose:
            print(path)
        image = mel.lib.image.load_image(path)
        mask = mel.rotomap.mask.load(path)
        guessed_moles = mel.rotomap.detectmoles.moles(image, mask)
        loaded_moles = mel.rotomap.moles.load_image_moles(path)
        moles = _merge_in_radiuses(
            loaded_moles,
            radii_sources=guessed_moles,
            error_distance=args.error_distance,
            only_merge=args.only_merge,
        )
        mel.rotomap.moles.save_image_moles(moles, path)


def process_args(args):
    # TODO: Make data from frames.
    # TODO: Make model from pre-trained Resnet.
    # TODO: TODO: Drop regions that are all mask.
    # TODO: Record outputs from Resnet backbone.
    # TODO: Cache outputs from Resnet backbone.
    # TODO: Train and validate model on pre-recorded outputs.
    pass


# def train_frames(training_frame_list, validation_frame_list, model, opt):
#     training_data = make_labelled_dataset(training_frame_list)
#     validation_data = make_labelled_dataset(validation_frame_list)
#     raise NotImplementedError()


def make_labelled_data(frame):
    raise NotImplementedError()


def make_unlabelled_data(frame):
    raise NotImplementedError()


def train(training_data, validation_data):
    raise NotImplementedError()


def detect_moles(image, mask):
    raise NotImplementedError()


def green_mask_image(image, mask):
    green = numpy.zeros(image.shape, numpy.uint8)
    green[:, :, 1] = 255
    image = cv2.bitwise_and(image, image, mask=mask)
    not_mask = cv2.bitwise_not(mask)
    green = cv2.bitwise_and(green, green, mask=not_mask)
    image = cv2.bitwise_or(image, green)
    return image


@contextlib.contextmanager
def record_input_context(module_to_record):
    activations = []

    def record_response(module, input_):
        nonlocal activations
        activations.append(input_)

    hook = module_to_record.register_forward_pre_hook(record_response)
    with contextlib.ExitStack() as stack:
        stack.callback(hook.remove)
        yield activations


def frame_to_dataset(frame, transforms):
    image = frame.load_image()
    tiles = []
    locations = []
    expected_outputs = []
    tile_size = 32
    green = [0, 255, 0]
    for y1 in range(0, image.shape[0], tile_size):
        for x1 in range(0, image.shape[1], tile_size):
            t = image[y1:y1 + tile_size, x1:x1 + tile_size]
            if (t[:, :] == green).all():
                continue
            tiles.append(transforms(t))
            locations.append(torch.tensor((y1, x1)))
            has_mole = torch.tensor([0.0])
            top_left = (x1, y1)
            bottom_right = (x1 + tile_size, y1 + tile_size)
            for point in frame.moledata.uuid_points.values():
                if all(point >= top_left) and all(point <= bottom_right):
                    has_mole = torch.tensor([1.0])
            expected_outputs.append(has_mole)
    return list(zip(tiles, locations, expected_outputs))


# def catted(list_tensor, element_tensor):
#     if list_tensor is None:
#         return element_tensor.unsqueeze(0)
#     return torch.cat((list_tensor, element_tensor.unsqueeze(0)))


# def record_inputs(model, to_record, dataset):
#     model.eval()
#     with record_input_context(to_record) as activations:
#         with torch.no_grad():
#             for data in tqdm.tqdm(dataset):
#                 model(data[1].unsqueeze(0))
#     return activations


# -----------------------------------------------------------------------------
# Copyright (C) 2020 Angelos Evripiotis.
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
