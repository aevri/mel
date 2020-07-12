"""Automatically mark moles on rotomap images."""

import sys

import cv2

import torch
import torchvision

import mel.lib.image

import mel.rotomap.detectmoles
import mel.rotomap.mask
import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        "IMAGES", nargs="+", help="A list of paths to images to automark."
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print information about the processing.",
    )


def process_args(args):
    try:
        melroot = mel.lib.fs.find_melroot()
    except mel.lib.fs.NoMelrootError:
        print("Not in a mel repo, could not find melroot", file=sys.stderr)
        return 1

    cpu_device = torch.device("cpu")

    model_dir = melroot / mel.lib.fs.DEFAULT_CLASSIFIER_PATH
    model_path = model_dir / "detectmoles.pt"

    model = mel.rotomap.detectmolesnn.DenseUnet(
        channels_in=3, channels_per_layer=16, num_classes=1
    )
    model.load_state_dict(torch.load(model_path, map_location=cpu_device))
    model.to(cpu_device)
    model.eval()

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.1940, 0.9525, 0.1776],
                std=[0.3537, 0.0972, 0.3244],
                inplace=True,
            ),
        ]
    )

    for path in args.IMAGES:
        if args.verbose:
            print(path)

        frame = mel.rotomap.moles.RotomapFrame(path)
        masked_image = _load_masked_image(frame)
        with torch.no_grad():
            out = model(transforms(masked_image).unsqueeze(0))

        minimums = _show_minimums(out[0][0], maximum=10)
        connected = _connected_pixels(minimums)
        mole_locations = [
            torch.mean(torch.tensor(x, dtype=float), axis=0) for x in connected
        ]

        moles = []
        for x, y in mole_locations:
            mel.rotomap.moles.add_mole(moles, int(x), int(y))

        mel.rotomap.moles.save_image_moles(moles, path)


def _connected_pixels(binary_image):
    # Note that PyTorch arrays don't work in maps.
    clusters = {(int(i[1]), int(i[0])): [] for i in binary_image.nonzero()}
    for pixel, joined_pixels in clusters.items():
        x, y = pixel
        if not joined_pixels:
            joined_pixels.append(pixel)
        for other_pixel in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            other_joined_pixels = clusters.get(other_pixel, None)
            if other_joined_pixels is None:
                continue
            if joined_pixels is other_joined_pixels:
                continue
            if not other_joined_pixels:
                other_joined_pixels = [other_pixel]
            joined_pixels.extend(other_joined_pixels)
            for pixel_to_update in other_joined_pixels:
                clusters[pixel_to_update] = joined_pixels

    return {tuple(c) for c in clusters.values()}


def _load_masked_image(frame):
    image = frame.load_image()
    mask = frame.load_mask()
    image = mel.rotomap.detectmolesnn.green_mask_image(image, mask)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _show_minimums(image, maximum):
    new_image = torch.empty_like(image)
    new_image[image <= maximum] = 1.0
    new_image[image > maximum] = 0.0
    return new_image


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
