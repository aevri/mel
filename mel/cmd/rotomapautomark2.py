"""Automatically mark moles on rotomap images."""

import copy
import json
import pathlib
import sys

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

    model_dir = melroot / mel.lib.fs.DEFAULT_CLASSIFIER_PATH
    model_path = model_dir / "detectmoles.pth"
    metadata_path = model_dir / "detectmoles.json"

    tile_size = 32

    with open(metadata_path) as f:
        init_dict = json.load(f)

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )
    # resnet = torchvision.models.resnet18(pretrained=True)
    resnet = torchvision.models.resnet50(pretrained=True)
    resnet.eval()
    model = mel.rotomap.detectmolesnn.NeighboursLinearSigmoidModel(**init_dict)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for path in args.IMAGES:
        if args.verbose:
            print(path)
        frame = mel.rotomap.moles.RotomapFrame(path)
        get_data = mel.rotomap.detectmolesnn.get_tile_locations_activations
        with torch.no_grad():
            locations_activations = get_data(
                frame, transforms, resnet, reduce_nonmoles=False
            )
        if locations_activations is None:
            print(path, ":", "no locations to test.")
            continue

        locations, activations = locations_activations
        print(len(locations), "tiles")

        part = mel.rotomap.detectmolesnn.image_path_to_part(pathlib.Path(path))
        match = mel.rotomap.detectmolesnn.match_with_neighbours
        locs_acts_neighbour_acts = match(locations, activations, tile_size)
        dataset = [
            (act, part, neighbour_acts)
            for loc, act, neighbour_acts in locs_acts_neighbour_acts
        ]
        print("Neighbour blocks:", len(dataset))

        results = []
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
            for batch in dataloader:
                results.append(model(*batch))
        results = torch.cat(results)
        print(len(results), "results")

        moles = copy.deepcopy(frame.moles)
        match_locs = [
            loc for loc, act, neighbour_acts in locs_acts_neighbour_acts
        ]
        num_moles_before = len(moles)
        for (x, y), (mole_likelihood, xoff, yoff) in zip(match_locs, results):
            if mole_likelihood > 0.85:
                new_x = int(x + (1 + xoff) * (tile_size * 0.5))
                new_y = int(y + (1 + yoff) * (tile_size * 0.5))
                mel.rotomap.moles.add_mole(moles, new_x, new_y)
        mel.rotomap.moles.save_image_moles(moles, path)
        print("Added", len(moles) - num_moles_before, "moles.")


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
