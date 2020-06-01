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
    resnet = torchvision.models.resnet18(pretrained=True)
    # resnet = torchvision.models.resnet50(pretrained=True)
    resnet.eval()
    # model = mel.rotomap.detectmolesnn.NeighboursLinearSigmoidModel2(**init_dict)
    # model = mel.rotomap.detectmolesnn.LinearSigmoidModel2(**init_dict)
    model = mel.rotomap.detectmolesnn.DenseUnetModel(**init_dict)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    for path in args.IMAGES:
        if args.verbose:
            print(path)
        frame = mel.rotomap.moles.RotomapFrame(path)
        # get_data = mel.rotomap.detectmolesnn.get_tile_locations_activations
        # with torch.no_grad():
        #     locations_activations = get_data(
        #         frame, transforms, resnet, reduce_nonmoles=False
        #     )
        # if locations_activations is None:
        #     print(path, ":", "no locations to test.")
        #     continue

        # locations, activations = locations_activations
        # print(len(locations), "tiles")

        # part = mel.rotomap.detectmolesnn.image_path_to_part(pathlib.Path(path))
        # match = mel.rotomap.detectmolesnn.match_with_neighbours
        # locs_acts_neighbour_acts = match(locations, activations, tile_size)
        # dataset = [
        #     # (act, part, neighbour_acts)
        #     (act, part)
        #     # for loc, act, neighbour_acts in locs_acts_neighbour_acts
        #     for act in activations
        # ]
        # print("Neighbour blocks:", len(dataset))

        # results = []
        # with torch.no_grad():
        #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
        #     for batch in dataloader:
        #         results.append(model(*batch))
        # results = torch.cat(results)
        # print(len(results), "results")

        results = []
        dataloader, dataset = load_dataset2([pathlib.Path(path)], 64)
        with torch.no_grad():
            for batch in dataloader:
                (
                    batch_ids,
                    batch_activations,
                    batch_parts,
                    batch_expected_outputs,
                    # batch_neighbours,
                ) = batch
                results.append(model(batch_activations))
        results = torch.cat(results)
        print(len(results), "results")

        moles = copy.deepcopy(frame.moles)
        # match_locs = [
        #     loc for loc, act, neighbour_acts in locs_acts_neighbour_acts
        # ]
        num_moles_before = len(moles)

        min_likelihood = 0.85
        likelihood_x_y = _collate_results(
            # match_locs, results, tile_size, min_likelihood
            dataset.location,
            results,
            tile_size,
            min_likelihood,
        )
        if likelihood_x_y is None:
            print("No moles found.")
            continue
        likelihood_x_y = _merge_close_results(likelihood_x_y, tile_size)

        for likelihood, x, y in likelihood_x_y:
            if likelihood > 0.85:
                new_x = int(x)
                new_y = int(y)
                mel.rotomap.moles.add_mole(moles, new_x, new_y)

        mel.rotomap.moles.save_image_moles(moles, path)
        print("Added", len(moles) - num_moles_before, "moles.")


def load_dataset2(images, batch_size):
    print(f"Will load from {len(images)} images.")
    dataset = mel.rotomap.detectmolesnn.TileDataset2(images, 32)
    print(f"Loaded {len(dataset)} tiles.")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
    )
    return dataloader, dataset


def _collate_results(match_locs, results, tile_size, min_likelihood):
    if len(match_locs) != len(results):
        raise ValueError(
            f"Length of match_locs ({len(match_locs)}) != "
            f"length of results ({len(results)})."
        )
    likelihood_x_y_list = []
    for (x, y), (mole_likelihood, xoff, yoff) in zip(match_locs, results):
        if mole_likelihood >= min_likelihood:
            new_x = x + (1 + xoff) * (tile_size * 0.5)
            new_y = y + (1 + yoff) * (tile_size * 0.5)
            likelihood_x_y_list.append(
                torch.tensor([mole_likelihood, new_x, new_y])
            )
    if likelihood_x_y_list:
        result = torch.stack(likelihood_x_y_list)
        assert len(result.shape) == 2
        assert result.shape[1] == 3
    else:
        result = None
    return result


def _merge_close_results(likelihood_x_y, min_dist):

    min_dist_sq = min_dist ** 2

    x_y = likelihood_x_y[:, 1:]

    x_y_a = x_y.unsqueeze(0)
    x_y_b = x_y.unsqueeze(1)
    assert x_y_a.shape == (1, len(x_y), 2)
    assert x_y_b.shape == (len(x_y), 1, 2)

    x_y_dist = x_y_a - x_y_b
    x_y_dist_sq = x_y_dist ** 2
    dist_sq = x_y_dist_sq[:, :, 0] + x_y_dist_sq[:, :, 1]
    assert x_y_dist.shape == (len(x_y), len(x_y), 2)
    assert x_y_dist_sq.shape == (len(x_y), len(x_y), 2)
    assert dist_sq.shape == (len(x_y), len(x_y))

    index_to_cluster = {}

    for i, mole in enumerate(dist_sq):
        curr_cluster = index_to_cluster.get(i, set([i]))
        index_to_cluster[i] = curr_cluster
        for j in range(i, len(dist_sq)):
            j_dist_sq = mole[j]
            if j_dist_sq < min_dist_sq:
                other_cluster = index_to_cluster.get(j, set([j]))
                curr_cluster |= other_cluster
                for k in other_cluster:
                    index_to_cluster[k] = curr_cluster

    clusters = {tuple(s) for s in index_to_cluster.values()}

    # TODO: get unique clusters, merge items based on average position,
    # weighted by likelihood.

    new_likelihood_x_y = []
    for indices in clusters:
        i = indices[0]
        new_likelihood_x_y.append(likelihood_x_y[i])

    return torch.stack(new_likelihood_x_y)


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
