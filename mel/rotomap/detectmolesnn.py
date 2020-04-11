"""Detect moles in an image, using deep neural nets."""

import contextlib
import pathlib

import cv2
import numpy
import torch
import tqdm

import mel.lib.math
import mel.rotomap.moles


def cat_allow_none(left, right):
    if left is None:
        return right
    return torch.cat((left, right))


def train(model, train_dataloader, valid_dataloader, loss_func, num_epochs=10):
    threshold = 0.8
    opt = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        steps_per_epoch=len(train_dataloader),
        max_lr=0.01,
        epochs=num_epochs,
    )
    # loss_func = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        print("epoch:", epoch + 1)

        model.train()
        for batch in train_dataloader:
            (
                batch_ids,
                batch_activations,
                batch_parts,
                batch_expected_outputs,
                batch_neighbours,
            ) = batch
            opt.zero_grad()
            out = model(batch_activations, batch_parts, batch_neighbours)
            # out = model(batch_activations, batch_parts)
            # print(batch_expected_outputs.unsqueeze(1).shape)
            # print(out.shape)
            loss = loss_func(out, batch_expected_outputs)
            loss.backward()
            opt.step()
            scheduler.step()
        print("train:", loss)

        # model.eval()
        # num_correct = 0
        # num_total = 0
        # for batch in train_dataloader:
        #     (
        #         batch_ids,
        #         batch_activations,
        #         batch_parts,
        #         batch_expected_outputs,
        #     ) = batch
        #     with torch.no_grad():
        #         out = model(batch_activations, batch_parts)
        #         choices = out > 0.8
        #         expected_choices = batch_expected_outputs.unsqueeze(1) > 0.8
        #         loss = loss_func(out, batch_expected_outputs.unsqueeze(1))
        #         num_correct += (choices == expected_choices).float().sum()
        #         num_total += len(batch_ids)
        # print("train eval:", loss)
        # print(
        #     "train acc:",
        #     float(100 * num_correct / num_total),
        #     int(num_correct),
        #     int(num_total),
        # )

        model.eval()
        num_correct = 0
        num_total = 0
        num_moles_correct = 0
        num_predicted_moles_total = 0
        num_moles_total = 0
        for batch in valid_dataloader:
            (
                batch_ids,
                batch_activations,
                batch_parts,
                batch_expected_outputs,
                batch_neighbours,
            ) = batch
            with torch.no_grad():
                out = model(batch_activations, batch_parts, batch_neighbours)
                # out = model(batch_activations, batch_parts)
                choices = out[:, 0] > threshold
                expected_choices = (batch_expected_outputs > threshold)[:, 0]
                loss = loss_func(out, batch_expected_outputs)
                num_correct += (choices == expected_choices).float().sum()
                num_moles_correct += (
                    ((choices == expected_choices) & expected_choices)
                    .float()
                    .sum()
                )
                num_predicted_moles_total += choices.float().sum()
                num_moles_total += expected_choices.float().sum()
                num_total += len(batch_ids)
        print("valid:", loss)
        print(
            "valid recall:", float(100 * num_moles_correct / num_moles_total),
        )
        print(
            "valid precision:",
            float(100 * num_moles_correct / num_predicted_moles_total),
        )
        print(
            "valid acc:",
            float(100 * num_correct / num_total),
            int(num_correct),
            int(num_total),
        )
        print()


class NeighboursLinearSigmoidModel(torch.nn.Module):
    def __init__(
        self, part_to_id, num_input_features, num_intermediate, num_layers
    ):
        super().__init__()
        self._part_to_id = part_to_id
        self._num_input_features = num_input_features
        self._num_intermediate = num_intermediate
        self._num_layers = num_layers
        num_parts = len(part_to_id)
        self._embedding_len = num_parts // 2
        self.embedding = torch.nn.Embedding(num_parts, self._embedding_len)
        self.sequence = torch.nn.Sequential(
            torch.nn.Linear(
                self._num_input_features
                + self._embedding_len
                + self._num_input_features,
                num_intermediate,
            ),
            torch.nn.BatchNorm1d(num_intermediate),
            torch.nn.ReLU(inplace=True),
            *[
                torch.nn.Sequential(
                    torch.nn.Linear(num_intermediate, num_intermediate),
                    torch.nn.BatchNorm1d(num_intermediate),
                    torch.nn.ReLU(inplace=True),
                )
                for _ in range(self._num_layers)
            ],
            torch.nn.Linear(num_intermediate, 3),
        )

    def init_dict(self):
        return {
            "part_to_id": self._part_to_id,
            "num_input_features": self._num_input_features,
            "num_intermediate": self._num_intermediate,
            "num_layers": self._num_layers,
        }

    def forward(self, activations, parts, neighbour_activations):
        # print(activations.shape)
        # print(parts.shape)
        # print(neighbour_activations.shape)
        # print(activations)
        # print(parts)
        parts_tensor = torch.tensor([self._part_to_id[p] for p in parts])
        parts_embedding = self.embedding(parts_tensor)
        neighbours = torch.mean(neighbour_activations, 1)
        # print(parts_embedding)
        input_ = torch.cat((activations, parts_embedding, neighbours), 1)
        # print(input_)
        seq = self.sequence(input_)
        sig = torch.sigmoid(seq[:, 0:1])
        pos = torch.tanh(seq[:, 1:3])
        return torch.cat([sig, pos], dim=1)


class LinearSigmoidModel(torch.nn.Module):
    def __init__(self, part_to_id):
        super().__init__()
        self._part_to_id = part_to_id
        resnet18_num_features = 512
        num_intermediate = 10
        num_parts = len(part_to_id)
        self._embedding_len = num_parts // 2
        self.embedding = torch.nn.Embedding(num_parts, self._embedding_len)
        self.sequence = torch.nn.Sequential(
            torch.nn.Linear(
                resnet18_num_features + self._embedding_len, num_intermediate
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(num_intermediate, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, activations, parts):
        # print(activations)
        # print(parts)
        parts_tensor = torch.tensor([self._part_to_id[p] for p in parts])
        parts_embedding = self.embedding(parts_tensor)
        # print(parts_embedding)
        input_ = torch.cat((activations, parts_embedding), 1)
        # print(input_)
        return self.sequence(input_)


def get_neighbors(location, tile_size):
    offsets = torch.tensor(
        [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1],]
    )
    return (offsets * tile_size) + location


class NeighboursDataset:
    def __init__(self, tile_dataset):
        self._tile_dataset = tile_dataset
        self._indices = [
            i
            for i, neighbours in enumerate(tile_dataset._neigbour_activations)
            if len(neighbours) == 8
        ]

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError()
        index = self._indices[key]
        vals = self._tile_dataset[index]
        neighbours = torch.stack(
            self._tile_dataset._neigbour_activations[index]
        )
        return tuple(vals) + (neighbours,)


class TileDataset:
    def __init__(self, image_paths, tile_size):
        self._tile_size = tile_size
        self._image_path = []
        self._location = []
        self._activations = []
        self._expected_output = []
        self._part = []
        self._image_location_to_index = {}
        with tqdm.tqdm(image_paths) as pbar:
            for path in pbar:
                self._append_image_data(path)

        self._location = torch.cat(self._location)
        self._activations = torch.cat(self._activations)
        self._expected_output = torch.cat(self._expected_output)
        self._calc_neighbour_activations()

    def _append_image_data(self, image_path):
        datapath = str(image_path) + ".resnet18.pt"
        # datapath = str(image_path) + ".resnet50.pt"
        if not pathlib.Path(datapath).exists():
            return
        location, activations = torch.load(datapath)
        if len(location) != len(activations):
            raise ValueError("Location and activation length mismatch.")
        frame = mel.rotomap.moles.RotomapFrame(image_path)

        expected_output = locations_to_expected_output(
            location, frame.moledata.moles
        )

        self._image_path.extend([image_path] * len(location))
        self._location.append(location)
        self._activations.append(activations)
        self._expected_output.append(expected_output)
        self._part.extend([image_path_to_part(image_path)] * len(location))
        for loc in location:
            index = len(self._image_location_to_index)
            image_location_key = self._image_location_to_key(image_path, loc)
            self._image_location_to_index[image_location_key] = index

    def _calc_neighbour_activations(self):
        self._neigbour_activations = []
        with tqdm.tqdm(range(len(self._image_path))) as pbar:
            for main_index in pbar:
                image_path = self._image_path[main_index]
                loc = self._location[main_index]
                neighbour_activations = []
                for nloc in get_neighbors(loc, self._tile_size):
                    image_location_key = self._image_location_to_key(
                        image_path, nloc
                    )
                    index = self._image_location_to_index.get(
                        image_location_key, None
                    )
                    if index is not None:
                        neighbour_activations.append(self._activations[index])
                self._neigbour_activations.append(neighbour_activations)

    def __len__(self):
        return len(self._image_path)

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError()
        # return key, self._activations[key], self._expected_output[key]
        return (
            key,
            self._activations[key],
            self._part[key],
            self._expected_output[key],
        )

    def _image_location_to_key(self, image_path, location):
        return str(image_path) + " : " + str([int(i) for i in location])


def match_with_neighbours(locations, activations, tile_size):
    matched_locations = []
    matched_activations = []
    matched_neighbours_activations = []

    # Note that you can't really use tensors as dictionary keys, as a newly
    # constructed tensor with the same int values as an existing key won't be
    # found in the dictionary.

    locationtuple_to_activation = {
        loc: act
        for loc, act in zip(int_tensor2d_to_tuple(locations), activations)
    }

    for loc in locations:
        neighbours_activations = []
        for nloc in get_neighbors(loc, tile_size):
            nact = locationtuple_to_activation.get(
                int_tensor1d_to_tuple(nloc), None
            )
            if nact is None:
                break
            neighbours_activations.append(nact)
        if len(neighbours_activations) == 8:
            act = locationtuple_to_activation.get(
                int_tensor1d_to_tuple(loc), None
            )
            matched_locations.append(loc)
            matched_activations.append(act)
            matched_neighbours_activations.append(
                torch.stack(neighbours_activations)
            )

    return list(
        zip(
            matched_locations,
            matched_activations,
            matched_neighbours_activations,
        )
    )


def image_path_to_part(image_path):
    subpart_path = image_path.parent.parent
    part_path = subpart_path.parent
    return f"{part_path.name}, {subpart_path.name}"


def tiles_to_activations(tiles, resnet):
    tile_dataloader = torch.utils.data.DataLoader(tiles, batch_size=64)
    resnet.eval()
    with record_input_context(resnet.avgpool) as batch_activation_tuples:
        with torch.no_grad():
            with tqdm.tqdm(tile_dataloader) as pbar:
                for tiles in pbar:
                    resnet(tiles)

    batch_activations = [
        batch[0].flatten(1) for batch in batch_activation_tuples
    ]

    return torch.cat(batch_activations)


def unique_locations(locations):
    """

        >>> a = torch.tensor([[0, 0], [0, 0]])
        >>> unique_locations(a)
        tensor([[0, 0]])

    """
    new_locations = set(int_tensor2d_to_tuple(locations))
    return torch.tensor(list(new_locations))


def add_neighbour_locations(locations, tile_size):
    """

        >>> a = torch.tensor([[0, 0]])
        >>> add_neighbour_locations(a, 1)
        tensor([[ 0,  0],
                [-1, -1],
                [ 0, -1],
                [ 1, -1],
                [-1,  0],
                [ 1,  0],
                [-1,  1],
                [ 0,  1],
                [ 1,  1]])

    """
    new_locations = [locations]
    for loc in locations:
        new_locations.append(get_neighbors(loc, tile_size))
    return torch.cat(new_locations)


def int_tensor1d_to_tuple(tensor1d):
    """

        >>> a = torch.tensor([1, 2])
        >>> int_tensor1d_to_tuple(a)
        (1, 2)

    """
    if tensor1d.dtype != torch.int64:
        raise ValueError("Must be torch.int64")
    if len(tensor1d.shape) != 1:
        raise ValueError("Must be 1d")
    return tuple(int(c) for c in tensor1d)


def int_tensor2d_to_tuple(tensor2d):
    """

        >>> a = torch.tensor([[1, 2], [3, 4]])
        >>> int_tensor2d_to_tuple(a)
        ((1, 2), (3, 4))

    """
    if tensor2d.dtype != torch.int64:
        raise ValueError("Must be torch.int64")
    if len(tensor2d.shape) != 2:
        raise ValueError("Must be 2d")
    return tuple(tuple(int(c) for c in row) for row in tensor2d)


def reduce_nonmole_locations(
    locations, mole_locations, min_sq_dist=64 * 64, non_mole_chance=0.001
):
    if not mole_locations:
        near_moles = torch.tensor([False] * len(locations))
    else:
        l1_distances = [locations - mole_loc for mole_loc in mole_locations]
        l2sq_distances = [
            (mole_dist * mole_dist).sum(1).unsqueeze(1)
            for mole_dist in l1_distances
        ]
        l2sq_distances = torch.cat(l2sq_distances, 1)
        min_sq_distances = l2sq_distances.min(1).values
        near_moles = min_sq_distances < min_sq_dist

    lucky_locs = torch.rand(len(locations)) < non_mole_chance

    # TODO: use masked_select() somehow.
    # return torch.masked_select(locations, near_moles

    new_locations = []
    for loc, near_mole, lucky_loc in zip(locations, near_moles, lucky_locs):
        if not near_mole and not lucky_loc:
            continue
        new_locations.append(loc)

    return torch.stack(new_locations)


def drop_green_and_edge_locations(image, locations, tile_size=32):
    new_locations = []
    green = [0, 255, 0]
    for loc in locations:
        x1, y1 = loc
        t = image[y1 : y1 + tile_size, x1 : x1 + tile_size]
        if (t[:, :] == green).all():
            continue
        if t.shape != (tile_size, tile_size, 3):
            # TODO: fill-in edge bits with green.
            continue
        new_locations.append(loc)
    if not new_locations:
        return None
    return torch.stack(new_locations)


def image_locations_to_tiles(image, locations, transforms, tile_size=32):
    new_locations = []
    tiles = []
    for loc in locations:
        x1, y1 = loc
        x2, y2 = loc + tile_size
        t = image[y1:y2, x1:x2]
        tiles.append(transforms(t))
        new_locations.append(loc)
    if not new_locations:
        return [], []
    return torch.stack(new_locations), torch.stack(tiles)


def locations_to_expected_output(image_locations, moles, tile_size=32):

    mole_points = [
        (m["x"], m["y"])
        for m in moles
        if "looks_like" not in m or m["looks_like"] == "mole"
    ]
    mole_locations = torch.tensor(list(mole_points))

    if not len(mole_locations):
        return torch.tensor([[0.0, 0.0, 0.0]] * len(image_locations))

    centroids = image_locations + torch.tensor(
        (tile_size // 2, tile_size // 2)
    )

    # Insert an extra dimension for us to fit the comparisons with the moles
    # into. Note that broadcasting matches on the last dimension first, and
    # we'll get shapes like this:
    #
    #   mole_locations.shape = (1,        num_moles, 2)
    #        centroids.shape = (num_locs,         1, 2)
    #
    mole_locations = mole_locations.unsqueeze(0)
    centroids = centroids.unsqueeze(1)
    image_locations = image_locations.unsqueeze(1)

    pos_diff = (mole_locations - centroids) / (tile_size * 0.5)
    pos_diff_sq = pos_diff ** 2
    dist_sq = pos_diff_sq[:, :, 0] + pos_diff_sq[:, :, 1]
    nearest = dist_sq.argmin(1)
    nearest_pos_diff = pos_diff[torch.arange(len(image_locations)), nearest]

    ge_top_left = [
        mole_loc >= image_locations[:, 0, :]
        for mole_loc in mole_locations[0, :, :]
    ]
    lt_bottom_right = [
        mole_loc < image_locations[:, 0, :] + tile_size
        for mole_loc in mole_locations[0, :, :]
    ]
    mole_in_tile = [
        (ge & lt).sum(1) == 2 for ge, lt in zip(ge_top_left, lt_bottom_right)
    ]
    mole_in_tile = torch.stack(mole_in_tile).any(0)

    result = torch.cat(
        [mole_in_tile.float().unsqueeze(1), nearest_pos_diff.float()], dim=1
    )

    assert result.shape == (len(image_locations), 3)

    return result


def image_to_tiles_locations(image, transforms, tile_size=32):
    tiles = []
    locations = []
    green = [0, 255, 0]
    for y1 in range(0, image.shape[0], tile_size):
        for x1 in range(0, image.shape[1], tile_size):
            t = image[y1 : y1 + tile_size, x1 : x1 + tile_size]
            if (t[:, :] == green).all():
                continue
            if t.shape != (tile_size, tile_size, 3):
                # TODO: fill-in edge bits with green.
                continue
            tiles.append(transforms(t))
            locations.append(torch.tensor((y1, x1)))
    return torch.stack(tiles), torch.stack(locations)


def get_image_locations(image, tile_size=32):
    locations = []
    for y1 in range(0, image.shape[0], tile_size):
        for x1 in range(0, image.shape[1], tile_size):
            locations.append(torch.tensor((y1, x1)))
    return torch.stack(locations)


def green_mask_image(image, mask):
    green = numpy.zeros(image.shape, numpy.uint8)
    green[:, :, 1] = 255
    image = cv2.bitwise_and(image, image, mask=mask)
    not_mask = cv2.bitwise_not(mask)
    green = cv2.bitwise_and(green, green, mask=not_mask)
    image = cv2.bitwise_or(image, green)
    return image


def get_tile_locations_activations(
    frame, transforms, resnet, reduce_nonmoles=True
):
    image = frame.load_image()
    mask = frame.load_mask()
    masked_image = green_mask_image(image, mask)
    locations = get_image_locations(masked_image)
    if reduce_nonmoles:
        locations = reduce_nonmole_locations(
            locations, frame.moledata.uuid_points.values()
        )
        locations = add_neighbour_locations(locations, tile_size=32)
        locations = unique_locations(locations)
    locations = drop_green_and_edge_locations(masked_image, locations)
    if locations is None:
        return None
    locations, tiles = image_locations_to_tiles(
        masked_image, locations, transforms
    )
    activations = tiles_to_activations(tiles, resnet)
    return locations, activations


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
