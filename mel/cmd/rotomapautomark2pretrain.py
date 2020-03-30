"""Train to automatically mark moles on rotomap images."""

import contextlib

import cv2
import numpy
import pathlib
import torch
import torchvision

import mel.lib.math
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
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )

    resnet = torchvision.models.resnet18(pretrained=True)

    for path in args.IMAGES:
        if args.verbose:
            print(path)
        frame = mel.rotomap.moles.RotomapFrame(path)
        data = get_tile_locations_activations(frame, transforms, resnet)
        if data is not None:
            torch.save(data, path + ".resnet18.pt")
        else:
            print("Nothing to save.")


def get_tile_locations_activations(frame, transforms, resnet):
    image = frame.load_image()
    mask = frame.load_mask()
    masked_image = green_mask_image(image, mask)
    locations = get_image_locations(masked_image)
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


def get_image_locations(image, tile_size=32):
    locations = []
    for y1 in range(0, image.shape[0], tile_size):
        for x1 in range(0, image.shape[1], tile_size):
            locations.append(torch.tensor((y1, x1)))
    return torch.stack(locations)


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


def locations_to_expected_output(
    image_locations, mole_locations, tile_size=32
):
    if not len(mole_locations):
        return torch.tensor([0.0] * len(image_locations))

    ge_top_left = [mole_loc >= image_locations for mole_loc in mole_locations]
    lt_bottom_right = [
        mole_loc < image_locations + tile_size for mole_loc in mole_locations
    ]
    mole_in_tile = [
        (ge & lt).sum(1) == 2 for ge, lt in zip(ge_top_left, lt_bottom_right)
    ]
    mole_in_tile = torch.stack(mole_in_tile).any(0)
    return mole_in_tile.float()


def image_locations_to_tiles(image, locations, transforms, tile_size=32):
    new_locations = []
    tiles = []
    for loc in locations:
        x1, y1 = loc
        t = image[y1 : y1 + tile_size, x1 : x1 + tile_size]
        tiles.append(transforms(t))
        new_locations.append(loc)
    if not new_locations:
        return [], []
    return torch.stack(new_locations), torch.stack(tiles)


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


def unique_locations(locations):
    """

        >>> a = torch.tensor([[0, 0], [0, 0]])
        >>> unique_locations(a)
        tensor([[0, 0]])

    """
    new_locations = set(int_tensor2d_to_tuple(locations))
    return torch.tensor(list(new_locations))


def tiles_to_activations(tiles, resnet):
    tile_dataloader = torch.utils.data.DataLoader(tiles, batch_size=64)
    resnet.eval()
    with record_input_context(resnet.avgpool) as batch_activation_tuples:
        with torch.no_grad():
            for tiles in tile_dataloader:
                resnet(tiles)

    batch_activations = [
        batch[0].flatten(1) for batch in batch_activation_tuples
    ]

    return torch.cat(batch_activations)


def image_path_to_part(image_path):
    subpart_path = image_path.parent.parent
    part_path = subpart_path.parent
    return f"{part_path.name}, {subpart_path.name}"


class Dataset:
    def __init__(self, image_paths):
        self._image_path = []
        self._location = []
        self._activations = []
        self._expected_output = []
        self._part = []
        for path in image_paths:
            self._append_image_data(path)

        self._location = torch.cat(self._location)
        self._activations = torch.cat(self._activations)
        self._expected_output = torch.cat(self._expected_output)

    def _append_image_data(self, image_path):
        datapath = str(image_path) + ".resnet18.pt"
        if not pathlib.Path(datapath).exists():
            return
        location, activations = torch.load(datapath)
        if len(location) != len(activations):
            raise ValueError("Location and activation length mismatch.")
        frame = mel.rotomap.moles.RotomapFrame(image_path)

        look_like_moles_uuids = [
            m["uuid"]
            for m in frame.moledata.moles
            if "looks_like" not in m or m["looks_like"] == "mole"
        ]
        mole_points = [
            point
            for uuid, point in frame.moledata.uuid_points.items()
            if uuid in look_like_moles_uuids
        ]
        mole_location = torch.tensor(list(mole_points))
        expected_output = locations_to_expected_output(location, mole_location)

        self._image_path.extend([image_path] * len(location))
        self._location.append(location)
        self._activations.append(activations)
        self._expected_output.append(expected_output)
        self._part.extend([image_path_to_part(image_path)] * len(location))

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


def train(model, train_dataloader, valid_dataloader):
    num_epochs = 10
    opt = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        steps_per_epoch=len(train_dataloader),
        max_lr=0.01,
        epochs=num_epochs,
    )
    loss_func = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        print("epoch:", epoch + 1)

        model.train()
        for batch in train_dataloader:
            (
                batch_ids,
                batch_activations,
                batch_parts,
                batch_expected_outputs,
            ) = batch
            opt.zero_grad()
            out = model(batch_activations, batch_parts)
            # print(batch_expected_outputs.unsqueeze(1).shape)
            # print(out.shape)
            loss = loss_func(out, batch_expected_outputs.unsqueeze(1))
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
            ) = batch
            with torch.no_grad():
                out = model(batch_activations, batch_parts)
                choices = out > 0.8
                expected_choices = batch_expected_outputs.unsqueeze(1) > 0.8
                loss = loss_func(out, batch_expected_outputs.unsqueeze(1))
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


def cat_allow_none(left, right):
    if left is None:
        return right
    return torch.cat((left, right))


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
