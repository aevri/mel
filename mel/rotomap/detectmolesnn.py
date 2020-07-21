"""Detect moles in an image, using deep neural nets."""

import contextlib
import functools
import pathlib
import random

import cv2
import math
import numpy
import torch
import torchvision
import tqdm

# import PIL

import mel.lib.math
import mel.rotomap.moles

X_OFFSET = 1
Y_OFFSET = 1
TILE_MAGNIFICATION = 1


to_tensor = torchvision.transforms.ToTensor()


def draw_moles_dist_image(moles, width, height, max_dist=32):
    if not moles:
        return numpy.full((height, width), max_dist, dtype=numpy.float32)

    xpos = numpy.repeat(
        numpy.arange(0, width, dtype=numpy.float32)[numpy.newaxis, :],
        height,
        axis=0,
    )
    assert xpos.shape == (height, width)
    assert xpos[0, 1] > xpos[0, 0]
    assert xpos[0, 0] == xpos[1, 0]
    ypos = numpy.repeat(
        numpy.arange(0, height, dtype=numpy.float32)[:, numpy.newaxis],
        width,
        axis=1,
    )
    assert ypos.shape == (height, width)
    assert ypos[0, 1] == ypos[0, 0]
    assert ypos[1, 0] > ypos[0, 0]

    mole_x = numpy.array([m["x"] for m in moles], dtype=numpy.float32)
    mole_y = numpy.array([m["y"] for m in moles], dtype=numpy.float32)
    num_moles = len(moles)
    xdist = xpos[:, :, numpy.newaxis] - mole_x
    assert xdist.shape == (height, width, num_moles)
    ydist = ypos[:, :, numpy.newaxis] - mole_y
    assert ydist.shape == (height, width, num_moles)
    sqdist = xdist ** 2 + ydist ** 2
    nearest_sqdist = numpy.min(sqdist, axis=2)
    assert nearest_sqdist.shape == (height, width), nearest_sqdist
    dist = numpy.sqrt(nearest_sqdist)
    assert dist.dtype == numpy.float32, dist.dtype
    return numpy.minimum(dist, max_dist)


class FrameDataset:
    def __init__(self, image_paths, tile_size, max_dist):
        self.image_path = list(image_paths)
        self.tile_size = tile_size
        self._transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.1940, 0.9525, 0.1776],
                    std=[0.3537, 0.0972, 0.3244],
                    inplace=True,
                ),
            ]
        )
        self._expected_shape = None
        self.max_dist = max_dist

    def __len__(self):
        return len(self.image_path)

    @functools.lru_cache(10)
    def _getimage(self, path):
        frame = mel.rotomap.moles.RotomapFrame(path)
        image = frame.load_image()
        mask = frame.load_mask()
        image = mel.rotomap.detectmolesnn.green_mask_image(image, mask)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self._transforms(image)

    @functools.lru_cache(10)
    def _get_expected_image(self, path, image_shape):
        frame = mel.rotomap.moles.RotomapFrame(path)
        return draw_moles_dist_image(
            frame.moles, image_shape[2], image_shape[1], max_dist=self.max_dist
        )

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise ValueError("Only int keys are supported.")

        path = self.image_path[key]
        image = self._getimage(path)
        expected_image = self._get_expected_image(path, image.shape)

        if self._expected_shape is None:
            self._expected_shape = image.shape

        if image.shape != self._expected_shape:
            raise ValueError(
                f"image is unexpected shape: {path} {image.shape}"
            )

        x_offset = random.randint(0, image.shape[2] - self.tile_size)
        y_offset = random.randint(0, image.shape[1] - self.tile_size)

        return {
            "key": key,
            "image": image[
                :,
                y_offset : y_offset + self.tile_size,
                x_offset : x_offset + self.tile_size,
            ],
            "expected_image": expected_image[
                numpy.newaxis,
                y_offset : y_offset + self.tile_size,
                x_offset : x_offset + self.tile_size,
            ],
            "path": str(path),
        }


class FrameSampler(torch.utils.data.Sampler):
    def __init__(self, frame_dataset, repeats):
        super().__init__(frame_dataset)
        self.frame_dataset = frame_dataset
        self.repeats = repeats

    def __iter__(self):
        num_items = len(self.frame_dataset)
        indices = list(range(num_items))
        random.shuffle(indices)
        out_count = 0
        for i in indices[: num_items // self.repeats]:
            for _ in range(self.repeats):
                yield i
                out_count += 1
        while out_count < num_items:
            yield indices[-1]
            out_count += 1

    def __len__(self):
        return len(self.frame_dataset)


def get_masked_image(path):
    frame = mel.rotomap.moles.RotomapFrame(path)
    image = frame.load_image()
    mask = frame.load_mask()
    if mask is not None:
        image = green_mask_image(image, mask)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def calc_images_mean_std(image_iter):
    # Weight each image equally, even if they have a different number of
    # pixels.
    # Don't apply Bessel's correction to the variance, because I don't
    # understand it enough to know if it is relevant here.
    mean_sum = 0
    meansq_sum = 0
    num_images = 0
    for image in image_iter:
        num_images += 1
        # Expect a tensor of shape ['Colour', 'Height', 'Width'].
        assert len(image.shape) == 3, image.shape
        assert image.shape[0] == 3, image.shape
        image_view = image.view(3, -1)
        mean_sum += image_view.mean(dim=1)
        meansq_sum += (image_view ** 2).mean(dim=1)

    meansq = meansq_sum / num_images
    mean = mean_sum / num_images
    variance = torch.max(meansq - (mean ** 2), torch.tensor([0.0, 0.0, 0.0]))
    std = variance.sqrt()
    return mean, std


def cat_allow_none(left, right):
    if left is None:
        return right
    return torch.cat((left, right))


def padded_hstack_images(images, num_images):
    if len(images) == num_images:
        return numpy.hstack(images)
    if len(images) > num_images:
        raise Exception("num_images too small.")
    shape = list(images[0].shape)
    shape[1] *= num_images
    montage = numpy.zeros_like(images[0], shape=shape)
    for i, img in enumerate(images):
        x0 = i * images[0].shape[1]
        x1 = x0 + images[0].shape[1]
        montage[:, x0:x1, :] = img
    return montage


def item_atlas(items, items_per_row=10):
    images = numpy.vstack(
        [
            padded_hstack_images(
                [i for i in items[j : j + items_per_row]], items_per_row
            )
            for j in range(0, len(items), items_per_row)
        ]
    )
    return images


def train_epoch(
    device,
    model,
    dataloader,
    loss_func,
    optimizer,
    scheduler,
    input_keys,
    expected_output_keys,
):
    model.train()
    avg_loss = 0
    num_batches = 0
    with tqdm.auto.tqdm(dataloader, disable=False, leave=False) as batcher:
        for batch in batcher:
            optimizer.zero_grad()
            for key in input_keys:
                batch[key].to(device)
            out = model(*[batch[key].to(device) for key in input_keys])
            loss = loss_func(
                out, *[batch[key].to(device) for key in expected_output_keys]
            )
            batcher.set_description(f"loss={float(loss):.4g}")
            avg_loss += loss.item()
            num_batches += 1
            loss.backward()
            optimizer.step()
            scheduler.step()
    # print("train loss:", float(loss))
    return avg_loss / num_batches


def train(
    model,
    train_dataloader,
    valid_dataloader,
    valid_dataset,
    loss_func,
    max_lr,
    num_epochs,
    log_dict,
):
    threshold = 0.8
    log_dict["threshold"] = threshold
    opt = torch.optim.AdamW(model.parameters())
    log_dict["opt"] = "AdamW"
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        steps_per_epoch=len(train_dataloader),
        max_lr=max_lr,
        epochs=num_epochs,
    )
    log_dict["scheduler"] = "OneCycleLR"
    log_dict["max_lr"] = max_lr
    log_dict["num_epochs"] = num_epochs
    # loss_func = torch.nn.MSELoss()
    log_dict["training_loop"] = []
    for epoch in range(num_epochs):
        epoch_dict = {}
        log_dict["training_loop"].append(epoch_dict)
        epoch_dict["num"] = epoch
        print("epoch:", epoch + 1)

        epoch_dict["train_batches"] = []
        model.train()
        with tqdm.auto.tqdm(train_dataloader) as batcher:
            for batch in batcher:
                batch_dict = {}
                epoch_dict["train_batches"].append(batch_dict)
                (
                    batch_ids,
                    batch_activations,
                    batch_parts,
                    batch_expected_outputs,
                    # batch_neighbours,
                ) = batch
                batch_dict["len"] = len(batch_ids)
                opt.zero_grad()
                # out = model(batch_activations, batch_parts, batch_neighbours)
                # out = model(batch_activations, batch_parts)
                out = model(batch_activations)
                # print(batch_expected_outputs.unsqueeze(1).shape)
                # print(out.shape)
                loss = loss_func(out, batch_expected_outputs)
                batch_dict["loss"] = float(loss)
                batcher.set_description(f"loss={float(loss):.3f}")
                loss.backward()
                opt.step()
                scheduler.step()
        print("train loss:", float(loss))

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
        batch_distances = []
        batch_is_true_positive = []
        epoch_dict["valid_batches"] = []
        with tqdm.auto.tqdm(valid_dataloader) as batcher:
            for batch in batcher:
                batch_dict = {}
                epoch_dict["valid_batches"].append(batch_dict)
                (
                    batch_ids,
                    batch_activations,
                    batch_parts,
                    batch_expected_outputs,
                    # batch_neighbours,
                ) = batch
                batch_dict["len"] = len(batch_ids)
                with torch.no_grad():
                    # out = model(batch_activations, batch_parts, batch_neighbours)
                    # out = model(batch_activations, batch_parts)
                    out = model(batch_activations)
                    batch_dict["image_path"] = [
                        str(valid_dataset.image_path[index])
                        for index in batch_ids
                    ]
                    batch_dict["location"] = [
                        [float(x) for x in valid_dataset.location[index]]
                        for index in batch_ids
                    ]
                    batch_dict["output"] = [
                        [float(x) for x in item] for item in out
                    ]
                    batch_dict["expected_output"] = [
                        [float(x) for x in item]
                        for item in batch_expected_outputs
                    ]
                    choices = out[:, 0] > threshold
                    expected_choices = (batch_expected_outputs > threshold)[
                        :, 0
                    ]
                    loss = loss_func(out, batch_expected_outputs)
                    batch_dict["loss"] = float(loss)
                    batcher.set_description(f"loss={float(loss):.3f}")
                    num_correct += (choices == expected_choices).float().sum()

                    pos_diff = out[:, 1:3] - batch_expected_outputs[:, 1:3]
                    pos_diff_sq = pos_diff ** 2
                    dist_sq = (
                        pos_diff_sq[:, 0] + pos_diff_sq[:, 1]
                    ).unsqueeze(1)
                    distance = torch.sqrt(dist_sq)
                    batch_distances.append(distance)

                    is_true_positive = (
                        choices == expected_choices
                    ) & expected_choices
                    batch_is_true_positive.append(is_true_positive)

                    num_moles_correct += (
                        ((choices == expected_choices) & expected_choices)
                        .float()
                        .sum()
                    )
                    num_predicted_moles_total += choices.float().sum()
                    num_moles_total += expected_choices.float().sum()
                    num_total += len(batch_ids)
        print("valid loss:", float(loss))
        print(
            "num_moles:",
            int(num_moles_total),
            f"({100 * num_moles_total / num_total:0.1f})%",
        )
        epoch_dict["num_moles"] = int(num_moles_total)
        print("num_total:", num_total)
        epoch_dict["num_total"] = int(num_total)
        valid_recall = float(100 * num_moles_correct / num_moles_total)
        print("valid recall:", valid_recall)
        epoch_dict["valid_recall"] = valid_recall
        valid_precision = float(
            100 * num_moles_correct / num_predicted_moles_total
        )
        print("valid precision:", valid_precision)
        epoch_dict["valid_precision"] = valid_precision
        valid_accuracy = float(100 * num_correct / num_total)
        epoch_dict["valid_accuracy"] = valid_accuracy
        print("valid acc:", valid_accuracy, int(num_correct), int(num_total))
        distances = torch.cat(batch_distances)
        distances *= 16.0
        true_positives = torch.cat(batch_is_true_positive)
        tp_distances = torch.masked_select(distances, true_positives)
        if len(tp_distances):
            p10 = numpy.percentile(tp_distances, 10)
            epoch_dict["tp distance 10%"] = p10
            print("tp distance 10%", p10)
            p25 = numpy.percentile(tp_distances, 25)
            epoch_dict["tp distance 25%"] = p25
            print("tp distance 25%", p25)
            p50 = numpy.percentile(tp_distances, 50)
            epoch_dict["tp distance 50%"] = p50
            print("tp distance 50%", p50)
            p75 = numpy.percentile(tp_distances, 75)
            epoch_dict["tp distance 75%"] = p75
            print("tp distance 75%", p75)
            p90 = numpy.percentile(tp_distances, 90)
            epoch_dict["tp distance 90%"] = p90
            print("tp distance 90%", p90)
        print()


class LinearSigmoidModel2(torch.nn.Module):
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

        self.resnet_interpreter = torch.nn.Sequential(
            torch.nn.Linear(
                self._num_input_features + self._embedding_len,
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
        )

        self.final = torch.nn.Linear(num_intermediate, 3)

    def init_dict(self):
        return {
            "part_to_id": self._part_to_id,
            "num_input_features": self._num_input_features,
            "num_intermediate": self._num_intermediate,
            "num_layers": self._num_layers,
        }

    def forward(self, activations, parts):
        parts_tensor = torch.tensor([self._part_to_id[p] for p in parts])
        parts_embedding = self.embedding(parts_tensor)
        assert activations.shape == (
            len(activations),
            self._num_input_features,
        )
        interpretations = self.resnet_interpreter(
            torch.cat([activations, parts_embedding], dim=1)
        )
        assert interpretations.shape == (
            len(activations),
            self._num_intermediate,
        )
        seq = self.final(interpretations)
        sig = torch.sigmoid(seq[:, 0:1])
        pos = torch.tanh(seq[:, 1:3]) * 2
        return torch.cat([sig, pos], dim=1)


class NeighboursLinearSigmoidModel2(torch.nn.Module):
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

        self.resnet_interpreter = torch.nn.Sequential(
            torch.nn.Linear(
                self._num_input_features + self._embedding_len,
                num_intermediate,
            ),
            torch.nn.BatchNorm1d(num_intermediate),
            torch.nn.ReLU(inplace=True),
            # *[
            #     torch.nn.Sequential(
            #         torch.nn.Linear(num_intermediate, num_intermediate),
            #         torch.nn.BatchNorm1d(num_intermediate),
            #         torch.nn.ReLU(inplace=True),
            #     )
            #     for _ in range(self._num_layers)
            # ],
        )

        self.integrator = torch.nn.Sequential(
            torch.nn.Linear(num_intermediate * 9, num_intermediate,),
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
        )

        self.final = torch.nn.Linear(num_intermediate, 3)

    def init_dict(self):
        return {
            "part_to_id": self._part_to_id,
            "num_input_features": self._num_input_features,
            "num_intermediate": self._num_intermediate,
            "num_layers": self._num_layers,
        }

    def forward(self, activations, parts, neighbour_activations):
        parts_tensor = torch.tensor([self._part_to_id[p] for p in parts])
        parts_embedding = self.embedding(parts_tensor)
        assert activations.shape == (
            len(activations),
            self._num_input_features,
        )
        activation_list = [
            neighbour_activations[:, i, :] for i in range(8)
        ] + [activations]
        interpretations = [
            self.resnet_interpreter(torch.cat([x, parts_embedding], dim=1))
            for x in activation_list
        ]
        assert len(interpretations) == 9
        assert interpretations[0].shape == (
            len(activations),
            self._num_intermediate,
        )
        integrated = self.integrator(torch.cat(interpretations, dim=1))
        assert integrated.shape == (len(activations), self._num_intermediate,)
        # neighbours = torch.flatten(neighbour_activations, start_dim=1)
        # assert neighbours.shape == (
        #     len(neighbour_activations),
        #     8 * self._num_input_features,
        # ), f"Got {neighbours.shape}."
        # input_ = torch.cat((activations, parts_embedding, neighbours), 1)
        # input_ = torch.cat((activations, parts_embedding, neighbours), 1)
        # seq = self.sequence(input_)
        seq = self.final(integrated)
        sig = torch.sigmoid(seq[:, 0:1])
        pos = torch.tanh(seq[:, 1:3]) * 2
        return torch.cat([sig, pos], dim=1)


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
                + self._num_input_features * 8,
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
        # neighbours = torch.cat(neighbour_activations)
        neighbours = torch.flatten(neighbour_activations, start_dim=1)
        assert neighbours.shape == (
            len(neighbour_activations),
            8 * self._num_input_features,
        ), f"Got {neighbours.shape}."
        # print(parts_embedding)
        input_ = torch.cat((activations, parts_embedding, neighbours), 1)
        # print(input_)
        seq = self.sequence(input_)
        sig = torch.sigmoid(seq[:, 0:1])
        pos = torch.tanh(seq[:, 1:3]) * 2
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
        self._tile_dataset._calc_neighbour_activations()
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


class TileDataset2:
    def __init__(self, image_paths, tile_size):
        self._tile_size = tile_size
        self.image_path = []
        self.location = []
        self._expected_output = []
        self._part = []
        with tqdm.auto.tqdm(image_paths) as pbar:
            for path in pbar:
                self._append_image_data(path)

        self.location = torch.cat(self.location)
        self._expected_output = torch.cat(self._expected_output)

        self._transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )

    def _append_image_data(self, image_path):
        frame = mel.rotomap.moles.RotomapFrame(image_path)
        location = get_tile_locations_for_frame(frame, self._tile_size)
        if location is None:
            return
        expected_output = locations_to_expected_output(
            location, frame.moledata.moles
        )
        self.image_path.extend([image_path] * len(location))
        self.location.append(location)
        self._expected_output.append(expected_output)
        self._part.extend([image_path_to_part(image_path)] * len(location))

    def __len__(self):
        return len(self.location)

    @functools.lru_cache(10)
    def _getimage(self, path):
        frame = mel.rotomap.moles.RotomapFrame(path)
        image = frame.load_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self._transforms(image)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise ValueError("Only int keys are supported.")

        location = self.location[key]
        path = self.image_path[key]
        image = self._getimage(path)

        # back_offset = self._tile_size
        # forward_offset = self._tile_size * 2
        back_offset = 0
        forward_offset = self._tile_size

        x1 = location[0] - back_offset
        x2 = location[0] + forward_offset
        y1 = location[1] - back_offset
        y2 = location[1] + forward_offset

        return (
            key,
            image[:, y1:y2, x1:x2],
            self._part[key],
            self._expected_output[key],
        )


def make_convnet2d(width, depth, channels_in):
    return torch.nn.Sequential(
        torch.nn.BatchNorm2d(channels_in),
        make_cnn_layer(channels_in, width),
        *[make_cnn_layer(width, width) for _ in range(depth - 1)],
        torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        Lambda(lambda x: x.view(x.size(0), -1)),
    )


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def make_cnn_layer(in_width, out_width, stride=2, bias=False):
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            in_width,
            out_width,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
        ),
        torch.nn.BatchNorm2d(out_width),
        torch.nn.ReLU(inplace=True),
    )


class DenseUnetModel(torch.nn.Module):
    def __init__(self, channels_in, channels_per_layer):
        super().__init__()
        self.cnn = DenseUnet(channels_in, channels_per_layer, num_classes=1)

    def init_dict(self):
        return {
            "channels_in": self.cnn.channels_in,
            "channels_per_layer": self.cnn.channels_per_layer,
        }

    def forward(self, images):
        assert len(images.shape) == 4
        maps = self.cnn(images)
        assert maps.shape == (
            len(images),
            1,
            images.shape[2],
            images.shape[3],
        )

        assert images.shape[2] == images.shape[3]

        size = images.shape[2]

        location_map_x = torch.linspace(-1.5, 1.5, steps=size).repeat(size, 1)
        location_map_y = (
            torch.linspace(-1.5, 1.5, steps=size).repeat(size, 1).T
        )

        has_mole = torch.max(
            torch.flatten(maps, start_dim=1), dim=1
        ).values.unsqueeze(1)
        assert has_mole.shape == (len(images), 1), str(has_mole.shape)
        x = (
            torch.flatten(maps[:, 0, :, :] * location_map_x, start_dim=1)
            .mean(dim=1)
            .unsqueeze(1)
        )
        assert x.shape == (len(images), 1)
        y = (
            torch.flatten(maps[:, 0, :, :] * location_map_y, start_dim=1)
            .mean(dim=1)
            .unsqueeze(1)
        )
        assert y.shape == (len(images), 1)

        result = torch.cat([has_mole, x, y], dim=1)
        assert result.shape == (len(images), 3)

        return result


def tensor_size_string(tensor):
    return f"{tensor.element_size() * tensor.numel():,}"


def print_tensor_size(name, tensor):
    # print(f"{name}: {tensor_size_string(tensor)} bytes.")
    pass


def image_loss_max_dist(in_, image_target, max_dist):
    image_in = in_[:, 0].unsqueeze(1)
    assert image_in.shape == image_target.shape, (
        image_in.shape,
        image_target.shape,
    )
    return torch.nn.functional.mse_loss(
        (max_dist - image_in) ** 2, (max_dist - image_target) ** 2,
    )


class DenseUnet(torch.nn.Module):
    def __init__(self, channels_in, channels_per_layer, num_classes):
        super().__init__()
        self.channels_in = channels_in
        self.channels_per_layer = channels_per_layer
        self.num_classes = num_classes

        self.pool = torch.nn.AvgPool2d(3, stride=2, padding=1)

        self.down_cnn1 = make_cnn_layer(
            self.channels_in, self.channels_per_layer
        )
        self.down_cnn2 = make_cnn_layer(
            self.channels_in + self.channels_per_layer, self.channels_per_layer
        )
        self.down_cnn3 = make_cnn_layer(
            self.channels_in + self.channels_per_layer * 2,
            self.channels_per_layer,
        )

        self.bottom_cnn = make_cnn_layer(
            self.channels_in + self.channels_per_layer * 3,
            self.channels_per_layer,
            stride=1,
        )

        self.upsample = torch.nn.Upsample(scale_factor=2)
        # Note: torch.nn.functional.interpolate() is an alternative.

        self.up_cnn3 = make_cnn_layer(
            self.channels_in + self.channels_per_layer * 4,
            self.channels_per_layer,
            stride=1,
        )
        self.up_cnn2 = make_cnn_layer(
            self.channels_in + self.channels_per_layer * 3,
            self.channels_per_layer,
            stride=1,
        )
        self.up_cnn1 = make_cnn_layer(
            self.channels_in + self.channels_per_layer * 2,
            self.num_classes,
            stride=1,
        )

    def init_dict(self):
        return {
            "channels_in": self.channels_in,
            "channels_per_layer": self.channels_per_layer,
            "num_classes": self.num_classes,
        }

    def forward(self, images):
        assert len(images.shape) == 4
        assert images.shape[1] == self.channels_in

        print_tensor_size("images", images)

        down_cnn1_in = images
        print_tensor_size("down_cnn1_in", down_cnn1_in)
        assert down_cnn1_in.shape == (
            len(images),
            self.channels_in,
            *images.shape[2:],
        )
        down_cnn1_out = self.down_cnn1(down_cnn1_in)
        print_tensor_size("down_cnn1_out", down_cnn1_out)
        assert down_cnn1_out.shape == (
            len(images),
            self.channels_per_layer,
            images.shape[2] // 2,
            images.shape[3] // 2,
        )

        down_cnn2_in = torch.cat(
            [self.pool(down_cnn1_in), down_cnn1_out], dim=1
        )
        print_tensor_size("down_cnn2_in", down_cnn2_in)
        assert down_cnn2_in.shape == (
            len(images),
            self.channels_in + self.channels_per_layer,
            images.shape[2] // 2,
            images.shape[3] // 2,
        )
        down_cnn2_out = self.down_cnn2(down_cnn2_in)
        print_tensor_size("down_cnn2_out", down_cnn2_out)
        assert down_cnn2_out.shape == (
            len(images),
            self.channels_per_layer,
            images.shape[2] // 4,
            images.shape[3] // 4,
        )

        down_cnn3_in = torch.cat(
            [self.pool(down_cnn2_in), down_cnn2_out], dim=1
        )
        print_tensor_size("down_cnn3_in", down_cnn3_in)
        assert down_cnn3_in.shape == (
            len(images),
            self.channels_in + self.channels_per_layer * 2,
            images.shape[2] // 4,
            images.shape[3] // 4,
        )
        down_cnn3_out = self.down_cnn3(down_cnn3_in)
        print_tensor_size("down_cnn3_out", down_cnn3_out)
        assert down_cnn3_out.shape == (
            len(images),
            self.channels_per_layer,
            images.shape[2] // 8,
            images.shape[3] // 8,
        )

        bottom_cnn_in = torch.cat(
            [self.pool(down_cnn3_in), down_cnn3_out], dim=1
        )
        print_tensor_size("bottom_cnn_in", bottom_cnn_in)
        assert bottom_cnn_in.shape == (
            len(images),
            self.channels_in + self.channels_per_layer * 3,
            images.shape[2] // 8,
            images.shape[3] // 8,
        )
        bottom_cnn_out = self.bottom_cnn(bottom_cnn_in)
        print_tensor_size("bottom_cnn_out", bottom_cnn_out)
        del bottom_cnn_in
        assert bottom_cnn_out.shape == (
            len(images),
            self.channels_per_layer,
            images.shape[2] // 8,
            images.shape[3] // 8,
        )

        up_cnn3_in = torch.cat(
            [
                down_cnn3_in,
                self.upsample(down_cnn3_out),
                self.upsample(bottom_cnn_out),
            ],
            dim=1,
        )
        del down_cnn3_in
        del down_cnn3_out
        del bottom_cnn_out
        assert up_cnn3_in.shape == (
            len(images),
            self.channels_in + self.channels_per_layer * 4,
            images.shape[2] // 4,
            images.shape[3] // 4,
        )
        up_cnn3_out = self.up_cnn3(up_cnn3_in)
        print_tensor_size("up_cnn3_out", up_cnn3_out)
        del up_cnn3_in
        assert up_cnn3_out.shape == (
            len(images),
            self.channels_per_layer,
            images.shape[2] // 4,
            images.shape[3] // 4,
        )

        up_cnn2_in = torch.cat(
            [
                down_cnn2_in,
                self.upsample(down_cnn2_out),
                self.upsample(up_cnn3_out),
            ],
            dim=1,
        )
        print_tensor_size("up_cnn2_in", up_cnn2_in)
        del down_cnn2_in
        del down_cnn2_out
        del up_cnn3_out
        assert up_cnn2_in.shape == (
            len(images),
            self.channels_in + self.channels_per_layer * 3,
            images.shape[2] // 2,
            images.shape[3] // 2,
        )
        up_cnn2_out = self.up_cnn2(up_cnn2_in)
        print_tensor_size("up_cnn2_out", up_cnn2_out)
        del up_cnn2_in
        assert up_cnn2_out.shape == (
            len(images),
            self.channels_per_layer,
            images.shape[2] // 2,
            images.shape[3] // 2,
        )

        up_cnn1_in = torch.cat(
            [
                down_cnn1_in,
                self.upsample(down_cnn1_out),
                self.upsample(up_cnn2_out),
            ],
            dim=1,
        )
        del down_cnn1_in
        del down_cnn1_out
        del up_cnn2_out
        print_tensor_size("up_cnn1_in", up_cnn1_in)
        assert up_cnn1_in.shape == (
            len(images),
            self.channels_in + self.channels_per_layer * 2,
            images.shape[2],
            images.shape[3],
        )
        up_cnn1_out = self.up_cnn1(up_cnn1_in)
        print_tensor_size("up_cnn1_out", up_cnn1_out)
        del up_cnn1_in
        assert up_cnn1_out.shape == (
            len(images),
            self.num_classes,
            images.shape[2],
            images.shape[3],
        )

        return up_cnn1_out


class DenseCNN(torch.nn.Module):
    def __init__(self, channels_in, channels_per_layer):
        super().__init__()
        self.channels_in = channels_in
        self.channels_per_layer = channels_per_layer

        self.bn_in = torch.nn.BatchNorm2d(self.channels_in)

        self.pool = torch.nn.AvgPool2d(3, stride=2, padding=1)

        self.cnn1 = make_cnn_layer(self.channels_in, self.channels_per_layer)
        self.cnn2 = make_cnn_layer(
            self.channels_in + self.channels_per_layer, self.channels_per_layer
        )
        self.cnn3 = make_cnn_layer(
            self.channels_in + self.channels_per_layer * 2,
            self.channels_per_layer,
        )
        self.cnn4 = make_cnn_layer(
            self.channels_in + self.channels_per_layer * 3,
            self.channels_per_layer,
        )

        # self.flatten = Lambda(lambda x: x.view(x.size(0), -1))

        self.dimension_divisor = 2 ** 4
        self.end_channels = self.channels_in + self.channels_per_layer * 4

    def forward(self, images):
        assert len(images.shape) == 4
        assert images.shape[1] == self.channels_in

        bn = self.bn_in(images)

        cnn1_in = bn
        assert cnn1_in.shape == (
            len(images),
            self.channels_in,
            *images.shape[2:],
        )
        cnn1_out = self.cnn1(cnn1_in)
        assert cnn1_out.shape == (
            len(images),
            self.channels_per_layer,
            images.shape[2] // 2,
            images.shape[3] // 2,
        )

        cnn2_in = torch.cat([self.pool(cnn1_in), cnn1_out], dim=1)
        assert cnn2_in.shape == (
            len(images),
            self.channels_in + self.channels_per_layer,
            images.shape[2] // 2,
            images.shape[3] // 2,
        )
        cnn2_out = self.cnn2(cnn2_in)
        assert cnn2_out.shape == (
            len(images),
            self.channels_per_layer,
            images.shape[2] // 4,
            images.shape[3] // 4,
        )

        cnn3_in = torch.cat([self.pool(cnn2_in), cnn2_out], dim=1)
        assert cnn3_in.shape == (
            len(images),
            self.channels_in + self.channels_per_layer * 2,
            images.shape[2] // 4,
            images.shape[3] // 4,
        )
        cnn3_out = self.cnn3(cnn3_in)
        assert cnn3_out.shape == (
            len(images),
            self.channels_per_layer,
            images.shape[2] // 8,
            images.shape[3] // 8,
        )

        cnn4_in = torch.cat([self.pool(cnn3_in), cnn3_out], dim=1)
        assert cnn4_in.shape == (
            len(images),
            self.channels_in + self.channels_per_layer * 3,
            images.shape[2] // 8,
            images.shape[3] // 8,
        )
        cnn4_out = self.cnn4(cnn4_in)
        assert cnn4_out.shape == (
            len(images),
            self.channels_per_layer,
            images.shape[2] // 16,
            images.shape[3] // 16,
        )

        result = torch.cat([self.pool(cnn4_in), cnn4_out], dim=1)
        assert result.shape == (
            len(images),
            self.end_channels,
            images.shape[2] // self.dimension_divisor,
            images.shape[3] // self.dimension_divisor,
        )
        return result


class LinearSigmoidModel3(torch.nn.Module):
    def __init__(self, part_to_id):
        super().__init__()
        self._part_to_id = part_to_id
        num_parts = len(part_to_id)
        self._embedding_len = num_parts // 2
        self.embedding = torch.nn.Embedding(num_parts, self._embedding_len)

        # self._cnn_width = 128
        channels_in = 3

        self.conv = DenseCNN(channels_in, channels_per_layer=32)
        self.flatten = Lambda(lambda x: x.view(x.size(0), -1))

        # self.conv = torch.nn.Sequential(
        #     torch.nn.BatchNorm2d(channels_in),
        #     make_cnn_layer(channels_in, self._cnn_width // 8),
        #     make_cnn_layer(self._cnn_width // 8, self._cnn_width // 4),
        #     make_cnn_layer(self._cnn_width // 4, self._cnn_width // 2),
        #     make_cnn_layer(self._cnn_width // 2, self._cnn_width // 1),
        #     Lambda(lambda x: x.view(x.size(0), -1)),
        # )

        end_dimension = 96 // self.conv.dimension_divisor
        self.cnn_result_len = self.conv.end_channels * (end_dimension ** 2)
        self.end_width = self.cnn_result_len + self._embedding_len

        self.final = torch.nn.Linear(self.end_width, 3)

    def init_dict(self):
        return {
            "part_to_id": self._part_to_id,
        }

    def forward(self, images, parts):
        parts_tensor = torch.tensor([self._part_to_id[p] for p in parts])
        parts_embedding = self.embedding(parts_tensor)

        features = self.flatten(self.conv(images))

        assert features.shape == (len(images), self.cnn_result_len), str(
            features.shape
        )

        final_input = torch.cat([features, parts_embedding], dim=1)
        assert final_input.shape == (len(images), self.end_width), str(
            final_input.shape
        )

        seq = self.final(final_input)
        sig = torch.sigmoid(seq[:, 0:1])
        pos = torch.tanh(seq[:, 1:3]) * 2
        return torch.cat([sig, pos], dim=1)


class TileDataset:
    def __init__(self, image_paths, tile_size):
        self._tile_size = tile_size
        self.image_path = []
        self.location = []
        self._activations = []
        self._expected_output = []
        self._part = []
        self._image_location_to_index = {}
        with tqdm.tqdm(image_paths) as pbar:
            for path in pbar:
                self._append_image_data(path)

        self.location = torch.cat(self.location)
        self._activations = torch.cat(self._activations)
        self._expected_output = torch.cat(self._expected_output)

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

        self.image_path.extend([image_path] * len(location))
        self.location.append(location)
        self._activations.append(activations)
        self._expected_output.append(expected_output)
        self._part.extend([image_path_to_part(image_path)] * len(location))
        for loc in location:
            index = len(self._image_location_to_index)
            image_location_key = self._image_location_to_key(image_path, loc)
            self._image_location_to_index[image_location_key] = index

    def _calc_neighbour_activations(self):
        self._neigbour_activations = []
        with tqdm.tqdm(range(len(self.image_path))) as pbar:
            for main_index in pbar:
                image_path = self.image_path[main_index]
                loc = self.location[main_index]
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
        return len(self.image_path)

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


def tiles_to_activations_no_flat_cat(tiles, resnet):
    tile_dataloader = torch.utils.data.DataLoader(tiles, batch_size=64)
    resnet.eval()
    with record_input_context(
        resnet.avgpool
    ) as avgpool_in, record_input_context(
        resnet.layer2
    ) as layer2_in, record_input_context(
        resnet.layer3
    ) as layer3_in, record_input_context(
        resnet.layer4
    ) as layer4_in:
        with torch.no_grad():
            for tiles in tile_dataloader:
                resnet(tiles)

    layer2_activations = [tile for batch in layer2_in for tile in batch[0]]
    layer3_activations = [tile for batch in layer3_in for tile in batch[0]]
    layer4_activations = [tile for batch in layer4_in for tile in batch[0]]
    avgpool_activations = [tile for batch in avgpool_in for tile in batch[0]]

    return (
        layer2_activations,
        layer3_activations,
        layer4_activations,
        avgpool_activations,
    )


def tiles_to_activations(tiles, resnet):
    tile_dataloader = torch.utils.data.DataLoader(tiles, batch_size=64)
    resnet.eval()
    with record_input_context(
        resnet.avgpool
    ) as avgpool_in, record_input_context(
        resnet.layer1
    ) as layer1_in, record_input_context(
        resnet.layer2
    ) as layer2_in, record_input_context(
        resnet.layer3
    ) as layer3_in, record_input_context(
        resnet.layer4
    ) as layer4_in:
        with torch.no_grad():
            for tiles in tile_dataloader:
                resnet(tiles)

    layer1_activations = torch.cat(
        [batch[0].flatten(1) for batch in layer1_in]
    )

    layer2_activations = torch.cat(
        [batch[0].flatten(1) for batch in layer2_in]
    )

    layer3_activations = torch.cat(
        [batch[0].flatten(1) for batch in layer3_in]
    )

    layer4_activations = torch.cat(
        [batch[0].flatten(1) for batch in layer4_in]
    )

    avgpool_activations = torch.cat(
        [batch[0].flatten(1) for batch in avgpool_in]
    )

    batch_activations = torch.cat(
        [
            layer1_activations,
            layer2_activations,
            layer3_activations,
            layer4_activations,
            avgpool_activations,
        ],
        dim=1,
    )

    return batch_activations


def tiles_to_central_activations(tiles, resnet, tile_magnification):
    batch_size = 64
    tile_dataloader = torch.utils.data.DataLoader(tiles, batch_size=batch_size)
    resnet.eval()
    with record_input_context(
        resnet.avgpool
    ) as avgpool_in, record_input_context(
        resnet.layer2
    ) as layer2_in, record_input_context(
        resnet.layer3
    ) as layer3_in, record_input_context(
        resnet.layer4
    ) as layer4_in:
        with torch.no_grad():
            for tiles in tile_dataloader:
                resnet(tiles)

    layer2_activations = flat_cat_central_activations(
        layer2_in, tile_magnification
    )
    layer3_activations = flat_cat_central_activations(
        layer3_in, tile_magnification
    )
    layer4_activations = flat_cat_central_activations(
        layer4_in, tile_magnification
    )
    avgpool_activations = flat_cat_central_activations(
        avgpool_in, tile_magnification
    )

    batch_activations = torch.cat(
        [
            layer2_activations,
            layer3_activations,
            layer4_activations,
            avgpool_activations,
        ],
        dim=1,
    )

    return batch_activations


def flat_cat_central_activations(layer, tile_magnification):
    for batch in layer:
        assert len(batch) == 1
    return torch.cat(
        [
            central_activations(batch[0], tile_magnification).flatten(1)
            for batch in layer
        ]
    )


# def central_activations(layer, tile_magnification):
#     assert len(layer.shape) == 4
#     assert layer.shape[-1] == layer.shape[-2]
#     tiles_per_side = 1 + (tile_magnification * 2)
#     blocks_per_tile = layer.shape[-1] / tiles_per_side
#     assert blocks_per_tile == int(blocks_per_tile)
#     blocks_per_tile = int(blocks_per_tile)
#     start = blocks_per_tile * tile_magnification
#     end = start + blocks_per_tile
#     return layer[:, :, start:end, start:end]


def central_activations(layer, tile_magnification):
    assert len(layer.shape) == 4
    assert layer.shape[-1] == layer.shape[-2]
    tiles_per_side = 1 + (tile_magnification * 2)

    blocks_per_tile = layer.shape[-1] / tiles_per_side
    assert blocks_per_tile == int(blocks_per_tile)
    blocks_per_tile = int(blocks_per_tile)

    # start = blocks_per_tile * tile_magnification
    start_x = blocks_per_tile * X_OFFSET
    end_x = start_x + blocks_per_tile
    start_y = blocks_per_tile * Y_OFFSET
    end_y = start_y + blocks_per_tile
    result = layer[:, :, start_y:end_y, start_x:end_x]
    assert result.shape[-1] == blocks_per_tile
    assert result.shape[-2] == blocks_per_tile
    return result


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


def drop_green_and_edge_big_locations(
    image, locations, tile_size, tile_magnification
):
    new_locations = []
    green = [0, 255, 0]

    # back_offset = tile_size * tile_magnification
    # forward_offset = tile_size + (tile_size * tile_magnification)
    big_tile_size = tile_size * (1 + (tile_magnification * 2))

    back_offset_x = tile_size * X_OFFSET
    forward_offset_x = (
        tile_size + (2 * tile_size * tile_magnification) - back_offset_x
    )

    back_offset_y = tile_size * Y_OFFSET
    forward_offset_y = (
        tile_size + (2 * tile_size * tile_magnification) - back_offset_y
    )

    assert back_offset_x + forward_offset_x == big_tile_size
    assert back_offset_y + forward_offset_y == big_tile_size

    for loc in locations:
        x1 = loc[0] - back_offset_x
        x2 = loc[0] + forward_offset_x
        y1 = loc[1] - back_offset_y
        y2 = loc[1] + forward_offset_y
        # x1, y1 = loc
        t = image[
            y1:y2, x1:x2,
        ]
        if (t[:, :] == green).all():
            continue
        if t.shape != (big_tile_size, big_tile_size, 3):
            # TODO: fill-in edge bits with green.
            continue
        new_locations.append(loc)
    if not new_locations:
        return None
    return torch.stack(new_locations)


def image_locations_to_big_tiles(
    image, locations, transforms, tile_size, tile_magnification
):
    new_locations = []
    tiles = []

    # back_offset = tile_size * tile_magnification
    # forward_offset = tile_size + (tile_size * tile_magnification)

    back_offset_x = tile_size * X_OFFSET
    forward_offset_x = (
        tile_size + (2 * tile_size * tile_magnification) - back_offset_x
    )

    back_offset_y = tile_size * Y_OFFSET
    forward_offset_y = (
        tile_size + (2 * tile_size * tile_magnification) - back_offset_y
    )

    big_tile_size = tile_size * (1 + (tile_magnification * 2))

    assert back_offset_x + forward_offset_x == big_tile_size
    assert back_offset_y + forward_offset_y == big_tile_size

    for loc in locations:
        # x1, y1 = loc - back_offset
        # x2, y2 = loc + forward_offset
        x1 = loc[0] - back_offset_x
        x2 = loc[0] + forward_offset_x
        y1 = loc[1] - back_offset_y
        y2 = loc[1] + forward_offset_y
        t = image[y1:y2, x1:x2]
        tiles.append(transforms(t))
        new_locations.append(loc)
    if not new_locations:
        return [], []

    tiles = torch.stack(tiles)
    assert tiles.shape[-1] == big_tile_size
    assert tiles.shape[-2] == big_tile_size

    return torch.stack(new_locations), tiles


def image_locations_to_squished_big_tiles(
    image, locations, transforms, tile_size, tile_magnification
):
    tiles = []

    # back_offset = tile_size * tile_magnification
    # forward_offset = tile_size + (tile_size * tile_magnification)

    back_offset = tile_size * tile_magnification
    forward_offset = tile_size + (tile_size * tile_magnification)

    big_tile_size = tile_size * (1 + (tile_magnification * 2))

    assert back_offset + forward_offset == big_tile_size

    for loc in locations:
        # x1, y1 = loc - back_offset
        # x2, y2 = loc + forward_offset
        x1 = loc[0] - back_offset
        x2 = loc[0] + forward_offset
        y1 = loc[1] - back_offset
        y2 = loc[1] + forward_offset
        t = image[y1:y2, x1:x2]
        assert t.shape[0] == big_tile_size
        assert t.shape[0] == big_tile_size
        t = cv2.resize(t, (tile_size, tile_size))
        tiles.append(transforms(t))

    tiles = torch.stack(tiles)
    assert tiles.shape[-1] == tile_size
    assert tiles.shape[-2] == tile_size

    return tiles


def locations_to_expected_output(image_locations, moles, tile_size=32):

    mole_points = [
        (m["x"], m["y"])
        for m in moles
        if "looks_like" not in m or m["looks_like"] == "mole"
    ]
    mole_locations = torch.tensor(list(mole_points))

    if not len(mole_locations):
        return torch.tensor([[0.0, 0.0, 0.0, 0.0]] * len(image_locations))

    centroids = image_locations + torch.tensor(
        (tile_size // 2, tile_size // 2)
    )

    num_moles = len(mole_locations)

    assert len(image_locations.shape) == 2
    assert image_locations.shape[1] == 2
    num_locs = len(image_locations)

    # Insert an extra dimension for us to fit the comparisons with the moles
    # into. Note that broadcasting matches on the last dimension first, and
    # we'll get shapes like this:
    #
    #   mole_locations.shape = (1,        num_moles, 2)
    #        centroids.shape = (num_locs,         1, 2)
    #
    mole_locations = mole_locations.unsqueeze(0)
    centroids = centroids.unsqueeze(1)
    assert mole_locations.shape == (1, num_moles, 2)
    assert centroids.shape == (num_locs, 1, 2)

    image_locations = image_locations.unsqueeze(1)
    assert len(image_locations.shape) == 3
    assert image_locations.shape == (num_locs, 1, 2)

    pos_diff = (mole_locations - centroids) / (tile_size * 0.5)
    assert pos_diff.shape == (num_locs, num_moles, 2)
    pos_diff_sq = pos_diff ** 2
    dist_sq = pos_diff_sq[:, :, 0] + pos_diff_sq[:, :, 1]
    assert dist_sq.shape == (num_locs, num_moles)
    nearest = dist_sq.argmin(1)
    assert nearest.shape == (num_locs,)

    nearest_pos_diff = pos_diff[torch.arange(len(image_locations)), nearest]
    assert nearest_pos_diff.shape == (num_locs, 2)

    # intilish_manhatten = torch.max(
    #     torch.tensor(0), torch.max(nearest_pos_diff, dim=1) - 1
    # )

    # nearest_dist_sq = dist_sq[torch.arange(len(image_locations)), nearest]
    # assert nearest_dist_sq.shape == (num_locs,)
    # nearest_dist = nearest_dist_sq.sqrt()
    # assert nearest_dist.shape == (num_locs,)
    # nearest_distmoid = torch.sigmoid(10 - (nearest_dist * 8))
    # # nearest_distmoid = nearest_dist
    # assert nearest_distmoid.shape == (num_locs,)

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
    mole_in_tile = torch.stack(mole_in_tile).any(0).unsqueeze(1)
    assert mole_in_tile.shape == (num_locs, 1)

    # nearest_dist_sq = dist_sq[torch.arange(num_locs), nearest]
    # assert nearest_dist_sq.shape == (num_locs,)
    # nearest_dist = nearest_dist_sq.sqrt()
    # assert nearest_dist.shape == (num_locs,)

    # result = torch.cat(
    #     [distmoid(nearest_dist).float().unsqueeze(1), nearest_pos_diff.float()],
    #     dim=1,
    # )

    result = torch.cat(
        [
            mole_in_tile.float(),
            intilish_manhatten(nearest_pos_diff).float().unsqueeze(1),
            nearest_pos_diff.float(),
        ],
        dim=1,
    )

    # assert result.shape == (len(image_locations), 3)
    assert result.shape == (len(image_locations), 4)

    return result


def distmoid(nearest_dist):
    nearest_distmoid = torch.sigmoid(10 - (nearest_dist * 8))
    # nearest_distmoid = nearest_dist
    assert nearest_distmoid.shape == (len(nearest_dist),)
    return nearest_distmoid


def intilish_manhatten(pos_diff):
    max_coord = torch.max(torch.abs(pos_diff), dim=1).values
    inside_box_always_zero = torch.max(torch.tensor(0.0), max_coord - 1.0)
    max_is_across_next_box = torch.min(
        torch.tensor(2.0), inside_box_always_zero
    )
    range_is_0_to_1 = max_is_across_next_box * 0.5
    range_is_1_to_0 = 1.0 - range_is_0_to_1
    # return range_is_1_to_0
    return range_is_1_to_0 ** 2
    # return torch.sigmoid(range_is_1_to_0 * 8 - 4)
    # return torch.sigmoid(range_is_1_to_0 * 10 - 6)
    # return torch.sigmoid(range_is_1_to_0 * 10 - 8)
    # return torch.sigmoid(range_is_1_to_0 * 8 + 2)


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


def get_image_locations(image, tile_size=32, xoffset=0, yoffset=0):
    locations = []
    for y1 in range(xoffset, image.shape[0], tile_size):
        for x1 in range(yoffset, image.shape[1], tile_size):
            locations.append(torch.tensor((y1, x1)))
    return torch.stack(locations)


def get_image_locations_activations(image, tile_size, transforms, resnet):
    assert tile_size == 32
    locations = get_image_locations(image, tile_size)
    layer_activation_list = get_image_activations(image, transforms, resnet)

    # locations, tiles = image_locations_to_tiles(image, locations, transforms)
    # activations = tiles_to_activations(tiles, resnet)

    # activations2 = tiles_to_activations_no_flat_cat(tiles, resnet)
    # layer_activation_list = []
    # for layer in activations2:
    #     assert len(layer[0].shape) == 3

    #     image_height, image_width = image.shape[:2]
    #     assert tile_size == ((image_height * image_width) / len(layer)) ** 0.5
    #     feature_blocks_per_tile_side = layer[0].shape[1]
    #     assert feature_blocks_per_tile_side == layer[0].shape[2]

    #     image_width_in_tiles = image_width / tile_size
    #     assert image_width_in_tiles == int(image_width_in_tiles)
    #     image_width_in_tiles = int(image_width_in_tiles)

    #     tensor_width = image_width_in_tiles * feature_blocks_per_tile_side
    #     assert isinstance(tensor_width, int)

    #     image_height_in_tiles = image_height / tile_size
    #     assert image_height_in_tiles == int(image_height_in_tiles)
    #     image_height_in_tiles = int(image_height_in_tiles)

    #     tensor_height = image_height_in_tiles * feature_blocks_per_tile_side
    #     assert isinstance(tensor_height, int)

    #     layer_tensor = torch.empty(
    #         layer[0].shape[0], tensor_height, tensor_width
    #     )
    #     for tile_y in range(image_height_in_tiles):
    #         for tile_x in range(image_width_in_tiles):
    #             tile_index = tile_x + tile_y * image_width_in_tiles
    #             layer_y = tile_y * feature_blocks_per_tile_side
    #             layer_x = tile_x * feature_blocks_per_tile_side
    #             layer_tensor[
    #                 :,
    #                 layer_y : layer_y + feature_blocks_per_tile_side,
    #                 layer_x : layer_x + feature_blocks_per_tile_side,
    #             ] = layer[tile_index]
    #     layer_activation_list.append(layer_tensor)

    # Note: assuming activations are in same left-right top-bottom order as
    # image.

    divisors = []
    for layer in layer_activation_list:
        width = layer.shape[-1]
        height = layer.shape[-2]
        float_divisor = image.shape[1] / width
        divisor = int(float_divisor)
        if divisor != float_divisor or divisor != image.shape[0] / height:
            raise NotImplementedError(
                "Need to handle images that don't tile perfectly"
            )
        assert len(layer.shape) == 3
        divisors.append(divisor)

    tile_activations = []
    for y1 in range(0, image.shape[0], tile_size):
        y2 = y1 + tile_size
        for x1 in range(0, image.shape[1], tile_size):
            x2 = x1 + tile_size
            activations_temp = []
            for divisor, layer in zip(divisors, layer_activation_list):
                activations_temp.append(
                    layer[
                        :,
                        y1 // divisor : y2 // divisor,
                        x1 // divisor : x2 // divisor,
                    ]
                )
            tile_activations.append(
                torch.cat([a.flatten() for a in activations_temp])
            )

    activations = torch.stack(tile_activations)

    assert len(locations) == len(activations)
    assert len(activations.shape) == 2
    assert all(a.shape == activations[0].shape for a in activations)

    return locations, activations


def get_image_activations(image, transforms, resnet):
    tile_dataloader = torch.utils.data.DataLoader(
        [transforms(image)], batch_size=1
    )
    resnet.eval()
    with record_input_context(
        resnet.avgpool
    ) as avgpool_in, record_input_context(
        resnet.layer2
    ) as layer2_in, record_input_context(
        resnet.layer3
    ) as layer3_in, record_input_context(
        resnet.layer4
    ) as layer4_in:
        with torch.no_grad():
            for tiles in tile_dataloader:
                resnet(tiles)

    # resnet.eval()
    # with record_input_context(
    #     resnet.avgpool
    # ) as avgpool_in, record_input_context(
    #     resnet.layer2
    # ) as layer2_in, record_input_context(
    #     resnet.layer3
    # ) as layer3_in, record_input_context(
    #     resnet.layer4
    # ) as layer4_in:
    #     with torch.no_grad():
    #         resnet(transforms(image).unsqueeze(0))

    assert layer2_in[0][0].shape[0] == 1
    assert layer3_in[0][0].shape[0] == 1
    assert layer4_in[0][0].shape[0] == 1
    assert avgpool_in[0][0].shape[0] == 1

    activations = [
        layer2_in[0][0][0],
        layer3_in[0][0][0],
        layer4_in[0][0][0],
        avgpool_in[0][0][0],
    ]

    assert all([len(x.shape) == 3 for x in activations])

    return activations


def green_mask_image(image, mask):
    if mask is None:
        raise ValueError("mask must not be None.")
    green = numpy.zeros(image.shape, numpy.uint8)
    green[:, :, 1] = 255
    image = cv2.bitwise_and(image, image, mask=mask)
    not_mask = cv2.bitwise_not(mask)
    green = cv2.bitwise_and(green, green, mask=not_mask)
    image = cv2.bitwise_or(image, green)
    return image


def green_expand_image_to_full_tiles(image, tile_size):
    height, width = image.shape[:2]
    big_height = math.ceil(height / tile_size) * tile_size
    big_width = math.ceil(width / tile_size) * tile_size

    if big_height == height and big_width == width:
        return image

    big_image = numpy.zeros((big_height, big_width, 3), numpy.uint8)
    big_image[:, :, 1] = 255
    big_image[:height, :width, :] = image[:, :, :]
    return big_image


def get_tile_locations_for_frame(frame, tile_size):
    image = frame.load_image()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = frame.load_mask()
    image = green_mask_image(image, mask)
    locations = torch.cat(
        [
            get_image_locations(image),
            # get_image_locations(image, xoffset=16),
            # get_image_locations(image, yoffset=16),
            # get_image_locations(image, xoffset=16, yoffset=16),
        ]
    )
    # locations = reduce_nonmole_locations(
    #     locations, frame.moledata.uuid_points.values()
    # )
    locations = unique_locations(locations)
    tile_magnification = TILE_MAGNIFICATION

    locations = drop_green_and_edge_big_locations(
        image, locations, tile_size, tile_magnification
    )
    return locations


def get_tile_locations_activations(
    frame, transforms, resnet, reduce_nonmoles=True
):
    tile_size = 32
    image = frame.load_image()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = PIL.Image.open(frame.path).convert("RGB")
    # image = numpy.array(image)
    mask = frame.load_mask()
    image = green_mask_image(image, mask)
    image = green_expand_image_to_full_tiles(image, tile_size)
    locations = torch.cat(
        [
            get_image_locations(image),
            get_image_locations(image, xoffset=16),
            get_image_locations(image, yoffset=16),
            get_image_locations(image, xoffset=16, yoffset=16),
        ]
    )
    # locations, activations = get_image_locations_activations(
    #     image, tile_size, transforms, resnet
    # )
    # locations_tuple = int_tensor2d_to_tuple(locations)
    if reduce_nonmoles:
        locations = reduce_nonmole_locations(
            locations, frame.moledata.uuid_points.values()
        )
        # locations = add_neighbour_locations(locations, tile_size=32)
        locations = unique_locations(locations)
    # locations = drop_green_and_edge_locations(image, locations)

    tile_magnification = TILE_MAGNIFICATION

    locations = drop_green_and_edge_big_locations(
        image, locations, tile_size, tile_magnification
    )
    if locations is None:
        return None

    # locations, tiles = image_locations_to_tiles(
    tiles = image_locations_to_squished_big_tiles(
        image, locations, transforms, tile_size, tile_magnification
    )
    activations = tiles_to_activations(tiles, resnet)

    # _, tiles2 = image_locations_to_big_tiles(
    _, tiles2 = image_locations_to_tiles(
        image, locations, transforms, tile_size
    )
    activations2 = tiles_to_activations(tiles2, resnet)
    # activations2 = tiles_to_central_activations(
    #     tiles2, resnet, tile_magnification
    # )

    # locations_to_activations = {
    #     location: activation
    #     for location, activation in zip(locations_tuple, activations)
    # }
    # filtered_locations_tuple = int_tensor2d_to_tuple(locations)
    # activations = torch.stack(
    #     [
    #         locations_to_activations[location]
    #         for location in filtered_locations_tuple
    #     ]
    # )
    assert len(locations) == len(activations)
    assert len(activations2) == len(activations)
    assert locations.shape == (len(locations), 2)
    assert len(activations.shape) == 2
    assert len(activations2.shape) == 2

    activations_mixed = torch.cat([activations, activations2], dim=1)
    assert len(activations_mixed.shape) == 2
    assert len(activations_mixed) == len(activations)
    assert (
        activations_mixed.shape[1]
        == activations.shape[1] + activations2.shape[1]
    )

    return locations, activations_mixed


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
