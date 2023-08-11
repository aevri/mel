"""Identify moles from their positions in images."""

import collections
import json
import pathlib

import pandas as pd
import torch
from tqdm.auto import tqdm

import mel.lib.fs
import mel.rotomap.dataset


def make_identifier():
    melroot = mel.lib.fs.find_melroot()
    model_dir = melroot / mel.lib.fs.DEFAULT_CLASSIFIER_PATH
    model_path = model_dir / "identify2.pth"
    metadata_path = model_dir / "identify2.json"
    return MoleIdentifier(metadata_path, model_path)


class MoleIdentifier:
    def __init__(self, metadata_path, model_path):
        # Some of these imports are expensive, so to keep program start-up time
        # lower, import them only when necessary.
        import torch

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.model = mel.rotomap.identifynn2.PosOnly(
            partnames_uuids=self.metadata["partnames_uuids"],
            num_neighbours=self.metadata["num_neighbours"],
        )
        self.model.load_state_dict(torch.load(model_path))

    def get_new_moles(self, path, extra_stem=None):
        import torch

        path = pathlib.Path(path)

        pathname = (
            f"{path.parent.parent.parent.stem}/{path.parent.parent.stem}"
        )

        old_moles, uuid_points = mel.rotomap.dataset.imagemoles_from_framepath(
            path, extra_stem=extra_stem
        )

        if not old_moles:
            return []

        x = self.model.prepare_batch([(pathname, uuid_points)])

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            preds = torch.argmax(logits, dim=1)

        new_moles = []
        for i, mole in enumerate(old_moles):
            if old_moles[i][mel.rotomap.moles.KEY_IS_CONFIRMED]:
                new_moles.append(old_moles[i])
                continue
            pred_i = preds[i]
            pred_uuid = self.model.uuids_map.int_to_item(int(pred_i))
            if pred_uuid is None:
                continue
            old_moles[i]["uuid"] = pred_uuid
            new_moles.append(old_moles[i])

        return new_moles


def mole_data_from_uuid_points(uuid_points, num_neighbours=4):
    """Given a list of (uuid, pos_xy) tuples, determine each mole's context."""

    if not uuid_points:
        return []

    actual_num_neighbours = min(num_neighbours, len(uuid_points) - 1)

    positions_tensor = torch.tensor([pos_xy for _, pos_xy in uuid_points])
    distances = torch.cdist(positions_tensor, positions_tensor)
    topk_distances, indices = torch.topk(
        distances, actual_num_neighbours + 1, largest=False, sorted=True
    )

    # Note each point is in it's own nearest neighbors list.

    uuids = [u for u, _ in uuid_points]
    points = [p for _, p in uuid_points]
    padding = num_neighbours - actual_num_neighbours
    x = [
        make_mole_row(uuids, points, row_indices, distances[i], padding)
        for i, row_indices in enumerate(indices)
    ]

    return x


def make_mole_row(uuids, points, indices, distances, padding):
    i = indices[0]
    self_item = (None, points[i], distances[i])
    self_point = points[i]
    neighbours = [
        (uuids[i], points[i] - self_point, distances[i]) for i in indices[1:]
    ]
    if padding:
        neighbours += [(None, None, None) for _ in range(padding)]
    return [self_item] + neighbours


class UuidMap:
    def __init__(self, uuids):
        self.uuidmap = {u: i for i, u in enumerate([None] + sorted(uuids))}

    def uuid_to_int(self, uuid):
        return self.uuidmap[uuid]

    def int_to_uuid(self, i):
        try:
            return list(self.uuidmap.keys())[i]
        except IndexError as e:
            raise IndexError(
                f"Could not interpret index {i}, "
                f"only have {len(self.uuidmap)} values."
            ) from e

    def intlist_to_uuids(self, int_list):
        return [self.int_to_uuid(i) for i in int_list]

    def uuidlist_to_ints(self, uuid_list):
        return [self.uuid_to_int(i) for i in uuid_list]

    def __len__(self):
        return len(self.uuidmap)


def infer_uuids(model, x):
    ids = model(x)
    part_name, _ = x
    return model.partnames_uuidmap[part_name].intlist_to_uuids(ids)


def part_uuids_to_indices(model, data):
    part_name, mole_data, path = data
    uuids = [x[0] for x in mole_data]
    return [model.uuids_map.item_to_int(uuid) for uuid in uuids]


class ResBlock(torch.nn.Module):
    def __init__(self, submodule):
        super().__init__()
        self.submodule = submodule

    def forward(self, x):
        return self.submodule(x) + x


class IndexMap:
    def __init__(self, items):
        self._item_to_int = {
            item: i for i, item in enumerate([None] + sorted(set(items)))
        }
        self._int_to_item = {i: item for item, i in self._item_to_int.items()}
        assert self._item_to_int[None] == 0
        assert self._int_to_item[0] == None

    def item_to_int(self, item):
        i = self._item_to_int.get(item, None)
        if i is None:
            return 0
        return i

    def int_to_item(self, i):
        return self._int_to_item.get(i)

    def items_ints(self):
        return self._item_to_int.items()

    def __contains__(self, x):
        return x in self._item_to_int

    def __len__(self):
        return len(self._item_to_int)


class SinusoidalEncoding(torch.nn.Module):
    def __init__(self, dimensions):
        super(SinusoidalEncoding, self).__init__()
        self.dimensions = dimensions

    def forward(self, x):
        pos_enc = torch.zeros(x.size(0), 2 * self.dimensions).to(x.device)

        base = 3

        for i in range(self.dimensions):
            pos_enc[:, 2 * i] = torch.sin(
                x[:, 0] / (base ** (2 * i / self.dimensions))
            )
            pos_enc[:, 2 * i + 1] = torch.cos(
                x[:, 1] / (base ** (2 * i / self.dimensions))
            )

        return pos_enc


class PosEncoder(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(2, width, bias=True),
            torch.nn.ReLU(),
            ResBlock(
                torch.nn.Sequential(
                    torch.nn.BatchNorm1d(width),
                    torch.nn.Linear(width, width, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(width),
                    torch.nn.Linear(width, width, bias=True),
                    torch.nn.ReLU(),
                )
            ),
            ResBlock(
                torch.nn.Sequential(
                    torch.nn.BatchNorm1d(width),
                    torch.nn.Linear(width, width, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(width),
                    torch.nn.Linear(width, width, bias=True),
                    torch.nn.ReLU(),
                )
            ),
        )

    def forward(self, x):
        return self.encoder(x)


def prepare_default_batch(batch, num_neighbours, partnames_map, uuids_map):
    batch = [
        (
            item[0],
            mole_data_from_uuid_points(item[1], num_neighbours=num_neighbours),
        )
        for item in batch
    ]

    partname_indices = []
    pos_values = []
    uuid_values = []

    def convert_none_pos(pos):
        if pos is None:
            return 0.0, 0.0
        x, y = pos
        return float(x), float(y)

    for x in batch:
        part_name, mole_list = x
        num_moles = len(mole_list)
        partname_indices.extend(
            [partnames_map.item_to_int(part_name)] * num_moles
        )
        pos_values.extend(
            [[convert_none_pos(m[1]) for m in mole] for mole in mole_list]
        )
        uuid_values.extend([[m[0] for m in mole] for mole in mole_list])

    uuid_values = [
        [uuids_map.item_to_int(u) for u in uuids] for uuids in uuid_values
    ]

    partname_indices = torch.tensor(
        partname_indices, dtype=torch.long, requires_grad=False
    )
    pos_values = torch.tensor(
        pos_values, dtype=torch.float32, requires_grad=False
    )
    uuid_values = torch.tensor(
        uuid_values, dtype=torch.long, requires_grad=False
    )

    return partname_indices, pos_values, uuid_values


class PosModel(torch.nn.Module):
    def __init__(self, partnames_uuids, num_neighbours):
        super().__init__()
        self.num_neighbours = num_neighbours
        self.width = 256
        self.selfpos_encoder = PosEncoder(self.width)
        self.relpos_encoder = PosEncoder(self.width)
        all_partnames = list(partnames_uuids.keys())
        all_uuids = [
            uuid for uuids in partnames_uuids.values() for uuid in uuids
        ]

        self.partnames_map = IndexMap(all_partnames)
        self.uuids_map = IndexMap(all_uuids)

        self.partnames_embedding = torch.nn.Embedding(
            len(self.partnames_map), self.width
        )
        transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.width, nhead=8, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(
            transformer_layer, num_layers=2
        )
        self.pool = torch.nn.AdaptiveMaxPool1d(1)

    def freeze_except_classifier(self):
        for sub in [
            self.selfpos_encoder,
            self.relpos_encoder,
            self.transformer,
        ]:
            for p in sub.parameters():
                p.requires_grad = False

    def prepare_batch(self, batch):
        return prepare_default_batch(
            batch,
            self.num_neighbours,
            self.partnames_map,
            self.uuids_map,
        )

    def update_partnames_uuids(self, partnames_uuids):
        all_partnames = list(partnames_uuids.keys())
        all_uuids = [
            uuid for uuids in partnames_uuids.values() for uuid in uuids
        ]

        new_partnames_map = IndexMap(all_partnames)
        new_uuids_map = IndexMap(all_uuids)

        new_partnames_embedding = torch.nn.Embedding(
            len(new_partnames_map), self.width
        )
        with torch.no_grad():
            for partname, i in new_partnames_map.items_ints():
                if partname in self.partnames_map:
                    old_index = self.partnames_map.item_to_int(partname)
                    new_partnames_embedding.weight[
                        i
                    ] = self.partnames_embedding.weight[old_index]

        self.partnames_uuids = partnames_uuids
        self.partnames_map = new_partnames_map
        self.uuids_map = new_uuids_map
        self.partnames_embedding = new_partnames_embedding

    def forward(self, batch):
        partname_indices, pos_values, uuid_values = batch

        partname_embedding = self.partnames_embedding(partname_indices)
        selfpos_emb = self.selfpos_encoder(pos_values[:, 0])

        relpos_embs = []
        for i in range(1, self.num_neighbours + 1):
            relpos_emb = self.relpos_encoder(pos_values[:, i])
            relpos_embs.append(relpos_emb)

        emb_sequence = torch.stack(relpos_embs).transpose(
            0, 1
        )  # Shape: (batch_size, sequence_length, embed_dim)

        transformer_output = self.transformer(
            emb_sequence
        )  # shape: (batch_size, sequence_length, embed_dim)

        pooled_output = self.pool(transformer_output.permute(0, 2, 1)).squeeze(
            -1
        )  # shape: (batch_size, embed_dim)

        # return pooled_output

        # transformer_output_flat = transformer_output.view(
        #     transformer_output.size(0), -1
        # )  # shape: (batch_size, sequence_length * embed_dim)

        return torch.cat(
            [selfpos_emb, pooled_output, partname_embedding], dim=-1
        )


class PosOnly(torch.nn.Module):
    def __init__(self, partnames_uuids, num_neighbours):
        super().__init__()
        self.pos_model = PosModel(partnames_uuids, num_neighbours)
        self.num_neighbours = num_neighbours
        self.width = self.pos_model.width
        all_uuids = [
            uuid for uuids in partnames_uuids.values() for uuid in uuids
        ]

        self.uuids_map = IndexMap(all_uuids)
        self.classifier = torch.nn.Linear(
            self.width * (3), len(self.uuids_map)
        )
        # self.classifier = Classifier(
        #     self.width * (2 + self.num_neighbours), 128, len(self.uuids_map)
        # )
        # self.classifier = torch.nn.Linear(self.width, len(self.uuids_map))

    def freeze_except_classifier(self):
        self.pos_model.freeze_except_classifier()

    def prepare_batch(self, batch):
        return self.pos_model.prepare_batch(batch)

    def update_partnames_uuids(self, partnames_uuids):
        all_partnames = list(partnames_uuids.keys())
        all_uuids = [
            uuid for uuids in partnames_uuids.values() for uuid in uuids
        ]

        new_partnames_map = IndexMap(all_partnames)
        new_uuids_map = IndexMap(all_uuids)

        new_classifier = torch.nn.Linear(
            self.width * (2 + self.num_neighbours), len(new_uuids_map)
        )
        with torch.no_grad():
            for uuid, i in new_uuids_map.items_ints():
                if uuid in self.uuids_map:
                    old_index = self.uuids_map.item_to_int(uuid)
                    new_classifier.weight[i] = self.classifier.weight[
                        old_index
                    ]
                    new_classifier.bias[i] = self.classifier.bias[old_index]

        self.partnames_map = new_partnames_map
        self.uuids_map = new_uuids_map
        self.classifier = new_classifier

    def forward(self, batch):
        pos_emb = self.pos_model(batch)
        return self.classifier(pos_emb)


class Classifier(torch.nn.Module):
    def __init__(self, width_in, width_hidden, num_classes):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(width_in),
            torch.nn.Linear(width_in, width_hidden, bias=True),
            torch.nn.ReLU(),
            ResBlock(
                torch.nn.Sequential(
                    torch.nn.BatchNorm1d(width_hidden),
                    torch.nn.Linear(width_hidden, width_hidden, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(width_hidden),
                    torch.nn.Linear(width_hidden, width_hidden, bias=True),
                    torch.nn.ReLU(),
                )
            ),
            ResBlock(
                torch.nn.Sequential(
                    torch.nn.BatchNorm1d(width_hidden),
                    torch.nn.Linear(width_hidden, width_hidden, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(width_hidden),
                    torch.nn.Linear(width_hidden, width_hidden, bias=True),
                    torch.nn.ReLU(),
                )
            ),
            torch.nn.Linear(width_hidden, num_classes, bias=True),
        )

    def forward(self, x):
        return self.decoder(x)


class IdentityModel(torch.nn.Module):
    def __init__(self, partnames_uuids, num_neighbours):
        super().__init__()
        self.num_neighbours = num_neighbours
        self.width = 64
        all_partnames = list(partnames_uuids.keys())
        all_uuids = [
            uuid for uuids in partnames_uuids.values() for uuid in uuids
        ]

        self.partnames_map = IndexMap(all_partnames)
        self.uuids_map = IndexMap(all_uuids)

        self.uuid_embedding = torch.nn.Embedding(
            len(self.uuids_map), self.width
        )

    def freeze_except_classifier(self):
        for sub in []:
            for p in sub.parameters():
                p.requires_grad = False

    def prepare_batch(self, batch):
        return prepare_default_batch(
            batch,
            self.num_neighbours,
            self.partnames_map,
            self.uuids_map,
        )

    def update_partnames_uuids(self, partnames_uuids):
        raise NotImplementedError()

    def forward(self, batch):
        partname_indices, pos_values, uuid_values = batch
        identity_embedding = self.uuid_embedding(uuid_values)
        return identity_embedding.view(identity_embedding.size(0), -1)


class IdentityOnly(torch.nn.Module):
    def __init__(self, partnames_uuids, num_neighbours):
        super().__init__()
        self.identity_model = IdentityModel(partnames_uuids, num_neighbours)
        self.num_neighbours = num_neighbours
        all_uuids = [
            uuid for uuids in partnames_uuids.values() for uuid in uuids
        ]

        self.uuids_map = IndexMap(all_uuids)
        self.classifier = torch.nn.Linear(
            self.identity_model.width * (1 + self.num_neighbours),
            len(self.uuids_map),
        )

    def freeze_except_classifier(self):
        self.identity_model.freeze_except_classifier()

    def prepare_batch(self, batch):
        return self.identity_model.prepare_batch(batch)

    def update_partnames_uuids(self, partnames_uuids):
        raise NotImplementedError()

    def forward(self, batch):
        identity_emb = self.identity_model(batch)
        return self.classifier(identity_emb)


class IdentityPos(torch.nn.Module):
    def __init__(self, partnames_uuids, num_neighbours):
        super().__init__()
        self.identity_model = IdentityModel(partnames_uuids, num_neighbours)
        self.pos_model = PosModel(partnames_uuids, num_neighbours)
        self.num_neighbours = num_neighbours
        all_uuids = [
            uuid for uuids in partnames_uuids.values() for uuid in uuids
        ]

        self.uuids_map = IndexMap(all_uuids)
        self.classifier = torch.nn.Linear(
            self.identity_model.width
            * (1 + 2 + self.num_neighbours + self.num_neighbours),
            len(self.uuids_map),
        )

    def freeze_except_classifier(self):
        self.identity_model.freeze_except_classifier()
        self.pos_model.freeze_except_classifier()

    def prepare_batch(self, batch):
        return self.identity_model.prepare_batch(batch)

    def update_partnames_uuids(self, partnames_uuids):
        raise NotImplementedError()

    def forward(self, batch):
        identity_emb = self.identity_model(batch)
        pos_emb = self.pos_model(batch)
        emb = torch.cat([identity_emb, pos_emb], dim=-1)
        return self.classifier(emb)


class EarlyStoppingException(Exception):
    pass


def identity(x):
    return x


def make_mask_with_limited_ones(rows, cols, num_ones, device):
    values = torch.randn(rows, cols, device=device)
    _, indices = values.topk(num_ones, dim=1)
    mask = torch.zeros(rows, cols, dtype=torch.bool, device=device)
    mask.scatter_(1, indices, True)
    return mask


def zero_some_items_in_sequence(t, num_items_to_zero=2):
    t = t.clone()
    batch_size, seq_length, _ = t.size()
    mask = make_mask_with_limited_ones(
        batch_size, seq_length, num_items_to_zero + 1, device=t.device
    )
    mask[:, 0] = 0
    t[mask] = 0
    return t


class TransformTensorDataset(torch.utils.data.Dataset):
    def __init__(self, *tensors, transforms=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transforms = transforms
        if self.transforms is None:
            raise ValueError("Must specify transforms.")
        if len(self.transforms) != len(self.tensors):
            raise ValueError(
                "Must specify corresponding transform per tensor."
            )

    def __getitem__(self, index):
        return tuple(
            self.transforms[i](tensor[index])
            for i, tensor in enumerate(self.tensors)
        )

    def __len__(self):
        return self.tensors[0].size(0)


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_data,
        valid_data,
        max_lr=0.01,
        patience=5,
        epochs=50,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_data = train_data
        self.valid_data = valid_data
        self.patience = patience

        self.train_loss = []
        self.train_acc = []
        self.valid_loss = []
        self.valid_acc = []
        self.valid_step = []

        self.batch_size = 4_000

        *valid_tensors, valid_debuginfo = self._prepare_debug_tensors(
            self.valid_data
        )
        self.valid_loader = self._make_dataloader(valid_tensors)
        self.valid_1_loader = self._make_dataloader(
            valid_tensors, batch_size=1
        )
        self.valid_debuginfo = valid_debuginfo
        self.train_tensors = self._prepare_tensors(self.train_data)

        # Compute the steps per epoch and total epochs for the scheduler.
        self.steps_per_epoch = len(self.make_train_dataloader())
        self.epochs = epochs

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
        )

        self.best_valid_loss = float("inf")
        self.patience_counter = 0

    def _prepare_tensors(self, data):
        xs = self.prepare_x(data)
        xs = tuple(x.to(self.device) for x in xs)
        y = self.prepare_y(data)
        y = y.to(self.device)
        return *xs, y

    def _prepare_debug_tensors(self, data):
        *xs, y = self._prepare_tensors(data)
        debuginfo = self.prepare_debuginfo(data)
        return *xs, y, debuginfo

    def make_train_dataloader(self):
        return self._make_dataloader(
            self.train_tensors,
            shuffle=True,
        )

    def _make_dataloader(
        self, tensors, transform=None, batch_size=None, **kwargs
    ):
        if transform is not None:
            tensors = (tensors[0], transform(tensors[1]), tensors[2])

        if batch_size is None:
            batch_size = self.batch_size

        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*tensors),
            batch_size=batch_size,
            **kwargs,
        )

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_acc = 0
            for *x, y in self.valid_loader:
                loss, acc = self.eval(x, y)
                total_loss += float(loss)
                total_acc += float(acc)
            self.valid_loss.append(total_loss / len(self.valid_loader))
            self.valid_acc.append(total_acc / len(self.valid_loader))
        self.valid_step.append(len(self.train_loss))

        if self.valid_loss[-1] < self.best_valid_loss:
            self.best_valid_loss = self.valid_loss[-1]
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                raise EarlyStoppingException(
                    "Early stopping due to validation loss not improving"
                )

    def print_worst_validation_examples(self):
        self.model.eval()
        with torch.no_grad():
            losses = []
            examples = []

            for (*x, y), debuginfo in tqdm(
                list(zip(self.valid_1_loader, self.valid_debuginfo))
            ):
                loss, acc = self.eval(x, y)
                uuid = self.model.uuids_map.int_to_item(int(y))
                # examples.append(uuid)
                examples.append(debuginfo)
                losses.append(float(loss))

        print(f"Num validation examples: {len(losses):,}")
        print("Min loss:", min(losses))
        print("Max loss:", max(losses))
        print("Mean loss:", sum(losses) / len(losses))

        uuid_losses = collections.defaultdict(list)
        for loss, uuid in zip(losses, examples):
            uuid_losses[uuid].append(loss)

        mean_uuid = sorted(
            (sum(losses) / len(losses), uuid)
            for uuid, losses in uuid_losses.items()
        )

        for mean, uuid in mean_uuid:
            print(uuid, mean)

        # worst_indices = sorted(
        #     range(len(losses)), key=lambda i: losses[i], reverse=True
        # )

        # worst_examples = [(examples[i], losses[i]) for i in worst_indices]

        # for example, loss in worst_examples:
        #     print(f"Loss: {loss}, Data: {example}")

    def train(self, num_iter=1):
        self.model.train()
        for _ in range(num_iter):
            for *x, y in self.make_train_dataloader():
                self.optimizer.zero_grad()
                loss, acc = self.eval(x, y)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.train_loss.append(float(loss))
                self.train_acc.append(float(acc))

    def prepare_x(self, dataset):
        return self.model.prepare_batch(dataset)

    def prepare_y(self, dataset):
        y_actual = []
        for item in dataset:
            y_actual.extend(part_uuids_to_indices(self.model, item))
        t_y_actual = torch.tensor(y_actual, requires_grad=False)
        return t_y_actual

    def prepare_debuginfo(self, dataset):
        z = []
        for partname, uuid_pos_list, path in dataset:
            for _ in uuid_pos_list:
                z.append(path)
        return z

    def eval(self, x, y_actual):
        logits_model = self.model(x)
        loss = self.criterion(logits_model, y_actual)
        _, y_model = torch.max(logits_model, dim=1)
        total_correct = (torch.eq(y_model, y_actual)).sum()
        total_moles = y_actual.shape[0]
        return loss, total_correct / total_moles

    def plot(self):
        train_df = pd.DataFrame(
            {"train loss": self.train_loss, "train accuracy": self.train_acc}
        )
        valid_df = pd.DataFrame(
            {"valid loss": self.valid_loss, "valid accuracy": self.valid_acc},
            index=self.valid_step,
        )
        ax = train_df.plot(y="train loss")
        ax = valid_df.plot(y="valid loss", ax=ax)
        ax = train_df.plot(y="train accuracy", secondary_y=True, ax=ax)
        ax = valid_df.plot(y="valid accuracy", secondary_y=True, ax=ax)


# -----------------------------------------------------------------------------
# Copyright (C) 2023 Angelos Evripiotis.
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
