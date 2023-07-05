"""Identify moles from their positions in images."""

import json
import pathlib

import pandas as pd
import torch

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

        x1, x2 = self.model.prepare_batch([(pathname, uuid_points)])

        self.model.eval()
        with torch.no_grad():
            logits = self.model((x1, x2))
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

    points = [p for _, p in uuid_points]
    padding = num_neighbours - actual_num_neighbours
    x = [
        make_mole_row(points, row_indices, distances[i], padding)
        for i, row_indices in enumerate(indices)
    ]

    return x


def make_mole_row(points, indices, distances, padding):
    i = indices[0]
    self_item = (i, points[i], distances[i])
    self_point = points[i]
    neighbours = [
        (i, points[i] - self_point, distances[i]) for i in indices[1:]
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
    part_name, mole_data = data
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

    def __len__(self):
        return len(self._item_to_int)


class PosEncoder(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.encoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2),
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


class SelfposOnly(torch.nn.Module):
    def __init__(self, partnames_uuids):
        super().__init__()
        self.width = 16
        self.selfpos_encoder = PosEncoder(self.width)
        all_partnames = list(partnames_uuids.keys())
        all_uuids = [
            uuid for uuids in partnames_uuids.values() for uuid in uuids
        ]

        self.partnames_map = IndexMap(all_partnames)
        self.uuids_map = IndexMap(all_uuids)

        self.partnames_embedding = torch.nn.Embedding(
            len(self.partnames_map), self.width
        )
        self.classifier = torch.nn.Linear(self.width * 2, len(self.uuids_map))

    def prepare_batch(self, batch):
        batch = [
            (
                item[0],
                mole_data_from_uuid_points(item[1], num_neighbours=3),
            )
            for item in batch
        ]

        # TODO: allow moles with 'None' uuid, to be non-moles.
        partname_indices = []
        pos_values = []

        for x in batch:
            part_name, mole_list = x
            num_moles = len(mole_list)
            partname_indices.extend(
                [self.partnames_map.item_to_int(part_name)] * num_moles
            )
            pos_values.extend([mole[0][1] for mole in mole_list])

        partname_indices = torch.tensor(
            partname_indices, dtype=torch.long, requires_grad=False
        )
        pos_values = torch.tensor(
            pos_values, dtype=torch.float32, requires_grad=False
        )

        return partname_indices, pos_values

    def forward(self, batch):
        partname_indices, pos_values = batch

        partname_embedding = self.partnames_embedding(partname_indices)
        pos_emb = self.selfpos_encoder(pos_values)

        emb = torch.cat([pos_emb, partname_embedding], dim=-1)
        return self.classifier(emb)


class PosOnlyLinear(torch.nn.Module):
    def __init__(self, partnames_uuids, num_neighbours):
        super().__init__()
        self.num_neighbours = num_neighbours
        self.width = 16

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
        self.classifier = torch.nn.Linear(
            self.width * (2 + self.num_neighbours), len(self.uuids_map)
        )

    def prepare_batch(self, batch):
        batch = [
            (
                item[0],
                mole_data_from_uuid_points(
                    item[1], num_neighbours=self.num_neighbours
                ),
            )
            for item in batch
        ]

        # TODO: allow moles with 'None' uuid, to be non-moles.
        partname_indices = []
        pos_values = []

        def convert_none_pos(pos):
            if pos is None:
                return 0.0, 0.0
            x, y = pos
            return float(x), float(y)

        for x in batch:
            part_name, mole_list = x
            num_moles = len(mole_list)
            partname_indices.extend(
                [self.partnames_map.item_to_int(part_name)] * num_moles
            )
            pos_values.extend(
                [[convert_none_pos(m[1]) for m in mole] for mole in mole_list]
            )

        partname_indices = torch.tensor(
            partname_indices, dtype=torch.long, requires_grad=False
        )
        pos_values = torch.tensor(
            pos_values, dtype=torch.float32, requires_grad=False
        )

        return partname_indices, pos_values

    def forward(self, batch):
        partname_indices, pos_values = batch

        partname_embedding = self.partnames_embedding(partname_indices)
        selfpos_emb = self.selfpos_encoder(pos_values[:, 0])

        relpos_embs = []
        for i in range(1, self.num_neighbours + 1):
            relpos_emb = self.relpos_encoder(pos_values[:, i])
            relpos_embs.append(relpos_emb)

        emb = torch.cat(
            [selfpos_emb] + relpos_embs + [partname_embedding], dim=-1
        )

        return self.classifier(emb)


class PosOnly(torch.nn.Module):
    def __init__(self, partnames_uuids, num_neighbours):
        super().__init__()
        self.num_neighbours = num_neighbours
        self.width = 16
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
        self.transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.width, nhead=8, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(
            self.transformer_layer, num_layers=1
        )
        self.classifier = torch.nn.Linear(
            self.width * (2 + self.num_neighbours), len(self.uuids_map)
        )

    def prepare_batch(self, batch):
        batch = [
            (
                item[0],
                mole_data_from_uuid_points(
                    item[1], num_neighbours=self.num_neighbours
                ),
            )
            for item in batch
        ]

        # TODO: allow moles with 'None' uuid, to be non-moles.
        partname_indices = []
        pos_values = []

        def convert_none_pos(pos):
            if pos is None:
                return 0.0, 0.0
            x, y = pos
            return float(x), float(y)

        for x in batch:
            part_name, mole_list = x
            num_moles = len(mole_list)
            partname_indices.extend(
                [self.partnames_map.item_to_int(part_name)] * num_moles
            )
            pos_values.extend(
                [[convert_none_pos(m[1]) for m in mole] for mole in mole_list]
            )

        partname_indices = torch.tensor(
            partname_indices, dtype=torch.long, requires_grad=False
        )
        pos_values = torch.tensor(
            pos_values, dtype=torch.float32, requires_grad=False
        )

        return partname_indices, pos_values

    def forward(self, batch):
        partname_indices, pos_values = batch

        partname_embedding = self.partnames_embedding(partname_indices)
        selfpos_emb = self.selfpos_encoder(pos_values[:, 0])

        relpos_embs = []
        for i in range(1, self.num_neighbours + 1):
            relpos_emb = self.relpos_encoder(pos_values[:, i])
            relpos_embs.append(relpos_emb)

        emb_sequence = torch.stack([selfpos_emb] + relpos_embs).transpose(
            0, 1
        )  # Shape: (batch_size, sequence_length, embed_dim)

        transformer_output = self.transformer(
            emb_sequence
        )  # shape: (batch_size, sequence_length, embed_dim)

        transformer_output_flat = transformer_output.view(
            transformer_output.size(0), -1
        )  # shape: (batch_size, sequence_length * embed_dim)

        emb = torch.cat([transformer_output_flat, partname_embedding], dim=-1)

        return self.classifier(emb)


class EarlyStoppingException(Exception):
    pass


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

        self.batch_size = 2_000

        self.valid_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                *self.prepare_x(self.valid_data),
                self.prepare_y(self.valid_data).to(self.device),
            ),
            batch_size=self.batch_size,
        )
        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                *self.prepare_x(self.train_data),
                self.prepare_y(self.train_data).to(self.device),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Compute the steps per epoch and total epochs for the scheduler.
        self.steps_per_epoch = len(self.train_loader)
        self.epochs = epochs

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
        )

        self.best_valid_loss = float("inf")
        self.patience_counter = 0

    def validate(self):
        with torch.no_grad():
            total_loss = 0
            total_acc = 0
            for x1, x2, y in self.valid_loader:
                x = (x1, x2)
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

    def train(self, num_iter=1):
        for _ in range(num_iter):
            for x1, x2, y in self.train_loader:
                self.optimizer.zero_grad()
                x = (x1, x2)
                loss, acc = self.eval(x, y)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.train_loss.append(float(loss))
                self.train_acc.append(float(acc))

    def prepare_x(self, dataset):
        x1, x2 = self.model.prepare_batch(dataset)
        return x1.to(self.device), x2.to(self.device)

    def prepare_y(self, dataset):
        y_actual = []
        for item in dataset:
            y_actual.extend(part_uuids_to_indices(self.model, item))
        t_y_actual = torch.tensor(y_actual, requires_grad=False)
        return t_y_actual

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
