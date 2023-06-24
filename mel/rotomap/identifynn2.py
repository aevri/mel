"""Identify moles from their positions in images."""

import pandas as pd
import torch


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


def make_partname_uuidmap(partnames_uuids):
    return {
        partname: UuidMap(uuids) for partname, uuids in partnames_uuids.items()
    }


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
    return model.partnames_uuidmap[part_name].uuidlist_to_ints(uuids)


class RandomChooser(torch.nn.Module):
    def __init__(self, partnames_uuids):
        super().__init__()
        self.partnames_uuidmap = make_partname_uuidmap(partnames_uuids)

    def forward(self, x):
        part_name, mole_data = x
        uuidmap = self.partnames_uuidmap[part_name]
        num_possible_uuids = len(uuidmap)
        dist = torch.distributions.one_hot_categorical.OneHotCategorical(
            torch.ones(num_possible_uuids) / num_possible_uuids
        )
        return dist.sample([len(mole_data)])


class ResBlock(torch.nn.Module):
    def __init__(self, submodule):
        super().__init__()
        self.submodule = submodule

    def forward(self, x):
        return self.submodule(x) + x


class SelfposOnly(torch.nn.Module):
    def __init__(self, partnames_uuids):
        super().__init__()
        self.width = 16
        self.selfpos_encoder = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(2),
            torch.nn.Linear(2, self.width, bias=True),
            torch.nn.ReLU(),
            ResBlock(
                torch.nn.Sequential(
                    # torch.nn.BatchNorm1d(self.width),
                    torch.nn.Linear(self.width, self.width, bias=True),
                    torch.nn.ReLU(),
                    # torch.nn.BatchNorm1d(self.width),
                    torch.nn.Linear(self.width, self.width, bias=True),
                    torch.nn.ReLU(),
                )
            ),
        )
        self.partnames_uuidmap = make_partname_uuidmap(partnames_uuids)
        self.partnames_classifiers = {
            partname: torch.nn.Sequential(
                torch.nn.Linear(self.width, len(uuids) + 1, bias=True),
            )
            for partname, uuids in partnames_uuids.items()
        }

    def forward(self, x):
        part_name, mole_list = x
        classifier = self.partnames_classifiers[part_name]
        pos = torch.tensor(
            [mole[0][1] for mole in mole_list], dtype=torch.float32
        )
        emb = self.selfpos_encoder(pos)
        return classifier(emb)


class SinglePass(torch.nn.Module):
    def __init__(self, partnames_uuids):
        super().__init__()
        self.width = 16
        # TODO: consider https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html
        self.selfpos_encoder = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(2),
            torch.nn.Linear(2, self.width, bias=True),
            torch.nn.ReLU(),
            ResBlock(
                torch.nn.Sequential(
                    # torch.nn.BatchNorm1d(self.width),
                    torch.nn.Linear(self.width, self.width, bias=True),
                    torch.nn.ReLU(),
                    # torch.nn.BatchNorm1d(self.width),
                    torch.nn.Linear(self.width, self.width, bias=True),
                    torch.nn.ReLU(),
                )
            ),
        )
        self.relpos_encoder = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(2),
            torch.nn.Linear(2, self.width, bias=True),
            torch.nn.ReLU(),
            ResBlock(
                torch.nn.Sequential(
                    # torch.nn.BatchNorm1d(self.width),
                    torch.nn.Linear(self.width, self.width, bias=True),
                    torch.nn.ReLU(),
                    # torch.nn.BatchNorm1d(self.width),
                    torch.nn.Linear(self.width, self.width, bias=True),
                    torch.nn.ReLU(),
                )
            ),
        )
        self.partnames_uuidmap = make_partname_uuidmap(partnames_uuids)
        self.partnames_classifiers = {
            partname: torch.nn.Sequential(
                torch.nn.Linear(self.width, len(uuids) + 1, bias=True),
            )
            for partname, uuids in partnames_uuids.items()
        }

    def forward(self, x):
        part_name, mole_list = x
        classifier = self.partnames_classifiers[part_name]
        pos = torch.tensor(
            [mole[0][1] for mole in mole_list], dtype=torch.float32
        )
        emb = self.selfpos_encoder(pos)
        return classifier(emb)


class Model(torch.nn.Module):
    def __init__(self, partnames_uuids):
        super().__init__()
        pass

    def forward(self, x):
        # x: (part_name, list of moles)
        #
        # moles: my_abs_pos, [nn_rel_pos, ...]
        #
        # for each mole, process my_abs_pos, process each nn_rel_pos.
        #
        # for each mole, apply transformer to embedding list, create new short
        # embedding
        #
        # for each mole, copy appropriate short embedding, combine with
        # original position embedding. Apply transformer. Create new short
        # embedding.
        #
        # Repeat.
        #
        # Subpart-specific linear layer with softmax to classify moles. Include
        # something for 'not a mole'.
        #
        # Minimal test version: just a linear layers looking at the self_pos.
        pass


class Trainer:
    def __init__(self, model, criterion, optimizer, train_data, valid_data):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_data = train_data
        self.valid_data = valid_data

        self.train_loss = []
        self.train_acc = []
        self.valid_loss = []
        self.valid_acc = []
        self.valid_step = []

    def validate(self):
        with torch.no_grad():
            loss, acc = eval_step(
                self.model,
                self.criterion,
                self.optimizer,
                self.valid_data,
            )
        self.valid_loss.append(float(loss))
        self.valid_acc.append(float(acc))
        self.valid_step.append(len(self.train_loss))

    def train(self, num_iter=1):
        loss, acc = train_step(
            self.model,
            self.criterion,
            self.optimizer,
            self.train_data,
        )
        self.train_loss.append(float(loss))
        self.train_acc.append(float(acc))

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


def eval_step(model, criterion, optimizer, training_set):
    loss_sum = 0
    total_correct = 0
    total_moles = 0
    for item in training_set:
        x = (
            item[0],
            mole_data_from_uuid_points(item[1], num_neighbours=3),
        )
        logits_model = model(x)
        y_actual = part_uuids_to_indices(model, item)
        t_y_actual = torch.tensor(y_actual)
        loss = criterion(logits_model, torch.tensor(y_actual))
        _, y_model = torch.max(logits_model, dim=1)
        total_correct += sum(torch.eq(y_model, t_y_actual))
        total_moles += len(y_actual)
        loss_sum += loss
    return loss_sum / len(training_set), total_correct / total_moles


def train_step(model, criterion, optimizer, training_set):
    optimizer.zero_grad()
    loss, acc = eval_step(model, criterion, optimizer, training_set)
    loss.backward()
    optimizer.step()
    return loss.detach(), acc


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
