"""Identify moles from their positions in images."""

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
        neighbours += [(None, None, None, None) for _ in range(padding)]
    return [self_item] + neighbours


class Model(torch.nn.Module):
    def __init__(self, partnames_uuids):
        super().__init__()
        self.selfpos_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 8, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8, bias=True),
            torch.nn.ReLU(),
        )
        self.partnames_uuidmap = {
            partname: {u: i for i, u in enumerate(sorted(uuids))}
            for partname, uuids in partnames_uuids.items()
        }
        self.partnames_classifiers = {
            partname: torch.nn.Sequential(
                torch.nn.Linear(8, len(uuids) + 1, bias=True),
                torch.nn.Softmax(),
            )
            for partname, uuids in partnames_uuids.items()
        }

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

        # Minimal test version: just a linear layers looking at the self_pos.
        return x


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
