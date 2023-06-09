"""Identify moles from their positions in images."""


import torch


def mole_neighbours_from_uuid_points(uuid_points, num_neighbours):
    """Given a list of (uuid, pos_xy) tuples, determine the nearest uuids.

    Returns a list of lists containing the uuids of the nearest neighbors for
    each tuple.
    """

    if not uuid_points:
        return []

    actual_num_neighbours = min(num_neighbours, len(uuid_points) - 1)

    positions_tensor = torch.tensor([pos_xy for _, pos_xy in uuid_points])
    distances = torch.cdist(positions_tensor, positions_tensor)
    _, indices = torch.topk(
        distances, actual_num_neighbours + 1, largest=False, sorted=True
    )

    # Exclude each point from it's own nearest neighbors list.
    indices = indices[:, 1:]

    # Map the indices back to uuids
    uuids = [uuid for uuid, _ in uuid_points]
    nearest_neighbors = [[uuids[i] for i in row] for row in indices]

    padding = num_neighbours - actual_num_neighbours
    if padding:
        nearest_neighbors = [
            uuids + ([None] * padding) for uuids in nearest_neighbors
        ]

    return nearest_neighbors
