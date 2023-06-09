"""Create datasets in memory from rotomaps on disk."""

import collections
import pathlib

import mel.rotomap.moles


def empty_pathdict():
    """Return an empty Dict of 'part' -> 'subpart' -> list-of-rotomap-paths."""
    return collections.defaultdict(lambda: collections.defaultdict(list))


def make_pathdict(repo_path: pathlib.Path):
    """Return a Dict of 'part' -> 'subpart' -> list-of-rotomap-paths.

    This is a useful for making datasets because we generally want to collect
    all the subparts and split the dataset by the rotomap date.

    """
    parts_path = repo_path / "rotomaps" / "parts"
    result = empty_pathdict()
    for part in parts_path.iterdir():
        for subpart in part.iterdir():
            subpart_paths = sorted(p for p in subpart.iterdir())
            for p in subpart_paths:
                result[f"{part.stem}"][f"{subpart.stem}"].append(p)
    return result


def drop_empty_paths(pathdict):
    """Drop empties from a Dict of 'part' -> 'subpart' -> list-of-rotomap-paths.

    A rotomap is empty if it has no ellipse metadata.

    Perhaps it has not been processed yet.

    """
    result = empty_pathdict()
    for part, subpart_rotomaps in pathdict.items():
        for subpart, rotomaps in subpart_rotomaps.items():
            for path in rotomaps:
                rdir = mel.rotomap.moles.RotomapDirectory(path)
                if all(
                    ("ellipse" not in frame.metadata)
                    for frame in rdir.yield_frames()
                ):
                    continue
                result[part][subpart].append(path)
    return result


def split_train_valid_last(pathdict):
    """Return (train, valid) from a given pathdict, valid has last map in each.

    Note that a pathdict is as returned by 'make_pathdict'.

    """
    train = empty_pathdict()
    valid = empty_pathdict()
    for part, subpart_rotomaps in pathdict.items():
        for subpart, rotomaps in subpart_rotomaps.items():
            train[part][subpart].extend(rotomaps[:-1])
            valid[part][subpart].append(rotomaps[-1])
    return train, valid


def listify_pathdict(pathdict):
    return [path for subpart in pathdict.values() for path in subpart.values()]
