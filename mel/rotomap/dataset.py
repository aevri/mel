"""Create datasets in memory from rotomaps on disk."""

import collections
import pathlib

import mel.lib.ellipsespace
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
    return [
        path
        for subpart_maps in pathdict.values()
        for maps in subpart_maps.values()
        for path in maps
    ]


def yield_imagemoles_from_pathlist(pathlist):
    """Return a list, with one entry per image. Each entry is a list of moles.

    Each entry in the resulting list is a self-contained example.

    The mole positions are normalized to their position in ellipse space.

    """
    for rotomap_path in pathlist:
        rdir = mel.rotomap.moles.RotomapDirectory(rotomap_path)
        for frame in rdir.yield_frames():
            uuid_points = list(frame.moledata.uuid_points_list)
            ellipse = frame.metadata["ellipse"]
            elspace = mel.lib.ellipsespace.Transform(ellipse)
            uuid_points = [
                (uuid, elspace.to_space(point)) for uuid, point in uuid_points
            ]
            if uuid_points:
                yield uuid_points


def make_partnames_uuids(pathdict):
    result = collections.defaultdict(set)
    for part, subpart_rotomaps in pathdict.items():
        for subpart, rotomaps in subpart_rotomaps.items():
            partname = f"{part}/{subpart}"
            for path in rotomaps:
                rdir = mel.rotomap.moles.RotomapDirectory(path)
                for frame in rdir.yield_frames():
                    for mole in frame.moles:
                        if mole[mel.rotomap.moles.KEY_IS_CONFIRMED]:
                            result[partname].add(mole["uuid"])

    return {
        partname: list(sorted(uuids)) for partname, uuids in result.items()
    }
