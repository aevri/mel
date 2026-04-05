"""Work with a collection of moles."""

import argparse
import json
import math
import pathlib
import uuid

import numpy as np

import mel.lib.fs
import mel.lib.image
import mel.lib.math
import mel.rotomap.mask

KEY_IS_CONFIRMED = "is_uuid_canonical"
KEY_IS_UNCHANGED = "is_unchanged"

IGNORE_NEW_FILENAME = "ignore-new"
IGNORE_MISSING_FILENAME = "ignore-missing"
IGNORE_MISSING_CARRYFORWARD_FILENAME = "ignore-missing-carryforward"
IGNORE_CHANGED_FILENAME = "ignore-changed"
ROTOMAP_DIR_LESIONS_FILENAME = "lesions.json"


class RotomapDirectory:
    """RotomapFrame-s for all images in a single rotomap dir."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        if not self.path.is_dir():
            msg = f'"{self.path}" is not a directory, so not a rotomap.'
            raise ValueError(msg)

        self.image_paths = [
            str(f) for f in self.path.iterdir() if mel.lib.fs.is_jpeg_name(f)
        ]
        self.image_paths.sort()

        self.lesions = load_rotomap_dir_lesions_file(self.path)

        if not self.image_paths:
            msg = f'"{self.path}" has no images, so not a rotomap.'
            raise ValueError(msg)

    def yield_mole_lists(self):
        """Yield (image_path, mole_list) for all mole image files."""
        for imagepath in self.image_paths:
            yield imagepath, load_image_moles(imagepath)

    def yield_frames(self, *, extra_stem=None):
        for imagepath in self.image_paths:
            yield RotomapFrame(imagepath, extra_stem=extra_stem)

    def calc_uuids(self):
        return {
            uuid_ for frame in self.yield_frames() for uuid_ in frame.moledata.uuids
        }

    def __repr__(self):
        """Return a string representation of the rotomap directory."""
        return f"RotomapDirectory({self.path!r})"


class RotomapFrame:
    """Image and mole data for a single image in a rotomap."""

    def __init__(self, path, *, extra_stem=None):
        self.path = pathlib.Path(path)
        self.extra_stem = extra_stem
        if self.path.is_dir():
            msg = f"Expected file, not directory: {path}"
            raise ValueError(msg)
        if not self.path.exists():
            msg = f"Path does not exist: {path}"
            raise ValueError(msg)
        if not mel.lib.fs.is_jpeg_name(self.path):
            msg = f"Unrecognised suffix for rotomap frame: {path}"
            raise ValueError(msg)

        self.moles = load_image_moles(self.path, extra_stem=extra_stem)
        self.moledata = MoleData(self.moles)
        self.metadata = load_image_metadata(self.path)

    def load_image(self):
        return mel.lib.image.load_image(self.path)

    def has_mole_file(self):
        if self.extra_stem is None:
            return pathlib.Path(f"{self.path}.json").exists()
        return pathlib.Path(f"{self.path}.{self.extra_stem}.json").exists()

    def has_mask(self):
        return mel.rotomap.mask.has_mask(self.path)

    def __repr__(self):
        """Return a string representation of the rotomap frame."""
        return f"RotomapFrame({self.path!r})"


class MoleData:
    """Iterables of UUIDs, locations, and other data on moles in an image."""

    def __init__(self, mole_iter):
        self.moles = tuple(mole_iter)
        self.uuids = frozenset(m["uuid"] for m in self.moles)
        self.uuid_points = to_uuid_points(self.moles)
        self.uuid_points_list = [(m["uuid"], mole_to_point(m)) for m in self.moles]

        # vulture will report this as unused unless we do this
        #
        # pylint: disable=pointless-statement
        _ = self.uuid_points_list
        # pylint: enable=pointless-statement


def make_argparse_rotomap_directory(path):
    """Use in the 'type=' parameter to add_argument()."""
    try:
        return RotomapDirectory(path)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def make_argparse_image_moles(path):
    """Use in the 'type=' parameter to add_argument()."""
    try:
        path = pathlib.Path(path)
        exists = path.exists()
    except (TypeError, ValueError, OSError) as e:
        raise argparse.ArgumentTypeError(str(e)) from e
    if not exists:
        msg = f"'{path}' does not exist."
        raise argparse.ArgumentTypeError(msg)
    try:
        if path.is_file():
            yield path, load_image_moles(path)
        else:
            yield from RotomapDirectory(path).yield_mole_lists()
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def make_argparse_image_moles_tree(path):
    """Use in the 'type=' parameter to add_argument()."""
    path = pathlib.Path(path)
    if path.is_dir():
        for item in sorted(path.iterdir()):
            if item.is_dir():
                yield from make_argparse_image_moles_tree(item)
            elif mel.lib.fs.is_jpeg_name(item):
                yield from make_argparse_image_moles(item)
    else:
        yield from make_argparse_image_moles(path)


class MoleListDiff:
    def __init__(self, old_uuids, new_uuids, ignore_new, ignore_missing):
        self.new = (new_uuids - old_uuids) - ignore_new
        self.missing = (old_uuids - new_uuids) - ignore_missing


def normalised_ellipse_mask(ellipse):
    """Return a normalized copy of the supplied ellipse.

    Here 'normalised' means that the rotation is as close to zero as possible.

    Examples:
        >>> normalised_ellipse_mask(
        ...     ((1, 2), (100, 200), 90)
        ... )
        ((1, 2), (200, 100), 0)

    """
    # Don't overwrite the original, we'll return a new ellipse.
    centre, extents, rotation = ellipse
    centre = list(centre[:])
    extents = list(extents[:])

    # Get the rotation as close to zero as possible.
    while rotation > 45:
        extents[0], extents[1] = extents[1], extents[0]
        rotation -= 90
    while rotation < -45:
        extents[0], extents[1] = extents[1], extents[0]
        rotation += 90

    return tuple(centre), tuple(extents), rotation


def validate_ellipse_mask(ellipse, max_x=10000, max_y=10000):
    max_length = max(max_x, max_y) * 2

    if ellipse[1][0] < 1 or ellipse[1][1] < 1:
        msg = f"Ellipse too small: {ellipse}"
        raise ValueError(msg)
    if ellipse[1][0] > max_length or ellipse[1][1] > max_length:
        msg = f"Ellipse too big: {ellipse}"
        raise ValueError(msg)
    if ellipse[0][0] < 0 or ellipse[0][1] < 0:
        msg = f"Ellipse out of bounds: {ellipse}"
        raise ValueError(msg)
    if ellipse[0][0] > max_x or ellipse[0][1] > max_y:
        msg = f"Ellipse out of bounds: {ellipse}"
        raise ValueError(msg)


def load_image_metadata(image_path):
    metadata_path = pathlib.Path(str(image_path) + ".meta.json")

    metadata = {}
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        if "ellipse" in metadata:
            try:
                validate_ellipse_mask(metadata["ellipse"])
            except ValueError as e:
                msg = f"Bad data from '{metadata_path}'."
                raise ValueError(msg) from e

    return metadata


def load_rotomap_dir_lesions_file(rotomap_dir_path):
    rotomap_dir_path = pathlib.Path(rotomap_dir_path)
    if not rotomap_dir_path.exists():
        msg = f"Rotomap directory does not exist: '{rotomap_dir_path}'."
        raise ValueError(msg)

    lesions_path = rotomap_dir_path / ROTOMAP_DIR_LESIONS_FILENAME

    lesions = []
    if lesions_path.exists():
        lesions = load_json(lesions_path)

    for m in lesions:
        if KEY_IS_UNCHANGED not in m:
            msg = f"Mole must have {KEY_IS_UNCHANGED} status: {lesions_path} {m}"
            raise ValueError(msg)

    for m in lesions:
        if m["uuid"] is None:
            msg = f"Lesion UUID cannot be None: {lesions_path} {m}"
            raise ValueError(msg)

    return lesions


def save_rotomap_dir_lesions_file(rotomap_dir_path, lesions):
    rotomap_dir_path = pathlib.Path(rotomap_dir_path)
    if not rotomap_dir_path.exists():
        msg = f"Rotomap directory does not exist: '{rotomap_dir_path}'."
        raise ValueError(msg)

    lesions_path = rotomap_dir_path / ROTOMAP_DIR_LESIONS_FILENAME
    save_json(lesions_path, lesions)


def load_image_moles(image_path, *, extra_stem=None):
    if not pathlib.Path(image_path).exists():
        msg = f"Mole image does not exist: '{image_path}'."
        raise ValueError(msg)

    suffix = ".json"
    if extra_stem is not None:
        suffix = f".{extra_stem}.json"
    moles_path = pathlib.Path(str(image_path) + suffix)

    moles = []
    if moles_path.exists():
        moles = load_json(moles_path)

    for m in moles:
        if KEY_IS_CONFIRMED not in m:
            msg = f"Mole must have {KEY_IS_CONFIRMED} status: {moles_path} {m}"
            raise ValueError(msg)

    for m in moles:
        m["x"] = int(m["x"])
        m["y"] = int(m["y"])

    for m in moles:
        if m["uuid"] is None:
            msg = f"Mole UUID cannot be None: {moles_path} {m}"
            raise ValueError(msg)

    return moles


def normalise_moles(moles):
    for m in moles:
        m["x"] = int(m["x"])
        m["y"] = int(m["y"])


def save_image_metadata(metadata, image_path):
    meta_path = image_path + ".meta.json"
    save_json(meta_path, metadata)


def save_image_moles(moles, image_path, *, extra_stem=None):
    # Explicitly convert 'image_path' to str. It might be a pathlib.Path, which
    # doesn't support addition in this way.
    moles_path = str(image_path)
    if extra_stem:
        moles_path += f".{extra_stem}"
    moles_path += ".json"
    save_json(moles_path, moles)


def load_json(path):
    with pathlib.Path(path).open() as f:
        return json.load(f)


def save_json(path, data):
    with pathlib.Path(path).open("w") as f:
        json.dump(data, f, indent=4, separators=(",", ": "), sort_keys=True)

        # There's no newline after dump(), add one here for happier viewing
        print(file=f)


def make_new_uuid():
    return uuid.uuid4().hex


def add_mole(moles, x, y, mole_uuid=None):
    is_uuid_canonical = True
    if mole_uuid is None:
        mole_uuid = make_new_uuid()
        is_uuid_canonical = False

    moles.append(
        {
            "x": x,
            "y": y,
            "uuid": mole_uuid,
            KEY_IS_CONFIRMED: is_uuid_canonical,
        }
    )


def nearest_mole_index(moles, x, y):
    return nearest_mole_index_distance(moles, x, y)[0]


def nearest_mole_index_distance(moles, x, y):
    nearest_index = None
    nearest_distance = None
    for i, mole in enumerate(moles):
        dx = x - mole["x"]
        dy = y - mole["y"]
        distance = math.sqrt(dx * dx + dy * dy)
        if nearest_distance is None or distance < nearest_distance:
            nearest_index = i
            nearest_distance = distance

    return nearest_index, nearest_distance


def uuid_mole_index(moles, mole_uuid):
    """Return the index of the first mole with the specified uuid."""
    for i, mole in enumerate(moles):
        if mole["uuid"] == mole_uuid:
            return i
    return None


def set_nearest_mole_uuid(moles, x, y, mole_uuid, *, is_canonical=True):
    nearest_index = nearest_mole_index(moles, x, y)
    if nearest_index is not None:
        moles[nearest_index]["uuid"] = mole_uuid
        moles[nearest_index][KEY_IS_CONFIRMED] = is_canonical


def get_nearest_mole_uuid(moles, x, y):
    nearest_index = nearest_mole_index(moles, x, y)
    if nearest_index is not None:
        return moles[nearest_index]["uuid"]

    return None


def move_nearest_mole(moles, x, y):
    nearest_index = nearest_mole_index(moles, x, y)

    if nearest_index is not None:
        moles[nearest_index]["x"] = x
        moles[nearest_index]["y"] = y


def remove_nearest_mole(moles, x, y):
    nearest_index = nearest_mole_index(moles, x, y)

    if nearest_index is not None:
        del moles[nearest_index]


def mole_list_to_pointvec(mole_list):
    return np.array(tuple((m["x"], m["y"]) for m in mole_list))


def mole_to_point(mole):
    pos = np.array((mole["x"], mole["y"]))
    mel.lib.math.raise_if_not_int_vector2(pos)
    return pos


def to_uuid_points(moles):
    uuid_points = {}
    for m in moles:
        uuid_points[m["uuid"]] = mole_to_point(m)
    return uuid_points


def load_potential_set_file(path, filename):
    ignore_set = set()
    file_path = path / filename
    if file_path.is_file():
        with file_path.open() as f:
            lines = f.read().splitlines()
        for text in lines:
            stripped_text = text.strip()
            if stripped_text and not stripped_text.startswith("#"):
                ignore_set.add(stripped_text)
    return ignore_set


# -----------------------------------------------------------------------------
# Copyright (C) 2016-2019, 2026 Angelos Evripiotis.
# Generated with assistance from Claude Code.
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
