"""Work with a collection of moles."""


import argparse
import collections
import json
import math
import pathlib
import uuid

import numpy

import mel.lib.image
import mel.lib.math

import mel.rotomap.mask


KEY_IS_CONFIRMED = 'is_uuid_canonical'

IGNORE_NEW_FILENAME = 'ignore-new'
IGNORE_MISSING_FILENAME = 'ignore-missing'


class RotomapDirectory:
    """RotomapFrame-s for all images in a single rotomap dir."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        if not self.path.is_dir():
            raise ValueError(
                '"{}" is not a directory, so not a rotomap.'.format(self.path)
            )

        self.image_paths = [
            str(f) for f in self.path.iterdir()
            if f.suffix.lower() == '.jpg'
        ]
        self.image_paths.sort()

        if not self.image_paths:
            raise ValueError(
                '"{}" has no images, so not a rotomap.'.format(self.path)
            )

    def yield_mole_lists(self):
        """Yield (image_path, mole_list) for all mole image files."""
        for imagepath in self.image_paths:
            yield imagepath, load_image_moles(imagepath)

    def yield_frames(self):
        for imagepath in self.image_paths:
            yield RotomapFrame(imagepath)

    def __repr__(self):
        return f'RotomapDirectory({self.path!r})'


class RotomapFrame:
    """Image and mole data for a single image in a rotomap."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        if self.path.is_dir():
            raise ValueError(f'Expected file, not directory: {path}')
        if not self.path.exists():
            raise ValueError(f'Path does not exist: {path}')
        if self.path.suffix.lower() != '.jpg':
            raise ValueError(f'Unrecognised suffix for rotomap frame: {path}')

        self.moles = load_image_moles(self.path)
        self.moledata = MoleData(self.moles)
        self.metadata = load_image_metadata(self.path)

    def load_image(self):
        return mel.lib.image.load_image(self.path)

    def load_mask(self):
        return mel.rotomap.mask.load_or_none(self.path)

    def has_mole_file(self):
        return pathlib.Path(str(self.path) + '.json').exists()

    def has_mask(self):
        return mel.rotomap.mask.has_mask(self.path)

    def __repr__(self):
        return f"RotomapFrame({self.path!r})"


class MoleData:
    """Iterables of UUIDs, locations, and other data on moles in an image."""

    def __init__(self, mole_iter):
        self.moles = tuple(mole_iter)
        self.uuids = frozenset(m['uuid'] for m in self.moles)
        self.uuid_points = to_uuid_points(self.moles)
        self.canonical_uuids = frozenset(
            m['uuid'] for m in self.moles if m[KEY_IS_CONFIRMED]
        )
        # self.uuid_moles = {m['uuid']: m for m in self.moles}


def make_argparse_rotomap_directory(path):
    """Use in the 'type=' parameter to add_argument()."""
    try:
        return RotomapDirectory(path)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def make_argparse_image_moles(path):
    """Use in the 'type=' parameter to add_argument()."""
    try:
        path = pathlib.Path(path)
        if path.is_file():
            yield path, load_image_moles(path)
        else:
            yield from RotomapDirectory(path).yield_mole_lists()
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


class MoleListDiff:
    def __init__(self, old_uuids, new_uuids, ignore_new, ignore_missing):

        self.new = (new_uuids - old_uuids) - ignore_new
        self.missing = (old_uuids - new_uuids) - ignore_missing
        self.matching = old_uuids & new_uuids

        self.ignored_new = (new_uuids - old_uuids) & ignore_new
        self.ignored_missing = (old_uuids - new_uuids) & ignore_missing
        self.would_ignore_new = ignore_new - (new_uuids - old_uuids)
        self.would_ignore_missing = ignore_missing - (old_uuids - new_uuids)


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
        raise ValueError(f"Ellipse too small: {ellipse}")
    elif ellipse[1][0] > max_length or ellipse[1][1] > max_length:
        raise ValueError(f"Ellipse too big: {ellipse}")
    elif ellipse[0][0] < 0 or ellipse[0][1] < 0:
        raise ValueError(f"Ellipse out of bounds: {ellipse}")
    elif ellipse[0][0] > max_x or ellipse[0][1] > max_y:
        raise ValueError(f"Ellipse out of bounds: {ellipse}")


def load_image_metadata(image_path):
    metadata_path = pathlib.Path(str(image_path) + '.meta.json')

    metadata = {}
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        if 'ellipse' in metadata:
            try:
                validate_ellipse_mask(metadata['ellipse'])
            except ValueError as e:
                raise ValueError(f"Bad data from '{metadata_path}'.") from e

    return metadata


def load_image_moles(image_path):
    if not pathlib.Path(image_path).exists():
        raise ValueError(f"Mole image does not exist: '{image_path}'.")

    moles_path = pathlib.Path(str(image_path) + '.json')

    moles = []
    if moles_path.exists():
        moles = load_json(moles_path)

    for m in moles:
        if KEY_IS_CONFIRMED not in m:
            m[KEY_IS_CONFIRMED] = True

    for m in moles:
        m['x'] = int(m['x'])
        m['y'] = int(m['y'])

    for m in moles:
        if m['uuid'] is None:
            raise Exception(f'Mole UUID cannot be None: {moles_path} {m}')

    return moles


def load_image_untracked(image_path):
    untracked_path = pathlib.Path(str(image_path) + '.untracked.json')

    untracked = []
    if untracked_path.exists():
        untracked = load_json(untracked_path)

    return untracked


def normalise_moles(moles):
    for m in moles:
        m['x'] = int(m['x'])
        m['y'] = int(m['y'])


def save_image_metadata(metadata, image_path):
    meta_path = image_path + '.meta.json'
    save_json(meta_path, metadata)


def save_image_moles(moles, image_path):
    # Explicitly convert 'image_path' to str. It might be a pathlib.Path, which
    # doesn't support addition in this way.
    moles_path = str(image_path) + '.json'
    save_json(moles_path, moles)


def save_image_untracked(untracked, image_path):
    # Explicitly convert 'image_path' to str. It might be a pathlib.Path, which
    # doesn't support addition in this way.
    untracked_path = str(image_path) + '.untracked.json'
    save_compact_json(untracked_path, untracked)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '), sort_keys=True)

        # There's no newline after dump(), add one here for happier viewing
        print(file=f)


def save_compact_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=0, sort_keys=True)

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
            'x': x,
            'y': y,
            'uuid': mole_uuid,
            KEY_IS_CONFIRMED: is_uuid_canonical,
        }
    )


def nearest_mole_index(moles, x, y):
    return nearest_mole_index_distance(moles, x, y)[0]


def nearest_mole_index_distance(moles, x, y):
    nearest_index = None
    nearest_distance = None
    for i, mole in enumerate(moles):
        dx = x - mole['x']
        dy = y - mole['y']
        distance = math.sqrt(dx * dx + dy * dy)
        if nearest_distance is None or distance < nearest_distance:
            nearest_index = i
            nearest_distance = distance

    return nearest_index, nearest_distance


def uuid_mole_index(moles, mole_uuid):
    """Return the index of the first mole with the specified uuid."""
    for i, mole in enumerate(moles):
        if mole['uuid'] == mole_uuid:
            return i
    return None


def set_nearest_mole_uuid(moles, x, y, mole_uuid, is_canonical=True):
    nearest_index = nearest_mole_index(moles, x, y)
    if nearest_index is not None:
        moles[nearest_index]['uuid'] = mole_uuid
        moles[nearest_index][KEY_IS_CONFIRMED] = is_canonical


def get_nearest_mole_uuid(moles, x, y):
    nearest_index = nearest_mole_index(moles, x, y)
    if nearest_index is not None:
        return moles[nearest_index]['uuid']

    return None


def move_nearest_mole(moles, x, y):
    nearest_index = nearest_mole_index(moles, x, y)

    if nearest_index is not None:
        moles[nearest_index]['x'] = x
        moles[nearest_index]['y'] = y


def remove_nearest_mole(moles, x, y):
    nearest_index = nearest_mole_index(moles, x, y)

    if nearest_index is not None:
        del moles[nearest_index]


def mole_list_to_pointvec(mole_list):
    return numpy.array(
        tuple((m['x'], m['y']) for m in mole_list)
    )


def mole_to_point(mole):
    pos = numpy.array((mole['x'], mole['y']))
    mel.lib.math.raise_if_not_int_vector2(pos)
    return pos


def to_uuid_points(moles):
    uuid_points = {}
    for m in moles:
        uuid_points[m['uuid']] = mole_to_point(m)
    return uuid_points


def frames_to_uuid_frameposlist(frame_iterable, canonical_only=False):
    uuid_to_frameposlist = collections.defaultdict(list)

    for frame in frame_iterable:

        if 'ellipse' not in frame.metadata:
            raise Exception(
                f'{frame} has no ellipse metadata, '
                'try running "rotomap calc-space"'
            )

        ellipse = frame.metadata['ellipse']
        elspace = mel.lib.ellipsespace.Transform(ellipse)
        for uuid_, pos in frame.moledata.uuid_points.items():
            if canonical_only and uuid_ not in frame.moledata.canonical_uuids:
                continue
            uuid_to_frameposlist[uuid_].append(
                (str(frame), elspace.to_space(pos))
            )

    return uuid_to_frameposlist


def load_potential_set_file(path, filename):
    ignore_set = set()
    file_path = path / filename
    if file_path.is_file():
        with file_path.open() as f:
            lines = f.read().splitlines()
        for l in lines:
            text = l.strip()
            if text and not text.startswith('#'):
                ignore_set.add(text)
    return ignore_set


# -----------------------------------------------------------------------------
# Copyright (C) 2016-2019 Angelos Evripiotis.
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
