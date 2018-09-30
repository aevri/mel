"""Work with a collection of moles."""


import argparse
import collections
import json
import math
import pathlib
import uuid

import cv2
import numpy

import mel.lib.image
import mel.lib.math

import mel.rotomap.mask


KEY_IS_CONFIRMED = 'is_uuid_canonical'

IGNORE_NEW_FILENAME = 'ignore-new'
IGNORE_MISSING_FILENAME = 'ignore-missing'


class RotomapDirectory():
    """RotomapFrame-s for all images in a single rotomap dir."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        if not self.path.is_dir():
            raise ValueError(
                '"{}" is not a directory, so not a rotomap.'.format(self.path))

        self.image_paths = [
            str(f) for f in self.path.iterdir()
            if f.suffix.lower() == '.jpg'
        ]
        self.image_paths.sort()

        if not self.image_paths:
            raise ValueError(
                '"{}" has no images, so not a rotomap.'.format(self.path))

    def yield_mole_lists(self):
        """Yield (image_path, mole_list) for all mole image files."""
        for imagepath in self.image_paths:
            yield imagepath, load_image_moles(imagepath)

    def yield_frames(self):
        for imagepath in self.image_paths:
            yield RotomapFrame(imagepath)

    def __repr__(self):
        return f'RotomapDirectory({self.path!r})'


class RotomapFrame():
    """Image and mole data for a single image in a rotomap."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        if self.path.is_dir():
            raise ValueError(
                f'Expected file, not directory: {path}')
        if not self.path.exists():
            raise ValueError(
                f'Path does not exist: {path}')
        if self.path.suffix.lower() != '.jpg':
            raise ValueError(
                f'Unrecognised suffix for rotomap frame: {path}')

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


class MoleData():
    """Iterables of UUIDs, locations, and other data on moles in an image."""

    def __init__(self, mole_iter):
        self.moles = tuple(mole_iter)
        self.uuids = frozenset(m['uuid'] for m in self.moles)
        self.uuid_points = to_uuid_points(self.moles)
        self.canonical_uuids = frozenset(
            m['uuid']
            for m in self.moles
            if m[KEY_IS_CONFIRMED]
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


class MoleListDiff():

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


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(
            data,
            f,
            indent=4,
            separators=(',', ': '),
            sort_keys=True)

        # There's no newline after dump(), add one here for happier viewing
        print(file=f)


def make_new_uuid():
    return uuid.uuid4().hex


def add_mole(moles, x, y, mole_uuid=None):
    is_uuid_canonical = True
    if mole_uuid is None:
        mole_uuid = make_new_uuid()
        is_uuid_canonical = False

    moles.append({
        'x': x,
        'y': y,
        'uuid': mole_uuid,
        KEY_IS_CONFIRMED: is_uuid_canonical,
    })


def sorted_by_distances(mole_list, x, y):

    def sqdist(mole):
        dist_x = x - mole['x']
        dist_y = y - mole['y']
        return (dist_x * dist_x) + (dist_y * dist_y)

    return sorted(mole_list, key=sqdist)


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


def mole_to_point(mole):
    pos = numpy.array((mole['x'], mole['y']))
    mel.lib.math.raise_if_not_int_vector2(pos)
    return pos


def to_uuid_points(moles):
    uuid_points = {}
    for m in moles:
        uuid_points[m['uuid']] = mole_to_point(m)
    return uuid_points


def set_molepos_to_nparray(mole, nparray):
    mole['x'] = int(nparray[0])
    mole['y'] = int(nparray[1])


def is_value_in_range(value, lower, upper):
    return value >= lower and value <= upper


def is_point_in_rect(point, rect):
    return (
        is_value_in_range(point[0], rect[0], rect[2]) and
        is_value_in_range(point[1], rect[1], rect[3])
    )


def triangle_to_points(triangle):
    return (
        (triangle[0], triangle[1]),
        (triangle[2], triangle[3]),
        (triangle[4], triangle[5])
    )


def is_triangle_in_rect(triangle, rect):
    return all(is_point_in_rect(p, rect) for p in triangle_to_points(triangle))


def get_mole_triangles(mole_list, image_rect):
    subdiv = cv2.Subdiv2D(image_rect)
    for mole in mole_list:
        subdiv.insert((mole['x'], mole['y']))

    # filter the list of triangles to those that only have points that fit in
    # the rectangle
    triangle_list = []
    for triangle in subdiv.getTriangleList():
        if is_triangle_in_rect(triangle, image_rect):
            triangle_list.append(triangle)

    return triangle_list


def get_best_triangle_for_mapping(triangle_list, point):
    best_triangle = None
    best_result = None
    for triangle in triangle_list:
        # Ignore "Instance of 'tuple' has no 'astype' member (no-member)" from
        # pylint.
        # pylint: disable=no-member
        #
        # pointPolygonTest() will fail if we don't pass it values exactly like
        # this.
        contour = numpy.array(triangle_to_points(triangle)).astype('float32')
        # pylint: enable=no-member
        result = cv2.pointPolygonTest(contour, tuple(point), True)
        if best_result is None or result > best_result:
            best_result = result
            best_triangle = triangle

    return best_triangle


def get_moles_from_points(mole_list, point_list):
    output_moles = []
    for point in point_list:
        for mole in mole_list:
            molepoint = mole_to_point(mole)
            if numpy.allclose(point, molepoint):
                output_moles.append(mole)

    if len(point_list) != len(output_moles):
        raise ValueError('Not all points match moles: {}, {}'.format(
            point_list, mole_list))

    return output_moles


def get_best_moles_for_mapping(molepoint, mole_list, image_rect):

    if len(mole_list) < 3:
        if len(mole_list) == 0:
            return None
        else:
            return mole_list

    triangle_list = get_mole_triangles(mole_list, image_rect)
    best_triangle = get_best_triangle_for_mapping(triangle_list, molepoint)

    # Discard triangles that are not very equilateral, they seem to give bad
    # mappings.
    if best_triangle is not None:
        points = triangle_to_points(best_triangle)
        distances = [
            mel.lib.math.distance_2d(points[i - 1], points[i])
            for i in range(3)
        ]
        max_ = max(*distances)
        if max_ == 0:
            best_triangle = None
        else:
            norm_distances = [x / max_ for x in distances]
            for x in norm_distances:
                if x <= 0.5:
                    best_triangle = None

    moles_for_mapping = None
    if best_triangle is not None:
        moles_for_mapping = get_moles_from_points(
            mole_list, triangle_to_points(best_triangle))
    else:
        # Two nearest moles to map with is better than none
        return sorted_by_distances(
            mole_list, molepoint[0], molepoint[1])[:2]

    return moles_for_mapping


def mapped_pos(molepos, from_moles, to_moles):
    mel.lib.math.raise_if_not_int_vector2(molepos)

    if not from_moles:
        return molepos

    to_dict = {m['uuid']: m for m in to_moles}
    from_pos_list = [mole_to_point(m) for m in from_moles]
    to_pos_list = [mole_to_point(to_dict[m['uuid']]) for m in from_moles]

    num_pairs = len(from_pos_list)

    if num_pairs > 3:
        # Ideally we'd handle 4 points by using cv2.getPerspectiveTransform()
        # as in this article:
        #
        #     http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
        #
        # We could take advantage of more than 4 points by using
        # findHomography(), which can use various methods to deal with the
        # ambiguity introduced by multiple points.
        #
        # In practice, trying to use more than 3 points proved difficult as
        # naively including neighboring triangles resulted in adding co-linear
        # points.
        #
        raise ValueError('Too many moles')
    elif num_pairs == 3:
        # The best we can do here is to determine the translation, rotation and
        # scaling to apply in order to map from one triangle to the other. If
        # we want this to be a perspective transformation then we'd need an
        # additional point.
        #
        # pylint: disable=assignment-from-no-return
        transform = numpy.matrix(cv2.getAffineTransform(
            numpy.float32(from_pos_list),
            numpy.float32(to_pos_list))).transpose()
        # pylint: enable=assignment-from-no-return
        pos = numpy.array([molepos[0], molepos[1], 1.0]) * transform
        pos = numpy.array(pos)
        molepos = pos[0]
    elif num_pairs > 0:
        # Here we'll just assume that the transformation is a translation and
        # compute it from the first pair of points.
        translation = to_pos_list[0] - from_pos_list[0]
        molepos += translation
    # elif num_pairs > 2:
        # In later work, to take advantage of 2 pairs of points, we'll handle
        # it like so:
        #
        # Here we'll assume that the line through the supplied points is
        # roughly perpendicular to the axis of rotation. This means that we'd
        # expect the distance of the point from the line to be constant across
        # the transformation.

    return molepos


def frames_to_uuid_frameposlist(frame_iterable, canonical_only=False):
    uuid_to_frameposlist = collections.defaultdict(list)

    for frame in frame_iterable:

        if 'ellipse' not in frame.metadata:
            raise Exception(
                f'{frame} has no ellipse metadata, '
                'try running "rotomap-calc-space"')

        ellipse = frame.metadata['ellipse']
        elspace = mel.lib.ellipsespace.Transform(ellipse)
        for uuid_, pos in frame.moledata.uuid_points.items():
            if canonical_only and uuid_ not in frame.moledata.canonical_uuids:
                continue
            uuid_to_frameposlist[uuid_].append(
                (str(frame), elspace.to_space(pos)))

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
# Copyright (C) 2016-2017 Angelos Evripiotis.
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
