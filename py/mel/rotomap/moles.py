"""Work with a collection of moles."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import uuid


def load_image_moles(image_path):
    moles_path = image_path + '.json'
    moles = []
    if os.path.exists(moles_path):
        with open(moles_path) as moles_file:
            moles = json.load(moles_file)

    converted = []
    for m in moles:
        if type(m) is list:
            m = {'x': m[0], 'y': m[1]}
        if 'uuid' not in m:
            m['uuid'] = uuid.uuid4().hex
        converted.append(m)

    return converted


def save_image_moles(moles, image_path):
    moles_path = image_path + '.json'
    with open(moles_path, 'w') as moles_file:
        json.dump(
            moles,
            moles_file,
            indent=4,
            separators=(',', ': '),
            sort_keys=True)

        # There's no newline after dump(), add one here for happier viewing
        print(file=moles_file)


def add_mole(moles, x, y):
    moles.append({
        'x': x,
        'y': y,
        'uuid': uuid.uuid4().hex,
    })


def nearest_mole_index(moles, x, y):
    nearest_index = None
    nearest_distance = None
    for i, mole in enumerate(moles):
        dx = x - mole['x']
        dy = y - mole['y']
        distance = math.sqrt(dx * dx + dy * dy)
        if nearest_distance is None or distance < nearest_distance:
            nearest_index = i
            nearest_distance = distance

    return nearest_index


def set_nearest_mole_uuid(moles, x, y, mole_uuid):
    nearest_index = nearest_mole_index(moles, x, y)
    if nearest_index is not None:
        moles[nearest_index]['uuid'] = mole_uuid


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
