"""Work with a collection of moles."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
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
