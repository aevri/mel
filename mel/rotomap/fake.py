"""Create fake rotomap data."""

import math
import random

import cv2
import numpy as np

import mel.lib.common
import mel.lib.vec3 as vec3
import mel.rotomap.moles
import mel.rotomap.raytrace

# Note that we'll use these prefixes to make things easier to name:
#
# - p_: 'point'
# - d_: direction (normalized vector)
# - v_: direction and magnitude
# - m_: magnitude (scalar)
#

# Perhaps later we can also take inspiration from:
#
#   "Unreal Engine: Creating Human Skin"
#
# https://docs.unrealengine.com/en-US/Engine/Rendering/Materials/HowTo/Human_Skin


def random_moles(num_moles):
    return [
        {
            "uuid": mel.rotomap.moles.make_new_uuid(),
            "radius": (random.normalvariate(0.02, 0.005) * math.pi),
            "y_offset": random.uniform(-1, 1),
            "longitude_rads": random.uniform(0, math.pi * 2),
        }
        for _ in range(num_moles)
    ]


def render_moles(moles, image_width, image_height, rot_0_to_1):
    rot_offset_rads = rot_0_to_1 * math.pi * 2
    p_light = vec3.make(1.0, 0.0, -1.0)

    p_cyl = vec3.make(0.0, 0.0, 1.0)
    m_cyl_radius = 1.0

    p_ray = vec3.make(0.0, 0.0, -1.0)
    aspect_r = image_height / image_width
    x = np.tile(np.linspace(-1.0, 1.0, image_width), image_height)
    y = np.repeat(np.linspace(aspect_r, -aspect_r, image_height), image_width)
    z = np.repeat(-vec3.zval(p_ray), image_width * image_height)
    d_ray = vec3.normalized(vec3.make_from_columns(x, y, z))

    hit, p_hit = mel.rotomap.raytrace.intersect_ray_cylinder(
        p_ray, d_ray, p_cyl, m_cyl_radius
    )
    hit = np.squeeze(hit)

    assert np.any(hit)

    part_p_hit = p_hit[hit, :]
    part_d_ray = d_ray[hit, :]
    part_color = mel.rotomap.raytrace.light_cylinder(
        p_ray,
        part_d_ray,
        part_p_hit,
        p_light,
        p_cyl,
        m_cyl_radius,
        moles,
        rot_offset_rads,
    )
    color = vec3.zeros(len(hit))
    color[hit, :] = part_color

    visible_moles = []
    for m in moles:
        mole_y_pos = m["y_offset"]
        mole_rot = m["longitude_rads"] + rot_offset_rads
        p_mole = mel.rotomap.raytrace.cylinder_mole_pos(
            p_cyl, m_cyl_radius, mole_y_pos, mole_rot
        )
        d_mole_to_eye = vec3.normalized(p_ray - p_mole)
        p_screen = mel.rotomap.raytrace.intersect_ray_at_z_pos(p_mole, d_mole_to_eye, 0)
        d_eye_to_mole = d_mole_to_eye * -1
        hit, p_hit = mel.rotomap.raytrace.intersect_ray_cylinder(
            p_ray,
            d_eye_to_mole,
            p_cyl,
            m_cyl_radius,
        )

        if vec3.mag_sq(p_hit - p_mole) < 0.001:
            image_x = int((vec3.xval(p_screen) * image_width * 0.5) + image_width // 2)
            image_y = int(
                (vec3.yval(p_screen) * image_width * -0.5) + image_height // 2
            )
            mel.rotomap.moles.add_mole(visible_moles, image_x, image_y, m["uuid"])

    image = color.reshape((image_height, image_width, 3))
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, visible_moles


# -----------------------------------------------------------------------------
# Copyright (C) 2020 Angelos Evripiotis.
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
