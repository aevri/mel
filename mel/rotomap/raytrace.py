"""Functions for drawing fake rotomaps using raytracing."""

import math

import numpy as np

import mel.lib.vec3 as vec3


# Note that we'll use these prefixes to make things easier to name:
#
# - p_: 'point'
# - d_: direction (normalized vector)
# - v_: direction and magnitude
# - m_: magnitude (scalar)
#

# This idea for numpy-based ray-tracing comes from here:
# https://www.excamera.com/sphinx/article-ray.html


def vec3_flat(v):
    assert vec3.is_vec3(v)
    copy = v.copy()
    copy[:, 1] = 0.0
    return copy


def intersect_ray_sphere(p_ray, d_ray, p_sph, m_radius):
    v_ray_to_sph = p_sph - p_ray
    m_ray_to_nearest = vec3.dot(d_ray, v_ray_to_sph)
    m_ray_to_nearest_sq = m_ray_to_nearest * m_ray_to_nearest
    m_ray_to_sph_sq = vec3.mag_sq(v_ray_to_sph)
    m_nearest_to_sph_sq = m_ray_to_sph_sq - m_ray_to_nearest_sq
    did_intersect = m_nearest_to_sph_sq <= (m_radius * m_radius)
    m_intrusion_sq = (m_radius * m_radius) - m_nearest_to_sph_sq
    m_intrusion = np.sqrt(np.maximum(0, m_intrusion_sq))
    m_distance = m_ray_to_nearest - m_intrusion
    p_intersection = p_ray + d_ray * m_distance
    return did_intersect, p_intersection


def intersect_ray_at_z_pos(pos, dir_, z):
    z_target = z - vec3.zcol(pos)
    z_ratio = z_target / vec3.zcol(dir_)
    x = z_ratio * vec3.xcol(dir_) + vec3.xcol(pos)
    y = z_ratio * vec3.ycol(dir_) + vec3.ycol(pos)
    return vec3.make_from_columns(x, y, z)


def intersect_ray_cylinder(p_ray, d_ray, p_cyl, radius):
    # Roughen the edges a little.
    radius += np.random.random((vec3.count(d_ray), 1)) * 0.01

    did_intersect, p_flat_intersection = intersect_ray_sphere(
        vec3_flat(p_ray),
        vec3.normalized(vec3_flat(d_ray)),
        vec3_flat(p_cyl),
        radius,
    )

    p_intersection = intersect_ray_at_z_pos(
        p_ray, d_ray, vec3.zcol(p_flat_intersection)
    )

    return did_intersect, p_intersection


def light_cylinder(p_ray, d_ray, p_hit, p_light, p_cyl, radius, moles):
    assert vec3.is_vec3(p_ray)
    assert vec3.is_vec3(d_ray)
    assert vec3.is_vec3(p_hit)
    assert vec3.is_vec3(p_light)
    assert vec3.is_vec3(p_cyl)
    skin_color = skin_colour_cylinder(p_cyl, radius, p_hit, moles)
    # Ambient lighting.
    color = skin_color * 0.2
    # Diffuse lighting.
    v_cyl_to_hit = p_hit - p_cyl
    v_flat_cyl_to_hit = vec3_flat(v_cyl_to_hit)
    d_normal = vec3.normalized(v_flat_cyl_to_hit)

    d_hit_to_light = vec3.normalized(p_light - p_hit)
    light_angle_cos = np.maximum(vec3.dot(d_normal, d_hit_to_light), 0)
    color += skin_color * light_angle_cos * 0.8
    return color


def cylinder_mole_pos(p_cyl, cyl_radius, mole_y_pos, mole_rot):
    d_mole = vec3.make(-math.sin(mole_rot), 0, math.cos(mole_rot))
    p_flat_mole = p_cyl + d_mole * cyl_radius
    p_mole = vec3.make(
        vec3.xval(p_flat_mole), mole_y_pos, vec3.zval(p_flat_mole)
    )
    return p_mole


def skin_colour_cylinder(p_cyl, radius, p_hit, moles):
    # Skin tone colours:
    # https://www.schemecolor.com/real-skin-tones-color-palette.php
    light_skin_colour = vec3.make(1, 0.86, 0.67)
    dark_skin_colour = vec3.make(198, 134, 66) * (1 / 255)

    dark_param = None
    for m in moles:
        mole_radius_sq = m["radius"] ** 2
        mole_y_pos = m["y_offset"]
        mole_rot = m["longitude_rads"]
        front_dir = vec3.make(-math.sin(mole_rot), 0, math.cos(mole_rot))
        hit_flat_dir = vec3.normalized(vec3_flat(p_hit) - vec3_flat(p_cyl))
        hit_f_cos = vec3.dot(hit_flat_dir, front_dir)
        hit_angle = np.arccos(hit_f_cos)

        hit_x_dist = radius * hit_angle
        hit_y_dist = vec3.ycol(p_hit) - mole_y_pos
        hit_sq_dist = (hit_x_dist ** 2) + (hit_y_dist ** 2)
        is_in_mole = hit_sq_dist < mole_radius_sq

        curr_dark_param = np.where(is_in_mole, 1, 0)
        if dark_param is None:
            dark_param = curr_dark_param
        else:
            dark_param |= curr_dark_param

    colour = (dark_skin_colour * dark_param) + (
        light_skin_colour * (1 - dark_param)
    )

    return colour


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
