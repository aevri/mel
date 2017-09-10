"""Handy stuff for working in Jupyter notebooks."""

import cv2
import numpy

import mel.lib.moleimaging


def biggest_contour(mask):
    _, contours, _ = cv2.findContours(
        mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)

    # TODO: actually get the biggest
    return contours[0]


def ellipse_center_up_right(ellipse):
    center = ellipse[0]
    center = mel.lib.moleimaging.point_to_int_point(center)
    angle_degs = ellipse[2]

    # TODO: do this properly
    if angle_degs > 90:
        angle_degs -= 180

    up = (0, -1)
    up = mel.lib.moleimaging.rotate_point_around_pivot(
        up, (0, 0), angle_degs)

    right = (1, 0)
    right = mel.lib.moleimaging.rotate_point_around_pivot(
        right, (0, 0), angle_degs)

    umag = ellipse[1][1] / 2
    rmag = ellipse[1][0] / 2

    return center, up, right, umag, rmag


class EllipseSpace():

    def __init__(ellipse):
        self.center, self.up, self.right, umag, rmag =
            ellipse_center_up_right(ellipse)

        self.pos, self.center, self.up, self.right = (
            numpy.array(x) for x in (pos, center, up, right))

        self.mag = numpy.array((rmag, umag))
        self.inv_mag = 1 / self.mag

    def to_space(self, pos):
        pos -= self.center
        pos = numpy.array(
            numpy.dot(pos, self.right),
            numpy.dot(pos, self.up),
        )
        return pos * self.inv_mag

    def from_space(self, pos):
        return self.right * pos[0] * self.mag[0]
            + self.up * pos[1] * self.mag[1]
            + self.center


def from_ellipse_space(ellipse, pos):
    center, up, right, umag, rmag = ellipse_center_up_right(ellipse)

    p =  (
        int(right[0] * pos[0] * rmag + up[0] * pos[1] * umag + center[0]),
        int(right[1] * pos[0] * rmag + up[1] * pos[1] * umag + center[1]),
    )

    return p


def to_ellipse_space(ellipse, pos):
    center, up, right, umag, rmag = ellipse_center_up_right(ellipse)

    pos = (
        pos[0] - center[0],
        pos[1] - center[1],
    )

    pos = (
        pos[0] * right[0] + pos[1] * right[1],
        pos[0] * up[0] + pos[1] * up[1],
    )

    return (
        pos[0] / rmag,
        pos[1] / umag,
    )
