"""Handy stuff for working in Jupyter notebooks."""

import collections

import cv2
import numpy
import scipy

import mel.lib.moleimaging


def frames_to_uuid_poslist(frame_iterable):
    uuid_to_poslist = collections.defaultdict(list)

    for frame in frame_iterable:
        mask = frame.load_mask()
        contour = biggest_contour(mask)
        ellipse = cv2.fitEllipse(contour)
        elspace = EllipseSpace(ellipse)
        for uuid, pos in frame.moledata.uuid_points.items():
            uuid_to_poslist[uuid].append(elspace.to_space(pos))

    return uuid_to_poslist


class MoleClassifier():

    def __init__(self, uuid_to_poslist):
        self.uuids, self.poslistlist = zip(*uuid_to_poslist.items())
        yposlistlist = tuple(
            numpy.array(tuple(y for x, y in poslist))
            for poslist in self.poslistlist
        )
        self.ykernels = tuple(
            scipy.stats.gaussian_kde(yposlist)
            for yposlist in yposlistlist
        )

    def guess_from_ypos(self, ypos):
        densities = numpy.array(tuple(k(ypos) for k in self.ykernels))
        total_density = numpy.sum(densities)
        i = numpy.argmax(densities)
        d = densities[i][0]
        return self.uuids[i], d, d / total_density


def frames_to_uuid_frameposlist(frame_iterable):
    uuid_to_frameposlist = collections.defaultdict(list)

    for frame in frame_iterable:
        mask = frame.load_mask()
        contour = biggest_contour(mask)
        ellipse = cv2.fitEllipse(contour)
        elspace = EllipseSpace(ellipse)
        for uuid, pos in frame.moledata.uuid_points.items():
            uuid_to_frameposlist[uuid].append(
                (str(frame), elspace.to_space(pos)))

    return uuid_to_frameposlist


class MoleClassifier2():

    def __init__(self, uuid_to_frameposlist):
        self.uuids, self.frameposlistlist = zip(*uuid_to_frameposlist.items())
        # print(tuple(x[1] for x in self.frameposlistlist[0]))
        # self.poslistlist = tuple(
        #     numpy.array(x[1]) for x in self.frameposlistlist)
        self.poslistlist = tuple(
            numpy.array([numpy.array(pos) for frame, pos in frameposlist])
            for frameposlist in self.frameposlistlist
        )
        # print(self.poslistlist[0][:, 1])
        yposlistlist = tuple(
            poslist[:, 1]
            for poslist in self.poslistlist
        )
        print(yposlistlist)
        self.ykernels = tuple(
            scipy.stats.gaussian_kde(yposlist)
            for yposlist in yposlistlist
        )

    def guess_from_ypos(self, ypos):
        densities = numpy.array(tuple(k(ypos) for k in self.ykernels))
        total_density = numpy.sum(densities)
        i = numpy.argmax(densities)
        d = densities[i][0]
        return self.uuids[i], d, d / total_density


def biggest_contour(mask):
    _, contours, _ = cv2.findContours(
        mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_index = None
    for i, c in enumerate(contours):
        if c is not None and len(c) > 5:
            area = cv2.contourArea(c)
            if max_index is None or area > max_area:
                max_area = area
                max_index = i

    # TODO: actually get the biggest
    return contours[max_index]


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

    def __init__(self, ellipse):
        self.ellipse = ellipse

#         center, up, right, umag, rmag = ellipse_center_up_right(ellipse)

#         self.center, self.up, self.right = (
#             numpy.array(x) for x in (center, up, right))

#         self.mag = numpy.array((rmag, umag))
#         self.inv_mag = 1 / self.mag

    def to_space(self, pos):
        return to_ellipse_space(self.ellipse, pos)
        # pos = numpy.array(pos)
        # pos -= self.center
        # pos = numpy.array(
        #     numpy.dot(pos, self.right),
        #     numpy.dot(pos, self.up),
        # )
        # return pos * self.inv_mag

    def from_space(self, pos):
        return from_ellipse_space(self.ellipse, pos)
        # pos = numpy.array(pos)
        # return (self.right * pos[0] * self.mag[0]
        #     + self.up * pos[1] * self.mag[1]
        #     + self.center)


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
