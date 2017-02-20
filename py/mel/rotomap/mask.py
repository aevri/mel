"""Manage image masks."""

import cv2
import numpy


def load(mole_image_path):
    return cv2.imread(
        mole_image_path + '.mask.png',
        cv2.IMREAD_UNCHANGED)


def histogram_from_image_mask(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_hist = _calc_hist(hsv, mask)
    return skin_hist


def _calc_hist(image, mask=None):
    return cv2.calcHist(
        [image],
        [0, 0],
        mask,
        [8] * 2,
        [0, 256] * 2)


def guess_mask(image, skin_hist):
    width = image.shape[1]
    height = image.shape[0]
    stride = 10

    mask = numpy.zeros((height, width, 1), numpy.uint8)

    for y in range(0, width, stride):
        for x in range(0, height, stride):
            frag = image[y:y + stride, x:x + stride]
            hsv = cv2.cvtColor(frag, cv2.COLOR_BGR2HSV)
            hist = _calc_hist(hsv)
            distance = cv2.compareHist(hist, skin_hist, cv2.HISTCMP_HELLINGER)
            is_skin_hist = distance <= 0.5
            if is_skin_hist:
                mask[y:y + stride, x:x + stride] = 255

    # Pick only the biggest connected region - there may be other things in the
    # image which have a similar colour profile. Assume that the biggest region
    # is the area that we're interested in.

    _, contours, _ = cv2.findContours(
        mask.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE)

    max_area = 0
    max_index = None
    for i, c in enumerate(contours):
        if c is not None and len(c) > 5:
            area = cv2.contourArea(c)
            if max_index is None or area > max_area:
                max_area = area
                max_index = i

    mask = numpy.zeros((height, width, 1), numpy.uint8)
    if max_index is not None:
        c = contours[max_index]
        cv2.drawContours(mask, [c], -1, (255), -1)

    return mask
