"""Automatically mask rotomap images."""

import os

import cv2
import numpy

import mel.lib.common
import mel.lib.fs
import mel.lib.ui


def setup_parser(parser):
    parser.add_argument(
        'TRAINIMAGE',
        help="A path to the masked image to train on.")
    parser.add_argument(
        'IMAGES',
        nargs='*',
        help="A list of paths to images sets or images to automask.")


def process_args(args):

    skin_hist = train_image(
        cv2.imread(args.TRAINIMAGE),
        load_mask(args.TRAINIMAGE))

    for path in args.IMAGES:
        image = cv2.imread(path)
        # print('Test:')
        # test_image(image, mask, skin_hist)
        mask = mask_image(image, skin_hist)
        cv2.imwrite(path + '.mask.png', mask)


def load_mask(path):
    mask_path = path + '.mask.png'
    return cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)


def train_image(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_hist = calc_hist(hsv, mask)
    return skin_hist


# def test_image(image, mask, skin_hist):
#     width = image.shape[1]
#     height = image.shape[0]

#     stride = 10

#     hits = 0
#     misses = 0
#     for y in range(0, width, stride):
#         for x in range(0, height, stride):
#             frag = image[y:y+stride,x:x+stride]
#             hist = calc_hist(frag)
#             distance = cv2.compareHist(hist, skin_hist, cv2.HISTCMP_HELLINGER)
#             is_skin_hist = distance <= 0.5

#             mask_count = numpy.count_nonzero(mask[y:y+stride,x:x+stride])
#             mask_size = mask.size
#             is_skin_mask = mask_count > (mask.size // 2)
#             if is_skin_hist == is_skin_mask:
#                 hits += 1
#             else:
#                 misses += 1

#     print('hits:', hits, 'misses:', misses)


def mask_image(image, skin_hist):
    width = image.shape[1]
    height = image.shape[0]
    stride = 10

    mask = numpy.zeros((height, width, 1), numpy.uint8)

    for y in range(0, width, stride):
        for x in range(0, height, stride):
            frag = image[y:y + stride, x:x + stride]
            hsv = cv2.cvtColor(frag, cv2.COLOR_BGR2HSV)
            hist = calc_hist(hsv)
            distance = cv2.compareHist(hist, skin_hist, cv2.HISTCMP_HELLINGER)
            # distance = 1 - cv2.compareHist(hist, skin_hist, cv2.HISTCMP_CORREL)
            # distance = cv2.compareHist(hist, skin_hist, cv2.HISTCMP_CHISQR) / 100
            # distance = 1 - cv2.compareHist(hist, skin_hist, cv2.HISTCMP_INTERSECT)
            # distance = cv2.compareHist(hist, skin_hist, cv2.HISTCMP_CHISQR_ALT)
            # distance = cv2.compareHist(hist, skin_hist, cv2.HISTCMP_KL_DIV)
            is_skin_hist = distance <= 0.5
            if is_skin_hist:
                mask[y:y + stride, x:x + stride] = 255

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
        # c = numpy.delete(c, range(0, len(c), 2), 0)
        cv2.drawContours(mask, [c], -1, (255), -1)

    return mask


def calc_hist(image, mask=None):
    return cv2.calcHist(
        [image],
        [0, 0],
        mask,
        [8] * 2,
        [0, 256] * 2)


def print_hist(hist):
    max_value = hist.max()
    for row in hist:
        row_strings = []
        for value in row:
            percent = int(100 * value // max_value)
            out = percent_to_bar(percent)
            row_strings.append(out)
        print(' '.join(row_strings))


def percent_to_bar(value):
    num_marks = int(value // (100 // 5))
    marks = '*' * num_marks
    return '{:.<5}'.format(marks)
