"""Shatter rotomap images into many small fragments, for training networks."""


import json
import os

import cv2

import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        'PATH',
        nargs='+',
        help="Path to the rotomap image files.")


def process_args(args):
    for i, path in enumerate(args.PATH):
        print(path)
        moles = mel.rotomap.moles.load_image_moles(path)
        image = cv2.imread(path)
        shatter(str(i) + '_', image, moles)


def shatter(prefix, image, moles):
    final_segment_width = 24
    final_segment_height = final_segment_width

    segment_width = 72
    segment_height = segment_width

    height, width = image.shape[:2]
    horizontal_segments = width // segment_width
    vertical_segments = height // segment_height

    i = 0
    for column in range(horizontal_segments):
        for row in range(vertical_segments):
            left = column * segment_width
            right = (column + 1) * segment_width
            top = row * segment_width
            bottom = (row + 1) * segment_width
            new_image = image[top:bottom, left:right]

            new_image = cv2.resize(
                new_image,
                (final_segment_width, final_segment_height))

            path = prefix + str(i) + '.jpg'
            cv2.imwrite(path, new_image)
            i += 1
