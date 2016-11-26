"""Automatically mask rotomap images."""

import os

import cv2
import numpy

import mel.lib.common
import mel.lib.fs
import mel.lib.ui


def setup_parser(parser):
    parser.add_argument(
        'IMAGES',
        nargs='+',
        help="A list of paths to images sets or images to automark.")


def process_args(args):
    for path in args.IMAGES:
        image = cv2.imread(path)
        mask = load_mask(path)
        moles = detect_moles(image, mask)
        mel.rotomap.moles.save_image_moles(moles, path)


def load_mask(path):
    mask_path = path + '.mask.png'
    return cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)


def detect_moles(image, mask):

    # Moles show up brightly in the 'Saturation' dimension.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = image[:, :, 1]

    # Apply the mask
    image = cv2.bitwise_and(image, image, mask=mask)

    # The simple blob detector will find patches of darkness, so invert the
    # image to make the moles dark.
    image = cv2.bitwise_not(image, image)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByCircularity = False
    params.filterByConvexity = False

    params.minThreshold = 0
    params.maxThreshold = 256

    params.filterByArea = True
    params.minArea = 1500
    params.minArea = 50

    params.filterByInertia = True
    params.minInertiaRatio = 0.25

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)

    moles = []
    for point in keypoints:
        xy = point.pt
        mel.rotomap.moles.add_mole(moles, int(xy[0]), int(xy[1]))
        m = moles[-1]
        m['radius'] = point.size // 2
    return moles
