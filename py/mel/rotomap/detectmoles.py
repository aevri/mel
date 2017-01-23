"""Detect moles in an image."""

import cv2
import numpy


def draw_experimental(image, mask):
    keypoints, image = calc_keypoints(image, mask)
    image = cv2.drawKeypoints(
        image,
        keypoints,
        numpy.array([]),
        (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return image


def calc_keypoints(original_image, mask):
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    image = image[:, :, 1]

    image = cv2.bitwise_and(image, image, mask=mask)
    image = cv2.bitwise_not(image, image)

    # Note that the static analysis tool 'vulture' doesn't seem to be happy
    # with using attributes on 'params'. The only workaround appears to be
    # ignoring the whole file.
    params = cv2.SimpleBlobDetector_Params()
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    params.minThreshold = 0
    params.maxThreshold = 256

    params.filterByArea = True
    params.maxArea = 1500
    params.minArea = 50

    params.filterByInertia = True
    params.minInertiaRatio = 0.25

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)

    return keypoints, image
