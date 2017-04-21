"""Automatically mask rotomap images."""

import cv2
import numpy
import os

import mel.lib.common
import mel.lib.fs
import mel.lib.svm
import mel.lib.ui
import mel.rotomap.mask


def setup_parser(parser):
    parser.add_argument(
        '--source',
        '-s',
        nargs='*',
        help="Path to the masked images to train on.",
    )
    parser.add_argument(
        '--trial',
        nargs='*',
        default=[],
        help="Paths to images to trial.")
    parser.add_argument(
        '--target',
        '-t',
        nargs='*',
        default=[],
        help="Paths to images to automask.")
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help="Print information about the processing.")
    parser.add_argument(
        '--svm-c',
        type=float,
        default=0.03125,
        help="'C' parameter to the RBF SVM.")
    parser.add_argument(
        '--svm-gamma',
        type=float,
        default=0.03125,
        help="'gamma' parameter to the RBF SVM.")


def process_args(args):

    classifier = mel.lib.svm.NamedClassifier(
        mel.lib.svm.Classifier(
            c=args.svm_c,
            gamma=args.svm_gamma))

    for path in args.source:
        if args.verbose:
            print('Source:', path)
        train(classifier, _load_mole_data(path))

    if args.verbose:
        print('Training ..')
    classifier.train()

    for path in args.trial:
        if args.verbose:
            print('Trial on:', path)
        trial(classifier, _load_mole_data(path), args.svm_c, args.svm_gamma)

    for path in args.target:
        if args.verbose:
            print('Target:', path)
        target(classifier, _load_mole_data(path), path)


class _MoleImageData():

    def __init__(self):
        self.photo = None
        self.not_skin = None
        self.skin = None
        self.exclude = None
        self.automask = None
        self.moles = None


def _load_mask_image_or_none(path):
    if os.path.isfile(path):
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return None


def _load_mask_image_or_zeros(photo, path):
    mask = _load_mask_image_or_none(path)
    if mask is None:
        mask = numpy.zeros((*photo.shape[:2], 1), numpy.uint8)
    return mask


def _load_mole_data(photo_path):
    data = _MoleImageData()
    data.photo = cv2.imread(photo_path)

    data.automask = _load_mask_image_or_zeros(
        data.photo, photo_path + '.mask.png')
    data.skin = _load_mask_image_or_zeros(
        data.photo, photo_path + '.mask.skin.png')
    data.not_skin = _load_mask_image_or_zeros(
        data.photo, photo_path + '.mask.not_skin.png')
    data.exclude = _load_mask_image_or_zeros(
        data.photo, photo_path + '.mask.exclude.png')

    data.roi = cv2.bitwise_or(data.automask, data.skin)
    data.roi = cv2.bitwise_and(data.roi, cv2.bitwise_not(data.not_skin))
    data.roi = cv2.bitwise_and(data.roi, cv2.bitwise_not(data.exclude))

    data.moles = mel.rotomap.moles.load_image_moles(photo_path)
    return data


def train(classifier, data):
    hsv_photo = cv2.cvtColor(data.photo, cv2.COLOR_BGR2HSV)
    yield_regions = mel.rotomap.mask.yield_regions

    for photo, skin, not_skin in yield_regions(
            hsv_photo, data.skin, data.not_skin):

        region_class = None
        if skin.mean() == 255:
            region_class = "skin"
        elif not_skin.mean() == 255:
            region_class = "not skin"

        if region_class is not None:
            hist = mel.rotomap.mask.calc_hist(photo, width=32)
            classifier.add_sample(
                hist_to_sample(hist),
                region_class)


def trial(classifier, data, svm_c, svm_gamma):
    hsv_photo = cv2.cvtColor(data.photo, cv2.COLOR_BGR2HSV)
    yield_regions = mel.rotomap.mask.yield_regions

    hits = 0
    misses = 0
    for photo, skin, not_skin in yield_regions(
            hsv_photo, data.skin, data.not_skin):

        region_class = None
        if skin.mean() == 255:
            region_class = "skin"
        elif not_skin.mean() == 255:
            region_class = "not skin"

        if region_class is not None:
            hist = mel.rotomap.mask.calc_hist(photo, width=32)
            if classifier.predict(hist_to_sample(hist)) == region_class:
                hits += 1
            else:
                misses += 1

    success_rate = hits / max(hits + misses, 1)
    print(svm_c, svm_gamma, success_rate, sep=',')


def target(classifier, data, path):
    hsv_photo = cv2.cvtColor(data.photo, cv2.COLOR_BGR2HSV)
    data.automask = numpy.zeros((*data.photo.shape[:2], 1), numpy.uint8)
    yield_regions = mel.rotomap.mask.yield_regions

    for photo, mask in yield_regions(hsv_photo, data.automask):

        hist = mel.rotomap.mask.calc_hist(photo, width=32)
        if classifier.predict(hist_to_sample(hist)) == "skin":
            mask[:] = 255
        else:
            mask[:] = 0

    mask = mel.rotomap.mask.shrunk_to_largest_region(mask)

    # data.automask = cv2.bitwise_or(
    #     data.automask, data.skin)
    # data.automask = cv2.bitwise_and(
    #     data.automask, cv2.bitwise_not(data.not_skin))
    # data.automask = cv2.bitwise_and(
    #     data.automask, cv2.bitwise_not(data.exclude))

    cv2.imwrite(path + '.mask.png', data.automask)


def hist_to_sample(hist):
    return hist[:12, :12].flatten() / 100.0
