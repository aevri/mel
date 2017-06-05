"""Automatically mask rotomap images."""

import cv2
import numpy

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
        train(classifier, path)

    if args.verbose:
        print('Training ..')
    classifier.train()

    for path in args.trial:
        if args.verbose:
            print('Trial on:', path)
        trial(classifier, path, args.svm_c, args.svm_gamma)

    for path in args.target:
        if args.verbose:
            print('Target:', path)
        target(classifier, path)


def train(classifier, path):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV)
    mask = mel.rotomap.mask.load(path)
    yield_regions = mel.rotomap.mask.yield_regions

    for image_frag, mask_frag in yield_regions(image, mask):
        hist = mel.rotomap.mask.calc_hist(image_frag, width=32)
        coverage = mask_frag.mean()
        if coverage == 0 or coverage == 255:
            region_class = "not skin"
            if coverage == 255:
                region_class = "skin"
            classifier.add_sample(
                hist_to_sample(hist),
                region_class)


def trial(classifier, path, svm_c, svm_gamma):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV)
    mask = mel.rotomap.mask.load(path)
    yield_regions = mel.rotomap.mask.yield_regions

    hits = 0
    misses = 0
    for image_frag, mask_frag in yield_regions(image, mask):
        hist = mel.rotomap.mask.calc_hist(image_frag, width=32)
        coverage = mask_frag.mean()
        if coverage == 0 or coverage == 255:
            region_class = "not skin"
            if coverage == 255:
                region_class = "skin"
            if classifier.predict(hist_to_sample(hist)) == region_class:
                hits += 1
            else:
                misses += 1

    success_rate = hits / max(hits + misses, 1)
    print(svm_c, svm_gamma, success_rate, sep=',')


def target(classifier, path):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV)
    mask = numpy.zeros((*image.shape[:2], 1), numpy.uint8)
    yield_regions = mel.rotomap.mask.yield_regions

    for image_frag, mask_frag in yield_regions(image, mask):
        hist = mel.rotomap.mask.calc_hist(image_frag, width=32)
        if classifier.predict(hist_to_sample(hist)) == "skin":
            mask_frag[:] = 255
        else:
            mask_frag[:] = 0

    mask = mel.rotomap.mask.shrunk_to_largest_region(mask)

    mel.lib.common.write_image(path + '.mask.png', mask)


def hist_to_sample(hist):
    return hist[:12, :12].flatten() / 100.0
