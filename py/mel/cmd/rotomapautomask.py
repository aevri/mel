"""Automatically mask rotomap images."""

import cv2

import mel.lib.common
import mel.lib.fs
import mel.lib.ui
import mel.rotomap.mask


def setup_parser(parser):
    parser.add_argument(
        '--source',
        '-s',
        help="Path to the masked image to train on.",
    )
    parser.add_argument(
        '--target',
        '-t',
        nargs='+',
        help="Paths to images to automask.")
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help="Print information about the processing.")


def process_args(args):

    if args.verbose:
        print('Source:', args.source)
    skin_hist = mel.rotomap.mask.histogram_from_image_mask(
        cv2.imread(args.source),
        mel.rotomap.mask.load(args.source))

    for path in args.target:
        if args.verbose:
            print('Target:', path)
        image = cv2.imread(path)
        mask = mel.rotomap.mask.guess_mask(image, skin_hist)
        cv2.imwrite(path + '.mask.png', mask)
