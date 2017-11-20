"""Automatically mark moles on rotomap images."""

import cv2

import mel.rotomap.detectmoles
import mel.rotomap.mask
import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        'IMAGES',
        nargs='+',
        help="A list of paths to images to automark.")
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help="Print information about the processing.")


def process_args(args):
    for path in args.IMAGES:
        if args.verbose:
            print(path)
        image = mel.rotomap.moles.load_image(path)
        mask = mel.rotomap.mask.load(path)
        moles = mel.rotomap.detectmoles.moles(image, mask)
        mel.rotomap.moles.save_image_moles(moles, path)
