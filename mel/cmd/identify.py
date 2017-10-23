"""Guess which mole is which in a rotomap image."""


import json

import mel.lib.math
import mel.rotomap.relate


def setup_parser(parser):
    parser.add_argument(
        'TARGET',
        nargs='+',
        help="Paths to images to identify.")
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help="Print information about the processing.")


def process_args(args):
    for path in args.TARGET:
        if args.verbose:
            print('Target:', path)
        image = cv2.imread(path)
        mask = mel.rotomap.mask.guess_mask_otsu(image)
        mel.lib.common.write_image(path + '.mask.png', mask)
