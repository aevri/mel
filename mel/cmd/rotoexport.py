"""
Export command for rotomap.
This command processes all images in a given directory: if a corresponding mask file exists,
the masked areas are replaced with solid green for anonymity.
Assumes mask files are named as <basename>_mask<ext> (e.g., img.png -> img_mask.png).
"""

import cv2
import numpy as np

import mel.lib.common
import mel.rotomap.moles
import mel.rotomap.mask


def setup_parser(parser):
    parser.add_argument('source', help="image to load from.")
    parser.add_argument('output', help="image to save to.")


def load_image(image_path):
    # flags = cv2.IMREAD_UNCHANGED + cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR
    flags = cv2.IMREAD_COLOR
    try:
        original_image = cv2.imread(str(image_path), flags)
        if original_image is None:
            raise OSError(f"File not recognized by opencv: {image_path}")
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise OSError(f"Error handling image at: {image_path}") from e

    mask = mel.rotomap.mask.load(image_path)
    green = np.zeros(original_image.shape, np.uint8)
    green[:, :, 1] = 255
    image = cv2.bitwise_and(original_image, original_image, mask=mask)
    not_mask = cv2.bitwise_not(mask)
    green = cv2.bitwise_and(green, green, mask=not_mask)
    image = cv2.bitwise_or(image, green)
    return image


def process_args(args):
    image = load_image(args.source)
    
    # Convert RGB to BGR for cv2.imwrite
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    mel.lib.common.write_image(args.output, image)

    return 0
