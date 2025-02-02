"""Export command for rotomap.

This command processes all images in a given directory: if a corresponding mask file exists,
the masked areas are replaced with solid green for anonymity.
Assumes mask files are named as <basename>_mask<ext> (e.g., img.png -> img_mask.png).
"""

import cv2
import numpy as np

import mel.lib.common
import mel.rotomap.mask
import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument("source", help="image to load from.")
    parser.add_argument("output", help="image to save to.")


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


def draw_grid(image):
    import string

    height, width, _ = image.shape
    markers = 5  # number of ticks along each margin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.8
    thickness = 20
    tick_length = 50  # length of tick marks

    # Top margin: compute tick positions
    tick_top = int(15 * font_scale)
    tick_bottom = tick_top + tick_length
    ticks_x = [int(i * (width - 1) / (markers - 1)) for i in range(markers)]
    # draw ticks
    for x in ticks_x:
        cv2.line(
            image, (x, tick_top), (x, tick_bottom), (255, 255, 255), thickness
        )
    # draw labels halfway between ticks (numeric)
    for i in range(markers - 1):
        label = str(i + 1)
        x = (ticks_x[i] + ticks_x[i + 1]) // 2
        # get text dimensions
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        mid_y = (tick_top + tick_bottom) // 2
        text_x = x - text_width // 2
        text_y = mid_y + text_height // 2
        cv2.putText(
            image,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    # Left margin: compute tick positions with a vertical offset of 10
    tick_left = int(5 * font_scale)
    tick_right = tick_left + tick_length
    ticks_y = [
        int(i * (height - 1) / (markers - 1)) + 10 for i in range(markers)
    ]
    # draw ticks
    for y in ticks_y:
        cv2.line(
            image, (tick_left, y), (tick_right, y), (255, 255, 255), thickness
        )
    # draw labels halfway between ticks (alphabetical)
    for i in range(markers - 1):
        label = string.ascii_uppercase[i]
        y = (ticks_y[i] + ticks_y[i + 1]) // 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        mid_x = (tick_left + tick_right) // 2
        text_x = mid_x - text_width // 2
        text_y = y + text_height // 2
        cv2.putText(
            image,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return image


def process_args(args):
    image = load_image(args.source)
    image = draw_grid(image)  # added grid markers on margins

    # Convert RGB to BGR for cv2.imwrite
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mel.lib.common.write_image(args.output, image)

    return 0
