"""Create a montage image for a single mole from a rotomap."""

import cv2

import mel.lib.common
import mel.lib.image
import mel.rotomap.moles


def setup_parser(parser):

    parser.add_argument(
        'ROTOMAP',
        type=mel.rotomap.moles.make_argparse_rotomap_directory,
        help="Path to the rotomap to copy from.")

    parser.add_argument(
        'UUID',
        type=str,
        help="Unique id of the mole to copy.")

    parser.add_argument(
        'OUTPUT',
        type=str,
        help="Name of the image to write.")


def process_args(args):

    path_moles_list = []

    radius = 10
    montage_height = 1024

    for imagepath, moles in args.ROTOMAP.yield_mole_lists():
        for m in moles:
            if m['uuid'] == args.UUID:
                path_moles_list.append((imagepath, moles))

    if not path_moles_list:
        raise mel.cmd.error.UsageError(
            'UUID "{}" not found in rotomap "{}".'.format(
                args.UUID, args.ROTOMAP.path))

    # Pick 'best' image for this particular mole, assuming that the middle
    # image is where the mole is most prominent. This assumption is based on
    # the idea that the images represent a rotation around the subject, the
    # 'middle' image should be where the mole is most centered.
    #
    # TODO: Cater for the case where the image set represents a complete
    # rotation around the subject, and therefore the ends meet. e.g. if there
    # are 10 images, and the target mole appears in images 0, 1, 7, 8, 9 then
    # this rule will pick image 7 instead of 9.
    #
    path, mole_list = path_moles_list[len(path_moles_list) // 2]

    mole_dict = {m['uuid']: m for m in mole_list}
    mole = mole_dict[args.UUID]

    context_image = cv2.imread(path)
    x = mole['x']
    y = mole['y']
    mel.lib.common.indicate_mole(context_image, (x, y, radius))

    context_scale = montage_height / context_image.shape[0]

    detail_image = make_detail_image(context_image, x, y, montage_height)

    context_scaled_width = int(context_image.shape[1] * context_scale)
    context_image = cv2.resize(
        context_image,
        (context_scaled_width, montage_height))

    montage_image = mel.lib.image.montage_horizontal_inner_border(
        25, context_image, detail_image)

    cv2.imwrite(args.OUTPUT, montage_image)


def make_detail_image(context_image, x, y, size):
    half_size = size // 2
    left = max(x - half_size, 0)
    top = max(y - half_size, 0)
    right = left + half_size * 2
    bottom = top + half_size * 2
    return context_image[top:bottom, left:right]
