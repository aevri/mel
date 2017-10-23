"""Set moles to be manually confirmed to have the correct UUID."""

import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        'JSON_FILE',
        nargs='+',
        help="A list of paths to image json files.")


def process_args(args):
    changed_count = 0

    for path in args.JSON_FILE:
        moles = mel.rotomap.moles.load_json(path)
        for m in moles:
            if not m.get(mel.rotomap.moles.KEY_IS_CONFIRMED, False):
                changed_count += 1
                m[mel.rotomap.moles.KEY_IS_CONFIRMED] = True
        mel.rotomap.moles.save_json(path, moles)

    print(f'Confirmed {changed_count} moles.')
