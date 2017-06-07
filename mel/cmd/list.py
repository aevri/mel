"""List the moles in a mole catalog."""

import mel.micro.fs


def setup_parser(parser):
    parser.add_argument(
        '--only-no-micro',
        action='store_true',
        help="Only list moles that have no microscope images.")

    assistance = parser.add_mutually_exclusive_group()
    assistance.add_argument(
        '--without-assistance',
        action='store_true',
        help="Only list moles that don't require assistance to capture.")
    assistance.add_argument(
        '--with-assistance',
        action='store_true',
        help="Only list moles that require assistance to capture.")

    parser.add_argument(
        '--sort',
        default='unsorted',
        nargs='?',
        choices=['lastmicro'],
        help='Sort the moles by the date of their last micro image. '
             'This means moles with no images are first then the oldest '
             'images.')

    format_args = parser.add_mutually_exclusive_group()

    format_args.add_argument(
        '--format',
        default="{relpath}",
        help="Print the results with the specified format. Defaults to "
             "'{relpath}'. Available keys: relpath, lastmicro, "
             "lastmicro_age_days, id.")

    format_args.add_argument(
        '--ages',
        dest='format',
        action='store_const',
        const='{relpath} {lastmicro_age_days}',
        help="Print the relative paths of the moles and their ages.")

    format_args.add_argument(
        '--ids',
        dest='format',
        action='store_const',
        const='{relpath} {id}',
        help="Print the relative paths of the moles and their ids.")


def process_args(args):
    for mole in _yield_mole_dirs('.', args):
        mole_data = {
            'relpath': mole.refrelpath,
            'lastmicro': mole.last_micro,
            'lastmicro_age_days': mole.last_micro_age_days,
            'id': mole.id,
        }

        print(args.format.format(**mole_data))


def _yield_mole_dirs(rootpath, args):

    mole_iter = mel.micro.fs.yield_moles(rootpath)

    if args.sort == 'lastmicro' or args.sort is None:
        def keyfunc(x):
            if not x.micro_image_names:
                return str()
            return sorted(x.micro_image_names)[-1]
        mole_iter = sorted(mole_iter, key=keyfunc)

    for mole in mole_iter:

        if args.only_no_micro and mole.micro_image_names:
            continue

        if args.without_assistance and mole.need_assistance:
            continue

        if args.with_assistance and not mole.need_assistance:
            continue

        yield mole
