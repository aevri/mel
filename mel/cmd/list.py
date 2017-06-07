"""List the moles in a mole catalog."""

import mel.micro.fs


# If we're trying to stick to 'Every mole compared every month', we won't want
# to accidentally exclude moles that were compared just under a month ago.
# Setting the default to roughly half a month should lean towards
# over-comparing rather than under-comparing. This isn't medical advice, just
# the personal inclination of the author.
_DEFAULT_NO_RECENT_DAYS = 15


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
        '--ignore-with-recent-micro',
        '--no-recent',
        default=0,
        type=int,
        metavar='DAYS',
        nargs='?',
        help='Do not list moles with micro images less than this number of '
             'days. Off by default, will assume {days} days if none '
             'specified.'.format(days=_DEFAULT_NO_RECENT_DAYS))

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
            if not x.micro_image_details:
                return str()
            return x.micro_image_details[-1].name
        mole_iter = sorted(mole_iter, key=keyfunc)

    no_recent_days = args.ignore_with_recent_micro
    if no_recent_days is None:
        no_recent_days = _DEFAULT_NO_RECENT_DAYS

    for mole in mole_iter:

        if args.only_no_micro and mole.micro_image_details:
            continue

        if args.without_assistance and mole.need_assistance:
            continue

        if args.with_assistance and not mole.need_assistance:
            continue

        if mole.last_micro_age_days is not None and no_recent_days is not None:
            if mole.last_micro_age_days < no_recent_days:
                continue

        yield mole
