"""List the moles in a mole catalog."""

import datetime

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

    parser.add_argument(
        '--format',
        default="{relpath}",
        help="Print the results with the specified format. Defaults to "
             "'{relpath}'. Available keys: relpath, lastmicro, "
             "lastmicro_age_days, id.")



def process_args(args):
    if args.sort == 'lastmicro' or args.sort is None:
        def keyfunc(x):
            if not x.micro_image_names:
                return str()
            return sorted(x.micro_image_names)[-1]
    else:
        assert(args.sort == 'unsorted')

        def keyfunc(x):
            return 0

    now = datetime.datetime.now()

    for mole in sorted(_yield_mole_dirs('.', args), key=keyfunc):
        mole_data = {
            'relpath': mole.refrelpath,
            'lastmicro': '',
            'lastmicro_age_days': '',
            'id': mole.id,
        }
        if mole.micro_image_names:
            lastmicro = sorted(mole.micro_image_names)[-1]
            mole_data['lastmicro'] = lastmicro
            lastmicrodtstring = lastmicro.split('.', 1)[0]
            lastmicrodt = datetime.datetime.strptime(
                lastmicrodtstring, '%Y%m%dT%H%M%S')
            age = now - lastmicrodt
            mole_data['lastmicro_age_days'] = age.days

        print(args.format.format(**mole_data))


def _yield_mole_dirs(rootpath, args):
    for mole in mel.micro.fs.yield_moles(rootpath):

        if args.only_no_micro and mole.micro_image_names:
            continue

        if args.without_assistance and mole.need_assistance:
            continue

        if args.with_assistance and not mole.need_assistance:
            continue

        yield mole
