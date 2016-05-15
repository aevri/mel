"""List the moles in a mole catalog."""


import os


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
        choices=['lastmicro'])


def process_args(args):
    if not args.sort:
        for mole in _yield_mole_dirs('.', args):
            print(mole.catalog_relative_path)
    else:
        keyfunc = None

        if args.sort == 'lastmicro':
            def keyfunc(x):
                if not x.micro_filenames:
                    return str()
                return sorted(x.micro_filenames)[-1]

        for mole in sorted(_yield_mole_dirs('.', args), key=keyfunc):
            print(mole.catalog_relative_path)


class _Mole(object):

    def __init__(self, catalog_relative_path, micro_filenames):
        super(_Mole, self).__init__()
        self.catalog_relative_path = catalog_relative_path
        self.micro_filenames = micro_filenames


def _yield_mole_dirs(rootpath, args):
    for path, dirs, files in os.walk(rootpath):

        this_dirname = os.path.basename(path)

        if this_dirname == '__micro__':
            continue

        catalog_relpath = os.path.relpath(path, rootpath)

        # ignore directories with no files
        if not files:
            continue

        # ignore dot-directories in the root, like '.git'
        if catalog_relpath.startswith('.'):
            continue

        unknown_dirs = set(dirs)

        if args.only_no_micro and '__micro__' in unknown_dirs:
            if os.listdir(os.path.join(path, '__micro__')):
                continue

        if args.without_assistance and '__need_assistance__' in files:
            continue

        if args.with_assistance and '__need_assistance__' not in files:
            continue

        micro_filenames = []
        if '__micro__' in unknown_dirs:
            unknown_dirs.discard('__micro__')
            micro_filenames = os.listdir(os.path.join(path, '__micro__'))

        # mole clusters have a picture and all the moles as child dirs, ignore
        if unknown_dirs:
            continue

        yield _Mole(
            catalog_relpath,
            micro_filenames)
