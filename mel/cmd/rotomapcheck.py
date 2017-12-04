"""Check that rotomaps are consistent."""

import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        'ROTOMAP',
        type=mel.rotomap.moles.make_argparse_rotomap_directory,
        nargs='+',
        help="Paths to the rotomap directories to check.")
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help="Print information about the processing.")


def process_args(args):
    any_problems = False
    for dir_ in args.ROTOMAP:
        if args.verbose:
            print(f'Checking rotomap: {dir_.path}')
        for image_path, mole_list in dir_.yield_mole_lists():
            if args.verbose:
                print(f'Checking image: {image_path}')
            if print_mole_errors(image_path, mole_list):
                any_problems = True
    return not any_problems


def print_mole_errors(image_path, mole_list):
    """Print any errors found in the mole list, return False if none found.

    :image_path: a printable reference to the image the moles belong to.
    :mole_list: a list of moles, which are dictionaries.
    :returns: True if any errors found, False otherwise.

    """
    any_problems = False
    uuids = set()
    for mole in mole_list:
        u = mole['uuid']
        if u in uuids:
            print(f'{image_path}: duplicate uuid "{u}"')
            any_problems = True
        uuids.add(u)

    return any_problems
