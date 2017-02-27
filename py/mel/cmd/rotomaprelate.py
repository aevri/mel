"""Guess the relationships between moles in a rotomap."""


import json

import mel.lib.math
import mel.rotomap.relate


def setup_parser(parser):
    parser.add_argument(
        'FROM',
        type=str,
        help="Path of the 'from' rotomap json file.")
    parser.add_argument(
        'TO',
        type=str,
        nargs='+',
        help="Paths of the 'to' rotomap json files.")
    parser.add_argument(
        '--loop',
        action='store_true',
        help="Apply the relation as if the files specify a complete loop.")


def process_args(args):
    process_files(args.FROM, args.TO)
    if args.loop:
        process_files(args.FROM, reversed(args.TO))


def process_files(from_path, to_path_list):
    files = [from_path]
    files.extend(to_path_list)
    for from_path, to_path in pairwise(files):
        process_pair(from_path, to_path)


def pairwise(iterable):
    return zip(iterable, iterable[1:])


def process_pair(from_path, to_path):

    from_moles = load_json(from_path)
    to_moles = load_json(to_path)

    pairs = mel.rotomap.relate.best_offset_theory(
        from_moles, to_moles)

    if pairs is None:
        return

    for mole in to_moles:
        for p in pairs:
            if p[0] and p[1]:
                if mole['uuid'] == p[1]:
                    mole['uuid'] = p[0]
                    break

    with open(to_path, 'w') as f:
        json.dump(
            to_moles,
            f,
            indent=4,
            separators=(',', ': '),
            sort_keys=True)

        # There's no newline after dump(), add one here for happier viewing
        print(file=f)


def load_json(path):
    with open(path) as f:
        return json.load(f)
