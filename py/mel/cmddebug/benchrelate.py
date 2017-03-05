"""Benchmark the accuracy of rotomap.relate across a rotomap."""


import mel.lib.math
import mel.rotomap.moles
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

    from_moles = mel.rotomap.moles.load_image_moles(from_path)
    to_moles = mel.rotomap.moles.load_image_moles(to_path)

    if not from_moles or not to_moles:
        return

    expected_theory = make_default_map_theory(from_moles, to_moles)

    offset_theory = mel.rotomap.relate.best_offset_theory(
        from_moles, to_moles)

    expected_theory_set = set(expected_theory)
    offset_theory_set = set(offset_theory)

    for from_uuid, to_uuid in expected_theory_set ^ offset_theory_set:
        print('False', format_mapping(from_path, to_path, from_uuid, to_uuid))
    for from_uuid, to_uuid in expected_theory_set & offset_theory_set:
        print('True', format_mapping(from_path, to_path, from_uuid, to_uuid))


def format_mapping(from_path, to_path, from_uuid, to_uuid):
    fmt_str = '{}: ({} -> {}), ({} -> {})'
    if from_uuid is None or to_uuid is None:
        assert(from_uuid or to_uuid)
        return fmt_str.format(
            'negative', from_path, to_path, from_uuid, to_uuid)
    else:
        return fmt_str.format(
            'positive', from_path, to_path, from_uuid, to_uuid)


def make_default_map_theory(from_moles, to_moles):

    from_dict = mel.rotomap.relate.mole_list_to_uuid_dict(from_moles)
    to_dict = mel.rotomap.relate.mole_list_to_uuid_dict(to_moles)
    from_set = set(from_dict.keys())
    to_set = set(to_dict.keys())

    from_only = from_set - to_set
    to_only = to_set - from_set
    in_both = from_set & to_set

    theory = []
    theory.extend((u, None) for u in from_only)
    theory.extend((None, u) for u in to_only)
    theory.extend((u, u) for u in in_both)

    return theory
