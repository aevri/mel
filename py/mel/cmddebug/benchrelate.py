"""Benchmark the accuracy of rotomap.relate across a rotomap."""

import copy
import itertools
import uuid

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
    parser.add_argument(
        '--reset-uuids',
        type=int,
        default=0,
        help="Reset this number of uuids in the destination. Iterate over all "
             "combinations.")


def process_args(args):
    process_files(args.FROM, args.TO, args)
    if args.loop:
        process_files(args.FROM, reversed(args.TO), args)


def process_files(from_path, to_path_list, args):
    files = [from_path]
    files.extend(to_path_list)
    for from_path, to_path in pairwise(files):
        process_combinations(from_path, to_path, args)


def pairwise(iterable):
    return zip(iterable, iterable[1:])


def process_combinations(from_path, to_path, args):

    from_moles = mel.rotomap.moles.load_image_moles(from_path)
    to_moles = mel.rotomap.moles.load_image_moles(to_path)

    if not from_moles or not to_moles:
        return

    expected_theory = make_default_map_theory(from_moles, to_moles)

    for params in yield_reset_combinations(
            from_moles, to_moles, expected_theory, args.reset_uuids):
        process_pair(from_path, to_path, *params)


def yield_reset_combinations(from_moles, to_moles, expected_theory, num_reset):

    if num_reset == 0:
        yield from_moles, to_moles, expected_theory
        return

    to_uuids = set(x['uuid'] for x in to_moles)
    num_reset = min(len(to_uuids), num_reset)

    for uuids in itertools.combinations(to_uuids, num_reset):
        new_to_moles = copy.deepcopy(to_moles)
        new_theory = copy.deepcopy(expected_theory)
        for u in uuids:
            new_u = uuid.uuid4().hex
            for mole in new_to_moles:
                if mole['uuid'] == u:
                    mole['uuid'] = new_u

            def remapped_theory(x, y):
                return (x, y) if y != u else (x, new_u)

            new_theory = [
                remapped_theory(x, y) for x, y in new_theory
            ]
        yield from_moles, new_to_moles, new_theory


def process_pair(from_path, to_path, from_moles, to_moles, expected_theory):

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
