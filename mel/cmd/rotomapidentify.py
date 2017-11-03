"""Guess which mole is which in a rotomap image."""

import collections
import copy
import itertools
import json

import numpy

import mel.rotomap.moles
import mel.rotomap.identify


def setup_parser(parser):
    parser.add_argument(
        'TARGET_IMAGE',
        nargs='+',
        help="Paths to images to identify.")
    parser.add_argument(
        '--source',
        '-s',
        metavar='DIRECTORY',
        type=mel.rotomap.moles.make_argparse_rotomap_directory,
        nargs='+',
        default=[],
        help="Paths to rotomaps to read for reference.")
    parser.add_argument(
        '--cache',
        '-c',
        metavar='PATH',
        help="Path to cache training data. Will write if sources suppled, "
             "will read otherwise."
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help="Print information about the processing.")


def transformed_frames_to_uuid_frameposlist(uuid_to_frameposlist, pos_fn):
    out_uuid_to_frameposlist = {}
    for uuid_, frameposlist in uuid_to_frameposlist.items():
        out_frameposlist = []
        for frame, pos in frameposlist:
            out_frameposlist.append((frame, pos_fn(pos)))
        out_uuid_to_frameposlist[uuid_] = out_frameposlist

    return out_uuid_to_frameposlist


def save_frames_to_uuid_frameposlist(uuid_to_frameposlist, path):
    data = transformed_frames_to_uuid_frameposlist(uuid_to_frameposlist, list)
    with open(path, 'w') as f:
        json.dump(data, f)


def load_frames_to_uuid_frameposlist(path):
    with open(path) as f:
        data = json.load(f)
    return transformed_frames_to_uuid_frameposlist(data, numpy.array)


def process_args(args):

    if args.cache and not args.source:
        if args.verbose:
            print('Reading cache ..')
            uuid_to_frameposlist = load_frames_to_uuid_frameposlist(args.cache)
    else:
        if args.verbose:
            print('Loading from sources ..')

        training_frames = itertools.chain.from_iterable(
            x.yield_frames() for x in args.source)

        uuid_to_frameposlist = mel.rotomap.moles.frames_to_uuid_frameposlist(
            training_frames)

        if args.cache:
            if args.verbose:
                print('Writing cache ..')
                save_frames_to_uuid_frameposlist(
                    uuid_to_frameposlist, args.cache)

    target_frames = [
        mel.rotomap.moles.RotomapFrame(x) for x in args.TARGET_IMAGE
    ]

    if args.verbose:
        print('Training ..')

    # TODO: distinguish between canonical and non-canonical moles for training

    yrad = 0.1
    nrad = 1.2
    cold_classifier = mel.rotomap.identify.ColdGuessMoleClassifier(
        uuid_to_frameposlist, yrad, nrad)

    box_radius = 0.1
    warm_classifier = mel.rotomap.identify.MoleRelativeClassifier(
        uuid_to_frameposlist, box_radius)

    for frame in target_frames:
        if args.verbose:
            print('Processing', frame.path, '..')

        uuid_to_pos = mel.rotomap.identify.frame_to_uuid_to_pos(frame)
        canonical_uuid_set = set(
            mole['uuid']
            for mole in frame.moles
            if mole[mel.rotomap.moles.KEY_IS_CONFIRMED]
        )
        guesser = mel.rotomap.identify.Guesser(
            uuid_to_pos, cold_classifier, warm_classifier, canonical_uuid_set)

        cost, old_to_new = mel.rotomap.identify.best_match_combination(guesser)

        import pprint
        print('Cost', cost)
        pprint.pprint(old_to_new)

        new_id_set = set(old_to_new.values())
        new_moles = copy.deepcopy(frame.moles)
        for mole in new_moles:
            old_id = mole['uuid']
            new_id = old_to_new[old_id]
            if new_id is not None:
                mole['uuid'] = new_id
            elif old_id in new_id_set:
                raise Exception(f'{frame.path}: would duplicate {old_id}')

        mel.rotomap.moles.save_image_moles(new_moles, str(frame.path))
