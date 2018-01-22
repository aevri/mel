"""Guess which mole is which in a rotomap image."""

import copy
import itertools
import json

import numpy

import mel.rotomap.moles
import mel.rotomap.identify


# TODO: Determine a sensible upper-bound for early discard of hopeless avenues.

# TODO: Determine which groups of moles are 'closed', in that they are not
# considered neighbours of any other mole groups. We want to create bridges
# across these groups so that they may cross-check eachother and provide extra
# constraints to satisfy. This should increase the quality of the results.

def setup_parser(parser):
    parser.add_argument(
        '--target',
        '-t',
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
            training_frames, canonical_only=True)

        if args.cache:
            if args.verbose:
                print('Writing cache ..')
            save_frames_to_uuid_frameposlist(
                uuid_to_frameposlist, args.cache)

    if not args.target:
        if not args.cache:
            print('Nothing to do.')
        return

    target_frames = [
        mel.rotomap.moles.RotomapFrame(x) for x in args.target
    ]

    if args.verbose:
        print('Training ..')

    # yrad = 0.1
    # nrad = 1.2
    # cold_classifier = mel.rotomap.identify.ColdGuessMoleClassifier(
    #     uuid_to_frameposlist, yrad, nrad)

    box_radius = 0.1
    warm_classifier = mel.rotomap.identify.MoleRelativeClassifier(
        uuid_to_frameposlist, box_radius)

    possible_uuid_set = set(uuid_to_frameposlist.keys())
    for frame in target_frames:
        if args.verbose:
            print('Processing', frame.path, '..')

        uuid_to_pos = mel.rotomap.identify.frame_to_uuid_to_pos(frame)
        canonical_uuid_set = set(
            mole['uuid']
            for mole in frame.moles
            if mole[mel.rotomap.moles.KEY_IS_CONFIRMED]
        )
        helper = mel.rotomap.identify.PosGuesserHelper(
            uuid_to_pos, warm_classifier)
        guesser = mel.rotomap.identify.PosGuesser(
            tuple(uuid_to_pos.keys()),
            helper,
            canonical_uuid_set,
            possible_uuid_set)

        # cost, old_to_new = guess_old_to_new(
        # uuid_to_pos, cold_classifier, warm_classifier, canonical_uuid_set)
        cost, old_to_new = mel.rotomap.identify.best_match_combination(
            guesser, max_iterations=1 * 10**5)

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
# -----------------------------------------------------------------------------
# Copyright (C) 2017 Angelos Evripiotis.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------ END-OF-FILE ----------------------------------
