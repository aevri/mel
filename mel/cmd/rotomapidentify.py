"""Guess which mole is which in a rotomap image."""

import copy
import itertools
import sys

import mel.rotomap.identify
import mel.rotomap.lowerbound
import mel.rotomap.moles


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
        required=True,
        help="Paths to images to identify.",
    )
    parser.add_argument(
        '--source',
        '-s',
        metavar='DIRECTORY',
        type=mel.rotomap.moles.make_argparse_rotomap_directory,
        nargs='+',
        default=[],
        help="Paths to rotomaps to read for reference.",
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help="Print information about the processing.",
    )


def process_args(args):

    if args.verbose:
        print('Loading from sources ..')

    training_frames = itertools.chain.from_iterable(
        x.yield_frames() for x in args.source
    )

    uuid_to_frameposlist = mel.rotomap.moles.frames_to_uuid_frameposlist(
        training_frames, canonical_only=True
    )

    # We're going to use these uuids a lot as keys in dicts. The docs for
    # sys.intern say that it can speed things up in that case.
    uuid_to_frameposlist = {
        sys.intern(k): v for k, v in uuid_to_frameposlist.items()
    }

    target_frames = [mel.rotomap.moles.RotomapFrame(x) for x in args.target]

    if args.verbose:
        print(f"Got {len(target_frames)} sources.")
        print('Training ..')

    box_radius = 0.2
    warm_classifier = mel.rotomap.identify.MoleRelativeClassifier(
        uuid_to_frameposlist, box_radius
    )

    possible_uuid_set = frozenset(uuid_to_frameposlist.keys())
    for frame in target_frames:
        if args.verbose:
            print('Processing', frame.path, '..')

        uuid_to_pos = mel.rotomap.identify.frame_to_uuid_to_pos(frame)
        canonical_uuid_set = frozenset(
            mole['uuid']
            for mole in frame.moles
            if mole[mel.rotomap.moles.KEY_IS_CONFIRMED]
        )

        # Note that the order of adding UUIDs is important, in other places
        # we'll be relying on using '<' on ids to determine if an id refers to
        # a canonical mole.
        trans = mel.rotomap.identify.UuidToIndexTranslator()
        trans.add_uuids(canonical_uuid_set)
        num_canonicals = trans.num_uuids()
        trans.add_uuids_with_imposters(uuid_to_pos.keys())
        num_locations = trans.num_uuids() - trans.num_imposters()
        trans.add_uuids(possible_uuid_set)
        num_identities = trans.num_uuids()

        positions = trans.uuid_dict_to_index_tuple(uuid_to_pos, num_locations)

        calc_guesses = mel.rotomap.identify.make_calc_guesses(
            positions, trans, warm_classifier
        )
        predictors = mel.rotomap.identify.predictors(positions, num_canonicals)

        bounder = BounderWrapper(
            tuple(predictor_loc for (_, predictor_loc) in predictors),
            calc_guesses,
            num_identities,
            num_canonicals,
        )

        if args.verbose:
            print(f"Got {num_identities} identities.")
            print(f"Got {num_canonicals} canonicals.")

        guesser = mel.rotomap.identify.PosGuesser(
            num_locations, predictors, bounder, num_canonicals, num_identities
        )

        cost, old_to_new = mel.rotomap.identify.best_match_combination(
            guesser, max_iterations=1 * 10 ** 5
        )

        old_to_new = {
            trans.uuid_(old): trans.uuid_(new)
            for old, new in enumerate(old_to_new)
        }

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


class BounderWrapper:
    """Extend the capabilities of the C++ bounder, add 'possible_guesses'.

    If this turns out to be enough of a bottleneck, we can move the
    implementation to C++ too.
    """

    def __init__(
        self, predictors, calc_guesses, num_identities, num_canonicals
    ):
        self._predictors = predictors
        self._calc_guesses = calc_guesses
        self._bounder = mel.rotomap.lowerbound.Bounder(
            predictors, calc_guesses, num_identities, num_canonicals
        )

    def lower_bound(self, state):
        return self._bounder.lower_bound(state)

    def possible_guesses(
        self, state, possible_ident_set, already_taken, guess_loc
    ):
        predictor_loc = self._predictors[guess_loc]
        predictor_ident = state[predictor_loc]

        possibles = []
        if predictor_ident is not None:
            possibles = self._calc_guesses(
                (predictor_loc, predictor_ident), guess_loc
            )
        else:
            for predictor_ident in possible_ident_set - already_taken:
                if predictor_ident == guess_loc:
                    continue
                possibles.extend(
                    self._calc_guesses(
                        (predictor_loc, predictor_ident), guess_loc
                    )
                )

        return set([ident for ident, _ in possibles]) - already_taken


# -----------------------------------------------------------------------------
# Copyright (C) 2018 Angelos Evripiotis.
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
