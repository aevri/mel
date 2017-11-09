"""Guess which mole is which in a rotomap image."""

import collections
import copy
import itertools
import json
import uuid

import numpy

import mel.rotomap.moles
import mel.rotomap.identify


# TODO: tackle large amounts of unknowns in chunks, grouped by proximity to
# known moles. Perhaps the idea of 'adjacency' is important.

# TODO: define cost function based on nearest neighbour (later: neighbours
# within proportion of nearest). Provide lower-bound estimate by looking at all
# possibilities for nearest neighbour and their lowest cost for this mole.

# TODO: consider that we don't have a real cost function and that everything is
# currently an estimate instead.
#   o Think about what a real cost function would look like
#       o Try implementing a model where the cost of a mole guess is determined
#       finally, only by its nearest neighbour. Costs can be updated as we go.
#       This might give us muliple levels of estimates -
#       (impossible est., nearest known guesses, nearest guesses).
#   o Think about what algorithms could deal with just an estimate
#   o Perhaps the estimate / cost should be identical for identical states,
#   history maybe should not matter.

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

    possible_uuid_set = set(uuid_to_frameposlist.values())
    for frame in target_frames:
        if args.verbose:
            print('Processing', frame.path, '..')

        uuid_to_pos = mel.rotomap.identify.frame_to_uuid_to_pos(frame)
        canonical_uuid_set = set(
            mole['uuid']
            for mole in frame.moles
            if mole[mel.rotomap.moles.KEY_IS_CONFIRMED]
        )
        guesser = mel.rotomap.identify.PosGuesser(
            uuid_to_pos,
            warm_classifier,
            canonical_uuid_set,
            possible_uuid_set)

        # cost, old_to_new = guess_old_to_new(
        #     uuid_to_pos, cold_classifier, warm_classifier, canonical_uuid_set)
        cost, old_to_new = mel.rotomap.identify.best_match_combination(
            guesser, max_iterations=1*10**5)

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


def guess_old_to_new(
        uuid_to_pos, cold_classifier, warm_classifier, canonical_uuid_set):

    max_unknowns = 8

    remap_stack = []

    uuid_to_pos = dict(uuid_to_pos)
    reduced_uuid_to_pos = dict(uuid_to_pos)
    canonical_uuid_set = set(canonical_uuid_set)

    final_run = False
    while not final_run:

        unknown_uuids = set(uuid_to_pos.keys()) - canonical_uuid_set
        if len(unknown_uuids) > max_unknowns:
            accept_uuids = list(unknown_uuids)[:max_unknowns]
            reduced_uuid_to_pos = {
                key: value
                for key, value in uuid_to_pos.items()
                if key in canonical_uuid_set or key in accept_uuids
            }
        else:
            reduced_uuid_to_pos = uuid_to_pos
            final_run = True

        guesser = mel.rotomap.identify.Guesser(
            reduced_uuid_to_pos,
            cold_classifier,
            warm_classifier,
            canonical_uuid_set)

        cost, old_to_new = mel.rotomap.identify.best_match_combination(guesser)

        print('Num canon pre:', len(canonical_uuid_set))
        canonical_uuid_set = set(
            new for old, new in old_to_new.items()
            if new is not None
        )

        print('Num canon:', len(canonical_uuid_set))

        remap_stack.append({})
        new_uuid_to_pos = {}
        for old_uuid, pos in uuid_to_pos.items():
            if old_uuid not in old_to_new:
                if old_uuid in new_uuid_to_pos:
                    uuid_ = uuid.uuid4().hex
                    remap_stack[-1][uuid_] = old_uuid
                    new_uuid_to_pos[uuid_] = pos
                else:
                    remap_stack[-1][old_uuid] = old_uuid
                    new_uuid_to_pos[old_uuid] = pos
            else:
                new_uuid = old_to_new[old_uuid]
                if new_uuid in new_uuid_to_pos:
                    dup_pos = new_uuid_to_pos[new_uuid]
                    dup_uuid = uuid.uuid4().hex
                    remap_stack[-1][dup_uuid] = new_uuid
                    new_uuid_to_pos[dup_uuid] = dup_pos
                remap_stack[-1][new_uuid] = old_uuid
                new_uuid_to_pos[new_uuid] = pos
        uuid_to_pos = new_uuid_to_pos

    for remap in reversed(remap_stack[:-1]):
        old_to_new = {
            remap[old]: new
            for old, new in old_to_new.items()
        }

    return cost, old_to_new


def _process_args(args):
    cost, old_to_new = mel.rotomap.identify.best_match_combination(Guesser())

    import pprint
    print('Cost', cost)
    pprint.pprint(old_to_new)


class Guesser():

    def __init__(self):
        pass

    def initial_state(self):
        slots = 24
        state = {}
        for i in range(slots):
            state[str(i)] = None
        return (2 ** slots, slots, 2), state

    def yield_next_states(self, total_cost, state):
        unfilled = tuple(k for k, v in state.items() if v is None)
        guess = 2 ** (len(unfilled) - 1)
        num_unfilled = len(unfilled) - 1
        for u in unfilled:
            yield (
                (total_cost[2] * 2 * guess, num_unfilled, total_cost[2] * 2),
                updated(state, u, u)
            )
            yield (
                (total_cost[2] * 100 * guess, num_unfilled, total_cost[2] * 100),
                updated(state, u, u + '_wrong')
            )


def updated(d, key, value):
    new_d = dict(d)
    new_d[key] = value
    return new_d
