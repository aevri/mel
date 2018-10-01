"""Benchmark the accuracy of rotomap.relate across a rotomap."""

import numpy

import mel.lib.image
import mel.lib.math
import mel.lib.moleimaging

import mel.rotomap.detectmoles
import mel.rotomap.moles
import mel.rotomap.relate


def setup_parser(parser):
    parser.add_argument(
        'FROM', type=str, help="Path of the 'from' rotomap json file."
    )
    parser.add_argument(
        'TO', type=str, nargs='+', help="Paths of the 'to' rotomap json files."
    )
    parser.add_argument(
        '--loop',
        action='store_true',
        help="Apply the relation as if the files specify a complete loop.",
    )
    parser.add_argument(
        '--error-distance',
        default=5,
        type=int,
        help='Consider guesses this far from their target to be misses / '
        'errors.',
    )
    parser.add_argument('--verbose', '-v', action='count', default=0)


def process_args(args):
    hits, misses = process_files(args.FROM, args.TO, args)

    if args.loop:
        l_hits, l_misses = process_files(args.FROM, reversed(args.TO), args)
        hits += l_hits
        misses += l_misses

    if not misses and not hits:
        print('Nothing to test.')
        return

    hit_pct = 100 * (hits / (misses + hits))
    print(f'{hits} hits, {misses} misses, {hit_pct:.2f}% success.')


def process_files(from_path, to_path_list, args):
    files = [from_path]
    files.extend(to_path_list)
    total_hits = 0
    total_misses = 0
    for from_path, to_path in pairwise(files):
        hits, misses = process_combinations(from_path, to_path, args)
        total_hits += hits
        total_misses += misses
    return total_hits, total_misses


def pairwise(iterable):
    return zip(iterable, iterable[1:])


def process_combinations(from_path, to_path, args):

    from_moles = mel.rotomap.moles.load_image_moles(from_path)
    to_moles = mel.rotomap.moles.load_image_moles(to_path)

    if not from_moles or not to_moles:
        return 0, 0

    to_image = mel.lib.image.load_image(to_path)

    num_tests = 0
    num_hits = 0

    from_dict = mel.rotomap.relate.mole_list_to_uuid_dict(from_moles)
    to_dict = mel.rotomap.relate.mole_list_to_uuid_dict(to_moles)
    from_set = set(from_dict.keys())
    to_set = set(to_dict.keys())
    in_both = from_set & to_set

    for target_uuid in in_both:
        num_tests += 1

        to_moles_without_target = dict(to_dict)
        target_mole = to_moles_without_target[target_uuid]
        target_pos = mel.rotomap.moles.mole_to_point(target_mole)
        del to_moles_without_target[target_uuid]
        to_moles_without_target = list(to_moles_without_target.values())

        guess_pos = guess_mole_pos(
            target_uuid, from_moles, to_moles_without_target, to_image
        )

        if guess_pos is not None:
            distance = mel.lib.math.distance_2d(guess_pos, target_pos)
            if distance < args.error_distance:
                num_hits += 1
                if args.verbose >= 2:
                    print(
                        'Hit: ({} -> {}) for {}, distance {}.'.format(
                            from_path, to_path, target_uuid, distance
                        )
                    )
            else:
                if args.verbose >= 2:
                    print(
                        'Miss: ({} -> {}) for {}, distance {}.'.format(
                            from_path, to_path, target_uuid, distance
                        )
                    )
        else:
            if args.verbose >= 2:
                print(
                    'Could not guess: ({} -> {}) for {}.'.format(
                        from_path, to_path, target_uuid
                    )
                )

    num_misses = num_tests - num_hits

    if args.verbose >= 1:
        if num_misses:
            print(
                'Flawed mapping: ({} -> {}); {} hits, {} misses.'.format(
                    from_path, to_path, num_hits, num_misses
                )
            )
        else:
            print('Flawless mapping: ({} -> {})'.format(from_path, to_path))

    return num_hits, num_misses


def guess_mole_pos(target_uuid, from_moles, to_moles, to_image):

    # guess_pos = mel.rotomap.relate.guess_mole_pos(
    #     target_uuid, from_moles, to_moles)

    guess_pos = mel.rotomap.relate.guess_mole_pos_pair_method(
        target_uuid, from_moles, to_moles
    )

    if guess_pos is not None:
        guess_pos = snap_to_mole_findellipse(to_image, guess_pos)
        # guess_pos = snap_to_mole_detectmoles(to_image, guess_pos)
        pass

    return guess_pos


def snap_to_mole_findellipse(image, guess_pos, radius=50):
    _MAGIC_MOLE_FINDER_RADIUS = radius
    ellipse = mel.lib.moleimaging.find_mole_ellipse(
        image, guess_pos, _MAGIC_MOLE_FINDER_RADIUS
    )
    if ellipse is not None:
        guess_pos = numpy.array(ellipse[0], dtype=int)
    return guess_pos


# def snap_to_mole_detectmoles(image, guess_pos):
#     image = image.copy()
#     _MAGIC_MOLE_FINDER_RADIUS = 50
#     guessed_moles = find_moles(image, guess_pos, _MAGIC_MOLE_FINDER_RADIUS)
#     if guessed_moles:
#         index, dist = mel.rotomap.moles.nearest_mole_index_distance(
#             guessed_moles, *guess_pos)
#         mole = guessed_moles[index]
#         guess_pos = (mole['x'], mole['y'])
#     return guess_pos


# def find_moles(original, centre, radius):
#     lefttop = centre - (radius, radius)
#     rightbottom = centre + (radius + 1, radius + 1)
#     original = mel.lib.image.slice_square_or_none(
#         original, lefttop, rightbottom)
#     if original is None:
#         return None
#     image = original[:]
#     guessed_moles = mel.rotomap.detectmoles.moles(image, mask=None)
#     for mole in guessed_moles:
#         mole['x'] = mole['x'] + lefttop[0]
#         mole['y'] = mole['y'] + lefttop[1]
#     return guessed_moles
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
