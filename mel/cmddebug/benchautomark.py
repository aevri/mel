"""Benchmark the accuracy of a set of rotomaps vs a reference."""

import numpy

import mel.cmd.error
import mel.lib.common
import mel.lib.image
import mel.rotomap.detectmoles
import mel.rotomap.mask
import mel.rotomap.moles


def setup_parser(parser):

    parser.add_argument(
        "FROM_FRAMES",
        type=mel.rotomap.moles.make_argparse_image_moles_tree,
        help="Path of the 'reference' rotomap or image.",
    )
    parser.add_argument(
        "TO_FRAMES",
        type=mel.rotomap.moles.make_argparse_image_moles_tree,
        help="Path of the directory of rotomap or image.",
    )
    parser.add_argument(
        "--error-distance",
        default=5,
        type=int,
        help="Consider guesses this far from their target to be misses / "
        "errors.",
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)


def process_args(args):
    from_to_pairs = _pair_off_inputs(args.FROM_FRAMES, args.TO_FRAMES)
    num_matches = 0
    num_missing = 0
    num_added = 0
    for common_path, from_moles, to_moles in from_to_pairs:
        matches, missing, added = match_moles(
            from_moles, to_moles, args.error_distance
        )
        if args.verbose > 1:
            print_items(common_path, matches, "MATCH")
            print_items(common_path, missing, "MISSING")
            print_items(common_path, added, "ADDED")
        if args.verbose > 0:
            print(f"{common_path}: {len(matches)} matched")
            print(f"{common_path}: {len(missing)} missing")
            print(f"{common_path}: {len(added)} added")

        num_matches += len(matches)
        num_missing += len(missing)
        num_added += len(added)

    print(f"{num_matches} matched")
    print(f"{num_missing} missing")
    print(f"{num_added} added")
    precision = num_matches / (num_added + num_matches)
    recall = num_matches / (num_missing + num_matches)
    print(f"{precision * 100:0.1f}% precision")
    print(f"{recall * 100:0.1f}% recall")


def _pair_off_inputs(from_, to):
    for from_frame, to_frame in _zip_samelen(from_, to):
        from_image_path, from_moles = from_frame
        to_image_path, to_moles = to_frame
        common_path = _common_path(from_image_path, to_image_path)
        yield common_path, from_moles, to_moles


def _common_path(from_path, to_path):
    common = []
    for char_from, char_to in zip(
        reversed(str(from_path)), reversed(str(to_path))
    ):
        if char_from != char_to:
            break
        common.insert(0, char_from)
    return "".join(common)


def _zip_samelen(*args):
    iterators = [iter(it) for it in args]
    if not iterators:
        return
    while True:
        values = []
        num_stopped = 0
        for i, it in enumerate(iterators):
            try:
                values.append(next(it))
            except StopIteration:
                num_stopped += 1
        if num_stopped:
            if num_stopped != len(iterators):
                raise ValueError("Iterators must be the same length.")
            return
        yield tuple(values)


def print_items(common_path, items, label):
    for i in items:
        print(f"{label}: {common_path}: {i}")


def match_moles(from_moles, to_moles, error_distance):

    if from_moles and not to_moles:
        return [], [m["uuid"] for m in from_moles], []
    elif not from_moles and to_moles:
        return [], [], [m["uuid"] for m in to_moles]
    elif not from_moles and not to_moles:
        return [], [], []

    from_pos_vec = mel.rotomap.moles.mole_list_to_pointvec(from_moles)
    to_pos_vec = mel.rotomap.moles.mole_list_to_pointvec(to_moles)

    vec_matches, vec_missing, vec_added = _match_pos_vecs(
        from_pos_vec, to_pos_vec, error_distance
    )

    matches = [
        (from_moles[from_i]["uuid"], to_moles[to_i]["uuid"])
        for from_i, to_i in vec_matches
    ]
    missing = [from_moles[from_i]["uuid"] for from_i in vec_missing]
    added = [to_moles[to_i]["uuid"] for to_i in vec_added]

    return matches, missing, added


def _match_pos_vecs(from_pos_vec, to_pos_vec, error_distance):
    max_sqdist = error_distance ** 2

    # pylint: disable=no-member
    # Avoid this error:
    #
    #   E1101: Function 'subtract' has no 'outer' member (no-member)
    #
    distmatx = numpy.subtract.outer(from_pos_vec[:, 0], to_pos_vec[:, 0])
    distmaty = numpy.subtract.outer(from_pos_vec[:, 1], to_pos_vec[:, 1])
    # pylint: enable=no-member

    sqdistmat = numpy.square(distmatx) + numpy.square(distmaty)

    assert sqdistmat.shape == (len(from_pos_vec), len(to_pos_vec))

    # print(sqdistmat)

    matindex_to_fromindex = list(range(len(from_pos_vec)))
    matindex_to_toindex = list(range(len(to_pos_vec)))

    matches = []
    while _array_nonempty(sqdistmat):
        nextmin_i = numpy.argmin(sqdistmat)  # Gives us the 'flat index'.
        from_i, to_i = numpy.unravel_index(nextmin_i, sqdistmat.shape)
        nextmin = sqdistmat[from_i, to_i]
        if nextmin > max_sqdist:
            break
        # It seems pylint doesn't know about numpy indexing.
        # pylint: disable=invalid-sequence-index
        matches.append(
            (matindex_to_fromindex[from_i], matindex_to_toindex[to_i])
        )
        del matindex_to_fromindex[from_i]
        del matindex_to_toindex[to_i]
        sqdistmat = numpy.delete(sqdistmat, from_i, axis=0)
        sqdistmat = numpy.delete(sqdistmat, to_i, axis=1)
        # pylint: enable=invalid-sequence-index

    # print()

    missing = matindex_to_fromindex
    added = matindex_to_toindex

    return matches, missing, added


def _array_nonempty(numpy_array):
    return all(numpy_array.shape)


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
