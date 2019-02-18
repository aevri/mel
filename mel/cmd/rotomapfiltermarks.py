"""Automatically remove marked regions that are probably not moles.

This is experimental functionality and results may be variable.

In order to create the 'model file' and 'dataconfig file' required by this
command, you will need to have tagged a large number of moles by hand. You can
then create a dataset with the 'rotomap photoset' command. The first lesson in
the fast ai deep learning course will teach you to generate the model file:
"http://course.fast.ai/".

Please consule the fastai library documentation for how to generate the
dataconfig file.

Please install the 'fastai' python library and its dependencies in order for
this to work: https://pypi.org/project/fastai/

This has been tested with version 1.0.39 of fastai, as supplied by conda.
"""

import os
import sys

import mel.lib.fs

import mel.rotomap.moles

import cv2

import numpy

# The architecture we are using is 'resnet', which has a final layer size of
# 7 by 7. The image will go through a progressive series of halvings as it
# passes through the layers.
#
# Therefore, we pick a number from the series "7 * (2 ** i)" as the length of
# each side. In this case we pick "i = 5", so the size is 224.
#
# This seems to be a common magic number for input image size, for other
# architectures too.
#
# _HALF_SIZE = 112

# After experimentation, it seems that 56 is a good full-size. This represents
# a good trade-off between running time of this command, and the quality of the
# results.
_HALF_SIZE = 28


def setup_parser(parser):

    parser.add_argument(
        'FRAMES',
        type=mel.rotomap.moles.make_argparse_image_moles,
        nargs='+',
        help="Path to rotomap or image to filter.",
    )

    parser.add_argument(
        '--classifier-dir',
        help="Path to the classifier base dir, relative to the root of the "
        "mel repo. Names are relative to this."
        f"Defaults to {mel.lib.fs.DEFAULT_CLASSIFIER_PATH}.",
        default=mel.lib.fs.DEFAULT_CLASSIFIER_PATH,
    )

    parser.add_argument(
        '--model-name',
        help="Name of the model to use, relative to the classifier dir. "
        f"Defaults to {mel.lib.fs.DEFAULT_MOLE_MARK_MODEL_NAME}.",
        default=mel.lib.fs.DEFAULT_MOLE_MARK_MODEL_NAME,
    )

    parser.add_argument(
        '--dataconfig-name',
        help="Name of the dataconfig to use, relative to the classifier dir. "
        f"Defaults to {mel.lib.fs.DEFAULT_MOLE_MARK_DATACONFIG_NAME}.",
        default=mel.lib.fs.DEFAULT_MOLE_MARK_DATACONFIG_NAME,
    )

    parser.add_argument(
        '--dry-run',
        '-n',
        action='store_true',
        help="Don't save results of processing, just print.",
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help="Print information about the processing.",
    )

    parser.add_argument(
        '--include-canonical',
        action='store_true',
        help="Don't excluded canonical moles from processing.",
    )


def process_args(args):
    try:
        melroot = mel.lib.fs.find_melroot()
    except mel.lib.fs.NoMelrootError:
        print('Not in a mel repo, could not find melroot', file=sys.stderr)
        return 1

    if args.verbose:
        print("Initialising classifier .. ", end='', file=sys.stderr)
    classifier_path = melroot / args.classifier_dir
    is_mole = make_is_mole_func(
        classifier_path,
        args.dataconfig_name,
        classifier_path / args.model_name)
    if args.verbose:
        print("done", file=sys.stderr)
    for image_mole_iter in args.FRAMES:
        for image_path, moles in image_mole_iter:
            if args.verbose:
                print(image_path, file=sys.stderr)
            image = open_image_for_classifier(image_path)
            try:
                filtered_moles = filter_marks(
                    image_path,
                    is_mole,
                    image,
                    moles,
                    args.include_canonical
                )
            except Exception:
                raise Exception('Error while processing {}'.format(image_path))

            num_filtered = len(moles) - len(filtered_moles)
            if args.verbose:
                print(
                    f"Filtered {num_filtered} unlikely moles.",
                    file=sys.stderr)

            if not args.dry_run and num_filtered:
                mel.rotomap.moles.save_image_moles(filtered_moles, image_path)


def make_is_mole_func(metadata_dir, metadata_fname, model_path):

    # These imports can be very expensive, so we delay them as late as
    # possible.
    #
    # Also, they introduce a significant amount of extra dependencies. At
    # this point it is only experimental functionality, so importing late
    # like this allows folks not using it to avoid the extra burden.
    #
    # Since it's a little difficult to install fastai, exclude it from required
    # linting for now.
    # pylint: disable=import-error

    import fastai.vision

    data = fastai.vision.ImageDataBunch.load_empty(
        metadata_dir, metadata_fname)

    # pylint: enable=import-error

    # After experimentation, it seems that we can get away with using resnet18
    # as opposed to the deeper models. This sacrifices a little accuracy but
    # seems to improve the running time of this command somewhat.

    learn = fastai.vision.create_cnn(data, fastai.vision.models.resnet18)
    learn.load(model_path)

    def is_mole(image):
        x = fastai.vision.pil2tensor(image, numpy.float32)
        x.div_(255)
        x = fastai.vision.Image(x)
        raw_prediction = learn.predict(x)
        raw_label, classnum, class_scores = raw_prediction
        label = str(raw_label)
        if label not in ('moles', 'marks'):
            raise Exception('Unrecognised prediction result', label)

        if label == 'moles':
            if classnum != 1:
                raise Exception('mole class is not 1', classnum)
        elif label == 'marks':
            if classnum != 0:
                raise Exception('mark class is not 0', classnum)

        return (
            "moles" == label,
            int(class_scores[0] * 1000),
            int(class_scores[1] * 1000),
        )

    return is_mole


def filter_marks(
        image_path, is_mole, image, moles, include_canonical):
    """Return a list of moles with the unlikely ones filtered out."""

    filtered_moles = []
    for m in moles:
        # r = int(m['radius'])
        r = _HALF_SIZE

        r2 = r * 2
        x = m['x']
        y = m['y']

        is_currently_mole = m[mel.rotomap.moles.KEY_IS_CONFIRMED]
        current_label = "mole" if is_currently_mole else "non-mole"
        looks_like = m.get('looks_like', None)
        has_looks_like = looks_like is not None
        if has_looks_like:
            current_label = looks_like
        looks_like_s = "has_looks_like" if has_looks_like else "no_looks_like"

        print(
            f"{image_path}:{m['uuid']}:{x},{y}:"
            f"{looks_like_s}:{current_label}:",
            end='',
        )

        # Don't touch canonical moles.
        if not include_canonical and m[mel.rotomap.moles.KEY_IS_CONFIRMED]:
            print('SKIP:skipped because canonical')
            filtered_moles.append(m)
            continue

        image_fragment = image[
            y-r2:y+r2,
            x-r2:x+r2
        ]

        # TODO: decide what to do about fragments that overlap the image
        # boundary.
        if not all(image_fragment.shape):
            print('ERROR:skipped because zero radius')
            filtered_moles.append(m)
            continue

        is_probably_mole, score_a, score_b = is_mole(image_fragment)
        new_label = "mole" if is_probably_mole else "non-mole"

        if is_probably_mole:
            filtered_moles.append(m)

        print(f"{new_label}:{score_a}:{score_b}")

    return filtered_moles


def open_image_for_classifier(fn):
    flags = cv2.IMREAD_UNCHANGED + cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn):
        raise OSError('Is a directory: {}'.format(fn))

    try:
        im = cv2.imread(str(fn), flags)
        if im is None:
            raise OSError(f'File not recognized by opencv: {fn}')
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise OSError('Error handling image at: {}'.format(fn)) from e


# -----------------------------------------------------------------------------
# Copyright (C) 2018-2019 Angelos Evripiotis.
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
