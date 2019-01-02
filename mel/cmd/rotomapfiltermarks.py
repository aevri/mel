"""Automatically remove marked regions that are probably not moles.

This is experimental functionality and results may be variable.

In order to create the 'model file' required by this command, you will need to
have tagged a large number of moles by hand. You can then create a dataset with
the 'rotomap photoset' command. The first lesson in the fast ai deep learning
course will teach you to generate the file: "http://course.fast.ai/".

Please install the 'fastai' python library and its dependencies in order for
this to work: https://pypi.org/project/fastai/
"""

import os
import sys

import mel.lib.fs

import mel.rotomap.moles

import cv2

import numpy


def setup_parser(parser):

    parser.add_argument(
        'FRAMES',
        type=mel.rotomap.moles.make_argparse_image_moles,
        nargs='+',
        help="Path to rotomap or image to filter.",
    )

    parser.add_argument(
        '--model-path',
        help="Path to the model to use, relative to the root of the mel repo. "
        f"Defaults to {mel.lib.fs.DEFAULT_MOLE_MARK_MODEL_PATH}.",
        default=mel.lib.fs.DEFAULT_MOLE_MARK_MODEL_PATH,
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help="Print information about the processing.",
    )


def process_args(args):
    try:
        melroot = mel.lib.fs.find_melroot()
    except mel.lib.fs.NoMelrootError:
        print('Not in a mel repo, could not find melroot', file=sys.stderr)
        return 1

    if args.verbose:
        print("Initialising classifier .. ", end='')
    model_path = melroot / args.model_path
    is_mole = make_is_mole_func(model_path)
    if args.verbose:
        print("done")
    for image_mole_iter in args.FRAMES:
        for image_path, moles in image_mole_iter:
            if args.verbose:
                print(image_path)
            image = open_image_for_classifier(image_path)
            try:
                filtered_moles = filter_marks(
                    is_mole, image, moles, args.verbose
                )
            except Exception:
                raise Exception('Error while processing {}'.format(image_path))

            num_filtered = len(moles) - len(filtered_moles)
            if args.verbose:
                print(f"Filtered {num_filtered} unlikely moles.")

            if num_filtered:
                mel.rotomap.moles.save_image_moles(filtered_moles, image_path)


def make_is_mole_func(model_path):

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
    import fastai.conv_learner
    import fastai.core
    import fastai.dataloader
    import fastai.model
    import fastai.transforms

    # pylint: enable=import-error

    architecture = fastai.model.resnet34

    num_classes = 2
    is_multiclass = True
    is_regression = False

    models = fastai.conv_learner.ConvnetBuilder(
        architecture, num_classes, is_multiclass, is_regression
    )

    fastai.model.load_model(models.model, model_path)

    size = 224

    # Returns 'validation' transforms and 'training' transforms. We want the
    # validation ones, as they're closest to what we want in production.
    _, transforms = fastai.transforms.tfms_from_model(architecture, size)

    def is_mole(image):

        batch = numpy.stack([transforms(image)])
        tensor = fastai.dataloader.get_tensor(batch, pin=False).contiguous()

        result_vv = fastai.core.VV([tensor])
        models.model.eval()
        result_model = models.model(*result_vv)
        result_to_np = fastai.model.to_np(result_model)
        log_preds = fastai.model.get_prediction(result_to_np)
        log_preds = numpy.concatenate([log_preds])
        probs = numpy.exp(log_preds)
        preds = numpy.argmax(probs, axis=1)
        probs = probs[:, 1]

        return bool(preds[0])

    return is_mole


def filter_marks(is_mole, image, moles, verbose):
    """Return a list of moles with the unlikely ones filtered out.

    :image: A numpy array representing an RGB image.
    :moles: A list of mel.rotomap.moles.
    :returns: The filtered list of mel.rotomap.moles.
    """

    filtered_moles = []
    for m in moles:
        if verbose:
            print(m['uuid'], '..', end=' ')

        # Don't touch canonical moles.
        if m[mel.rotomap.moles.KEY_IS_CONFIRMED]:
            if verbose:
                print('skipped because canonical')
            continue

        r = int(m['radius'])

        r2 = r * 2
        x = m['x']
        y = m['y']

        image_fragment = image[
            y-r2:y+r2,
            x-r2:x+r2
        ]

        # TODO: decide what to do about fragments that overlap the image
        # boundary.
        if not all(image_fragment.shape):
            if verbose:
                print('skipped because zero radius')
            continue

        is_confirmed = is_mole(image_fragment)

        if is_confirmed:
            filtered_moles.append(m)

        if verbose:
            if is_confirmed:
                print('mole')
            else:
                print('non-mole')

    return filtered_moles


def open_image_for_classifier(fn):
    flags = cv2.IMREAD_UNCHANGED + cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn):
        raise OSError('Is a directory: {}'.format(fn))

    try:
        im = cv2.imread(str(fn), flags).astype(numpy.float32) / 255
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
