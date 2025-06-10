"""Train 'filter-marks'."""

import argparse
import json

import mel.lib.common
import mel.rotomap.filtermarks

# TODO: Check for masking errors, where the mole is obscured by the mask.
# TODO: Make the mask green.


def proportion_arg(x):
    try:
        x = float(x)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"'{x}' is not a float") from e

    if not (0.0 < x <= 1.0):
        raise argparse.ArgumentTypeError(f"'{x}' is not in range 0.0 < x <= 1.0")

    return x


def setup_parser(parser):
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print information about the processing.",
    )

    parser.add_argument(
        "--batch-size",
        default=mel.rotomap.filtermarks.DEFAULT_BATCH_SIZE,
        type=int,
        help=(
            "How many image patches to process at once. "
            "Higher is better, until you run out of RAM."
        ),
    )

    parser.add_argument(
        "--num-epochs",
        default=200,
        type=int,
        help=(
            "How many times to learn from each image. Higher is better, up to a point."
        ),
    )

    parser.add_argument(
        "--learning-rate",
        default=0.02,
        type=float,
        help=(
            "How quickly to adjust the model based on examples. "
            "Too low and nothing happens, too high and it doesn't learn."
        ),
    )

    parser.add_argument(
        "--train-proportion",
        "-t",
        type=proportion_arg,
        default=0.9,
        help=(
            "Proportion (0.0 < x <= 1.0) of data to use for training. Defaults to 0.9."
        ),
    )


def process_args(args):
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torch

    melroot = mel.lib.fs.find_melroot()
    model_dir = melroot / mel.lib.fs.DEFAULT_CLASSIFIER_PATH
    model_path = model_dir / "filtermarks.pth"
    metadata_path = model_dir / "filtermarks.json"
    print(f"Will save to {model_path}")
    print(f"         and {metadata_path}")

    pretrained_paths = mel.rotomap.filtermarks.find_pretrained(melroot)
    pretrained_data = mel.rotomap.filtermarks.load_pretrained(pretrained_paths)
    training_data, validation_data = mel.rotomap.filtermarks.split_data(
        pretrained_data, args.train_proportion
    )
    print(f"Loaded {len(training_data):,} training examples.")
    print(f"Loaded {len(validation_data):,} validation examples.")

    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    evaluators = [mel.rotomap.filtermarks.Evaluator(t) for t in thresholds]

    model = mel.rotomap.filtermarks.make_model_and_fit(
        training_data,
        validation_data,
        evaluators,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
    )

    if args.train_proportion < 1:
        print(f"{'threshold':<12} {'precision':<12} {'recall':<12}")
        for e in evaluators:
            print(f"{e.threshold:<12} {int(e.precision()):<12} {int(e.recall()):<12}")

    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_path}")
    with open(metadata_path, "w") as f:
        metadata = {
            "model_weights_version": mel.rotomap.filtermarks.get_model_weights_version()
        }
        json.dump(metadata, f)
        print(f"Saved {metadata_path}")


# -----------------------------------------------------------------------------
# Copyright (C) 2022 Angelos Evripiotis.
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
