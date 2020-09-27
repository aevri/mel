"""Train to guess which mole is which in a rotomap image.

It can be helpful to pre-train a model on a reasonable amount of fake data
before fine-tuning it on your own data. This is especially helpful when you
don't have much data of your own yet. For example:

    mkdir temp
    cd temp
    mel-debug gen-repo --num-rotomaps 10 --num-parts 10 .
    mel rotomap automask rotomaps/parts/*/*/*/*.jpg
    mel rotomap calc-space rotomaps/parts/*/*/*/*.jpg
    mel rotomap identify2-train --forget-moles

Then you can copy the model out of the fake repo, and then train it again on
your data.
"""

import argparse
import json


def proportion_arg(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{x}' is not a float")

    if not (x > 0.0 and x <= 1.0):
        raise argparse.ArgumentTypeError(
            f"'{x}' is not in range 0.0 > x <= 1.0"
        )

    return x


def setup_parser(parser):
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print information about the processing.",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=10,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--train-proportion",
        "-t",
        type=proportion_arg,
        default=0.9,
        help=(
            "Proportion (0.0-1.0) of data to use for training. "
            "Defaults to 0.9."
        ),
    )
    parser.add_argument(
        "--no-train-conv",
        action="store_true",
        help=(
            "Don't train the convnet, useful when fine-tuning "
            "a network that just needs to learn new moles."
        ),
    )
    parser.add_argument(
        "--forget-moles",
        action="store_true",
        help=(
            "Don't save the mapping from features to moles, "
            "only the bit that is good at identifying useful features. "
            "This is useful when training a model on simulated data, "
            "which will then be fine-tuned to a new set of moles."
        ),
    )


def process_args(args):
    # Some of are expensive imports, so to keep program start-up time lower,
    # import them only when necessary.
    import torch.utils.data

    import mel.rotomap.moles
    import mel.lib.ellipsespace
    import mel.lib.fs
    import mel.rotomap.identifynn

    melroot = mel.lib.fs.find_melroot()
    model_dir = melroot / mel.lib.fs.DEFAULT_CLASSIFIER_PATH
    model_path = model_dir / "identify.pth"
    metadata_path = model_dir / "identify.json"

    image_size = 32
    cnn_width = 128
    cnn_depth = 4

    old_results = None
    if model_path.exists():
        if not metadata_path.exists():
            raise Exception(
                f"Metadata for model does not exist: " f"{metadata_path}"
            )
        print(f"Will fine-tune {model_path}")
        print(f"           and {metadata_path}")

        with open(metadata_path) as f:
            old_results = json.load(f)
        old_results["model"] = mel.rotomap.identifynn.Model(
            **old_results["model_args"]
        )
        old_results["model"].load_state_dict(torch.load(model_path))
        if old_results["image_size"] != image_size:
            raise Exception(
                f"Old image size is not the same.\n"
                f"old: {old_results['image_size']}\n"
                f"new: {image_size}\n"
            )
    else:
        print(f"Will save to {model_path}")
        print(f"         and {metadata_path}")

    data_config = {
        "rotomaps": ("all"),
        "train_proportion": args.train_proportion,
        "image_size": image_size,
        "batch_size": 100,
        "do_augmentation": False,
        "do_channels": False,
    }

    print("Making data ..")
    data = mel.rotomap.identifynn.make_data(melroot, data_config)

    model_config = {
        "cnn_width": cnn_width,
        "cnn_depth": cnn_depth,
    }

    weight_decay = 0.01
    learning_rate = 0.01

    train_config = {
        "epochs": args.epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "train_conv": not args.no_train_conv,
    }

    results = mel.rotomap.identifynn.make_model_and_fit(
        *data, model_config, train_config, old_results
    )

    if data_config["train_proportion"] != 1:
        valid_fit_record = mel.rotomap.identifynn.FitRecord.from_dict(
            results["valid_fit_record"]
        )
        print(valid_fit_record.acc[-1])

    model = results["model"]

    if not model_dir.exists():
        model_dir.mkdir()
    if args.forget_moles:
        model.clear_non_cnn()
        results["part_to_index"] = {}
        results["classes"] = []
        results["model_args"]["num_parts"] = 0
        results["model_args"]["num_classes"] = 0
    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_path}")
    with open(metadata_path, "w") as f:
        metadata = {
            "model_args": results["model_args"],
            "part_to_index": results["part_to_index"],
            "classes": results["classes"],
            "image_size": image_size,
        }
        json.dump(metadata, f)
        print(f"Saved {metadata_path}")


# -----------------------------------------------------------------------------
# Copyright (C) 2019 Angelos Evripiotis.
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
