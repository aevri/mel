"""Train to guess which mole is which in a rotomap image."""

import json
import logging

import torch.utils.data

import mel.rotomap.moles
import mel.lib.ellipsespace
import mel.lib.fs
import mel.rotomap.identifynn


def setup_parser(parser):
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print information about the processing.",
    )


def process_args(args):
    melroot = mel.lib.fs.find_melroot()
    model_dir = melroot / mel.lib.fs.DEFAULT_CLASSIFIER_PATH
    model_path = model_dir / "identify.pth"
    metadata_path = model_dir / "identify.json"
    print(f"Will save to {model_path}.")

    image_size = 32
    cnn_width = 128
    cnn_depth = 4

    data_config = {
        "rotomaps": ("limbs"),
        # "rotomaps": ("subpart", "LeftLeg", "Lower"),
        "train_proportion": 0.9,
        "image_size": image_size,
        "photo_size": 224,
        "batch_size": 100,
        "do_photo": False,
        "do_augmentation": False,
        "do_channels": False,
    }

    print("Making data ..")
    data = mel.rotomap.identifynn.make_data(melroot, data_config)

    model_config = {
        "cnn_width": cnn_width,
        "cnn_depth": cnn_depth,
        "use_pos": False,
        "use_photo": data_config["do_photo"],
    }

    weight_decay = 0.005
    learning_rate = 0.01
    momentum = 0.95

    train_config = {
        "epochs": 10,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
    }

    results = mel.rotomap.identifynn.make_model_and_fit(*data, model_config, train_config)

    valid_fit_record = mel.rotomap.identifynn.FitRecord.from_dict(results["valid_fit_record"])
    print(valid_fit_record.acc[-1])

    model = results["model"]

    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_path}.")
    with open(metadata_path, "w") as f:
        metadata = {
            "model_args": results["model_args"],
            "part_to_index": results["part_to_index"],
            "classes": results["classes"],
            "image_size": image_size,
        }
        json.dump(metadata, f)


def setup_logging(verbosity):
    logtypes = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = min(len(logtypes) - 1, verbosity)
    logging.basicConfig(
        level=logtypes[level], format="%(levelname)s: %(message)s"
    )


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
