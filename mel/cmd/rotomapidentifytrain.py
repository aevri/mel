"""Train to guess which mole is which in a rotomap image.

It can be helpful to pre-train a model on a reasonable amount of fake data
before fine-tuning it on your own data. This is especially helpful when you
don't have much data of your own yet. For example:

-   mkdir temp
    cd temp
    mel-debug gen-repo --num-rotomaps 10 --num-parts 10 .
    mel rotomap automask rotomaps/parts/*/*/*/*.jpg
    mel rotomap calc-space rotomaps/parts/*/*/*/*.jpg
    mel rotomap identify-train --forget-moles

Then you can copy the model out of the fake repo, and then train it again on
your data.

Here is a good recipe if you have a GPU:

-   mkdir temp
    cd temp
    mel-debug gen-repo --num-rotomaps 100 --num-parts 10 .
    mel rotomap automask rotomaps/parts/*/*/*/*.jpg
    mel rotomap calc-space rotomaps/parts/*/*/*/*.jpg
    mel rotomap identify-train --epochs 100 --batch-size 500 --lr 0.001
    mel rotomap identify-train --epochs 200 --batch-size 500 --lr 0.0001
"""

import argparse
import json
import os
import warnings


def proportion_arg(x):
    try:
        x = float(x)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"'{x}' is not a float") from e

    if not (x > 0.0 and x <= 1.0) and not x == -1:
        raise argparse.ArgumentTypeError(
            f"'{x}' is not in range 0.0 > x <= 1.0, or -1."
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
        "--batch-size",
        "-b",
        type=int,
        default=100,
        help="Number of items in a batch. Best to increase until it crashes.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--train-proportion",
        "-t",
        type=proportion_arg,
        default=0.9,
        help=("Proportion (0.0-1.0) of data to use for training. Defaults to 0.9."),
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
    parser.add_argument(
        "--extra-stem",
        help="Add an extra bit to the filename stem, e.g. '0.jpg.EXTRA.json'.",
    )
    parser.add_argument(
        "--wandb",
        nargs=2,
        metavar=("project", "run_name"),
        help="Use a https://wandb.ai/ logger.",
    )


def process_args(args):
    # Some of are expensive imports, so to keep program start-up time lower,
    # import them only when necessary.
    import pytorch_lightning as pl
    import torch

    import mel.lib.ellipsespace
    import mel.lib.fs
    import mel.rotomap.identifynn
    import mel.rotomap.moles

    melroot = mel.lib.fs.find_melroot()
    model_dir = melroot / mel.lib.fs.DEFAULT_CLASSIFIER_PATH
    model_path = model_dir / "identify.pth"
    metadata_path = model_dir / "identify.json"

    image_size = 32
    cnn_width = 128
    cnn_depth = 4
    num_cnns = 3
    channels_in = 2

    old_metadata = None
    if model_path.exists():
        if not metadata_path.exists():
            raise Exception(f"Metadata for model does not exist: {metadata_path}")

        if not os.access(model_path, os.W_OK):
            print("No permission to write to", model_path)
            return 1

        if not os.access(metadata_path, os.W_OK):
            print("No permission to write to", metadata_path)
            return 1

        print(f"Will fine-tune {model_path}")
        print(f"           and {metadata_path}")

        with open(metadata_path) as f:
            old_metadata = json.load(f)

        def check_old_matches_new(name, old, new):
            if old != new:
                raise Exception(
                    f"Old {name} is not the same.\nold: {old}\nnew: {new}\n"
                )

        old_model_args = old_metadata["model_args"]
        check_old_matches_new("cnn width", old_model_args["cnn_width"], cnn_width)
        check_old_matches_new("cnn depth", old_model_args["cnn_depth"], cnn_depth)
        check_old_matches_new("num cnns", old_model_args["num_cnns"], num_cnns)
        check_old_matches_new("channels_in", old_model_args["channels_in"], channels_in)
        check_old_matches_new("image size", old_metadata["image_size"], image_size)
    else:
        print(f"Will save to {model_path}")
        print(f"         and {metadata_path}")

    if not model_dir.exists():
        model_dir.mkdir()

    data_config = {
        "rotomaps": ("all"),
        "train_proportion": args.train_proportion,
        "image_size": image_size,
        "batch_size": args.batch_size,
        "do_augmentation": False,
        "do_channels": False,
        "extra_stem": args.extra_stem,
    }

    base_trainer_kwargs = {
        "max_epochs": args.epochs,
        "accelerator": "auto",
        "log_every_n_steps": 5,
    }

    wandb_kwargs = {}
    if args.wandb:
        wandb_project, wandb_run_name = args.wandb
        wandb_kwargs = {
            "logger": pl.loggers.WandbLogger(project=wandb_project, name=wandb_run_name)
        }

    trainer_kwargs = base_trainer_kwargs | wandb_kwargs

    print("Making data ..")
    (
        train_dataset,
        _,
        train_dataloader,
        valid_dataloader,
        part_to_index,
    ) = mel.rotomap.identifynn.make_data(melroot, data_config)

    num_parts = len(part_to_index)
    num_classes = len(train_dataset.classes)

    model_args = {
        "cnn_width": cnn_width,
        "cnn_depth": cnn_depth,
        "num_parts": num_parts,
        "num_classes": num_classes,
        "num_cnns": num_cnns,
        "channels_in": channels_in,
    }

    init_model_args = model_args
    if old_metadata is not None:
        init_model_args = old_metadata["model_args"]

    pl_model = mel.rotomap.identifynn.LightningModel(
        init_model_args, not args.no_train_conv, lr=args.lr
    )

    metadata = {
        "model_args": model_args,
        "part_to_index": part_to_index,
        "classes": train_dataset.classes,
        "image_size": image_size,
    }

    if old_metadata is not None:
        pl_model.model.load_state_dict(torch.load(model_path))
        _fixup_old_model(old_metadata, metadata, pl_model.model)

    warnings.filterwarnings(
        "ignore",
        message=(
            r"The dataloader, \w+ dataloader( 0)?, "
            "does not have many workers which may be a bottleneck."
        ),
    )

    trainer = pl.Trainer(**trainer_kwargs)
    if not valid_dataloader:
        valid_dataloader = None

    trainer.fit(pl_model, train_dataloader, valid_dataloader)

    model = pl_model.model

    if args.forget_moles:
        model.clear_non_cnn()
        metadata["model_args"]["num_parts"] = 0
        metadata["model_args"]["num_classes"] = 0
        metadata["part_to_index"] = {}
        metadata["classes"] = []

    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_path}")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
        print(f"Saved {metadata_path}")

    if args.wandb:
        import wandb

        wandb.finish()
        return None
    return None


def _fixup_old_model(old_metadata, new_metadata, model):
    old_model_args = old_metadata["model_args"]
    new_model_args = new_metadata["model_args"]
    if (
        old_metadata["part_to_index"] != new_metadata["part_to_index"]
        or old_metadata["classes"] != new_metadata["classes"]
    ):
        # TODO: support copying over embeddings and other bits that are the
        # same, and don't need to be re-learned.
        model.reset_num_parts_classes(
            new_num_parts=new_model_args["num_parts"],
            new_num_classes=new_model_args["num_classes"],
        )
        old_model_args["num_parts"] = new_model_args["num_parts"]
        old_model_args["num_classes"] = new_model_args["num_classes"]
    if old_model_args != new_model_args:
        raise Exception(
            f"Old model args not compatible.\n"
            f"old: {old_model_args}\n"
            f"new: {new_model_args}\n"
        )


# -----------------------------------------------------------------------------
# Copyright (C) 2021 Angelos Evripiotis.
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
