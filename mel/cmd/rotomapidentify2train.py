"""Train to guess which mole is which in a rotomap image."""

import argparse
import json
import os
import warnings


def proportion_arg(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{x}' is not a float")

    if not (x > 0.0 and x <= 1.0):
        if not x == -1:
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
        "--extra-stem",
        nargs="+",
        help="Add an extra bit to the filename stem, e.g. '0.jpg.EXTRA.json'.",
    )
    parser.add_argument(
        "--only-train-classifier",
        action="store_true",
        help="Only train the classifier bits, good when fine-tuning.",
    )


def process_args(args):
    # Some of are expensive imports, so to keep program start-up time lower,
    # import them only when necessary.
    import torch
    from tqdm.auto import tqdm

    import mel.lib.ellipsespace
    import mel.lib.fs
    import mel.rotomap.identifynn2
    import mel.rotomap.moles
    import mel.rotomap.dataset

    melroot = mel.lib.fs.find_melroot()
    model_dir = melroot / mel.lib.fs.DEFAULT_CLASSIFIER_PATH
    model_path = model_dir / "identify2.pth"
    metadata_path = model_dir / "identify2.json"

    if not model_dir.exists():
        model_dir.mkdir()

    print("Making data ..")

    warnings.filterwarnings(
        "ignore",
        message="Creating a tensor from a list of numpy.ndarrays is extremely slow.",
    )

    pathdict = mel.rotomap.dataset.make_pathdict(melroot)
    pathdict = mel.rotomap.dataset.drop_empty_paths(pathdict)
    train, valid = mel.rotomap.dataset.split_train_valid_last(pathdict)
    partnames_uuids = mel.rotomap.dataset.make_partnames_uuids(pathdict)

    def process_dataset(pathdict, name):
        extra_stem = [None]
        if args.extra_stem:
            extra_stem.extend(args.extra_stem)
        d = mel.rotomap.dataset.listify_pathdict(pathdict)
        d = mel.rotomap.dataset.yield_imagemoles_from_pathlist(
            d,
            extra_stem_list=extra_stem,
        )
        d = list(d)
        print(f"There are {len(d)} {name} items.")
        return d

    train = process_dataset(train, "training")
    valid = process_dataset(valid, "validation")

    num_neighbours = 15
    # model = mel.rotomap.identifynn2.SelfposOnly(
    #     partnames_uuids
    # )
    # model = mel.rotomap.identifynn2.PosOnlyLinear(
    #     partnames_uuids, num_neighbours=num_neighbours
    # )
    if model_path.exists():
        print(f"Will fine-tune {model_path}")
        print(f"           and {metadata_path}")

        if not metadata_path.exists():
            raise Exception(
                f"Metadata for model does not exist: " f"{metadata_path}"
            )

        if not os.access(model_path, os.W_OK):
            print("No permission to write to", model_path)
            return 1

        if not os.access(metadata_path, os.W_OK):
            print("No permission to write to", metadata_path)
            return 1

        with open(metadata_path) as f:
            old_metadata = json.load(f)

        if old_metadata["num_neighbours"] != num_neighbours:
            print("Cannot fine-tune with different number of neighbours.")
            return 1

        model = mel.rotomap.identifynn2.PosOnly(
            partnames_uuids=old_metadata["partnames_uuids"],
            num_neighbours=old_metadata["num_neighbours"],
        )
        model.load_state_dict(torch.load(model_path))
        model.update_partnames_uuids(partnames_uuids)
        if args.only_train_classifier:
            print("Freezing except classifier.")
            model.freeze_except_classifier()
    else:
        print(f"Will save to {model_path}")
        print(f"         and {metadata_path}")

        model = mel.rotomap.identifynn2.PosOnly(
            partnames_uuids, num_neighbours=num_neighbours
        )
        if args.only_train_classifier:
            print("Makes no sense to 'only train classifier' from scratch.")
            return 1

    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    trainer = mel.rotomap.identifynn2.Trainer(
        model,
        criterion,
        optimizer,
        train,
        valid,
        patience=args.epochs,
        epochs=args.epochs,
        max_lr=0.002,
    )
    print("Device:", trainer.device)

    metadata = {
        "partnames_uuids": partnames_uuids,
        "num_neighbours": num_neighbours,
    }

    print("Batches per training epoch:", len(trainer.make_train_dataloader()))
    print("Batches per validation epoch:", len(trainer.valid_loader))

    try:
        for _ in (pbar := tqdm(range(trainer.epochs))):
            trainer.train()
            trainer.validate()
            pbar.set_description(
                f"val_acc:{trainer.valid_acc[-1]:.1%} "
                f"tra_acc:{trainer.train_acc[-1]:.1%} "
            )
    except mel.rotomap.identifynn2.EarlyStoppingException:
        print("Stopping training early due to no improvement.")
        pass

    print("Validation loss:", trainer.valid_loss[-1])
    print("Validation acc:", f"{trainer.valid_acc[-1]:.0%}")

    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_path}")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
        print(f"Saved {metadata_path}")


# -----------------------------------------------------------------------------
# Copyright (C) 2023 Angelos Evripiotis.
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
