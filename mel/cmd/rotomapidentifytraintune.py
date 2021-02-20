"""Find the best hyper-parameters for identify-train."""

import warnings


def setup_parser(parser):
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print information about the processing.",
    )


_DATA = None


def process_args(args):
    import optuna

    import mel.rotomap.moles
    import mel.lib.ellipsespace
    import mel.lib.fs
    import mel.rotomap.identifynn

    melroot = mel.lib.fs.find_melroot()

    image_size = 32
    batch_size = 100
    train_proportion = 0.9
    data_config = {
        "rotomaps": ("all"),
        # "rotomaps": ("subpart", "LeftLeg", "Lower"),
        "train_proportion": train_proportion,
        "image_size": image_size,
        "batch_size": batch_size,
        "do_augmentation": False,
        "do_channels": False,
    }
    print("Making data ..")
    (
        train_dataset,
        valid_dataset,
        train_dataloader,
        valid_dataloader,
        part_to_index,
    ) = mel.rotomap.identifynn.make_data(melroot, data_config)
    num_parts = len(part_to_index)
    num_classes = len(train_dataset.classes)
    global _DATA
    _DATA = (
        train_dataloader,
        valid_dataloader,
        num_parts,
        num_classes,
    )

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    # study.optimize(objective, n_trials=100, timeout=500 * 60)
    study.optimize(objective, n_trials=10, timeout=500 * 60)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def objective(trial):
    # Some of are expensive imports, so to keep program start-up time lower,
    # import them only when necessary.
    import pytorch_lightning as pl
    from optuna.integration import PyTorchLightningPruningCallback

    # from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    import mel.rotomap.identifynn

    cnn_width = trial.suggest_int("cnn_width", 4, 1024, log=True)
    cnn_depth = trial.suggest_int("cnn_depth", 1, 8)
    num_cnns = 3
    channels_in = 2
    epochs = 100
    # epochs = 1

    (
        train_dataloader,
        valid_dataloader,
        num_parts,
        num_classes,
    ) = _DATA

    model_args = dict(
        cnn_width=cnn_width,
        cnn_depth=cnn_depth,
        num_parts=num_parts,
        num_classes=num_classes,
        num_cnns=num_cnns,
        channels_in=channels_in,
    )

    pl_model = mel.rotomap.identifynn.LightningModel(
        model_args, trainable_conv=True
    )

    warnings.filterwarnings(
        "ignore",
        message=(
            r"The dataloader, \w+ dataloader( 0)?, "
            "does not have many workers which may be a bottleneck."
        ),
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="accuracy")],
    )
    if not valid_dataloader:
        valid_dataloader = None

    trainer.fit(pl_model, train_dataloader, valid_dataloader)

    return trainer.callback_metrics["accuracy"].item()


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
