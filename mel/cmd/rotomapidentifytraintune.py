"""Find the best hyper-parameters for identify-train."""

import pathlib
import pickle
import warnings

# 0 TrialState.COMPLETE  acc:91.9% {'cnn_width': 139, 'cnn_depth': 3}
# 1 TrialState.COMPLETE  acc:55.6% {'cnn_width': 7, 'cnn_depth': 3}
# 2 TrialState.COMPLETE  acc:90.2% {'cnn_width': 63, 'cnn_depth': 7}
# 3 TrialState.COMPLETE  acc:80.2% {'cnn_width': 17, 'cnn_depth': 7}
# 4 TrialState.COMPLETE  acc:88.8% {'cnn_width': 125, 'cnn_depth': 8}
# 5 TrialState.COMPLETE  acc:95.8% {'cnn_width': 147, 'cnn_depth': 4}
# 6 TrialState.PRUNED    acc:54.4% {'cnn_width': 8, 'cnn_depth': 4}
# 7 TrialState.PRUNED    acc:74.3% {'cnn_width': 26, 'cnn_depth': 5}
# 8 TrialState.PRUNED    acc:22.7% {'cnn_width': 4, 'cnn_depth': 1}
# 9 TrialState.PRUNED    acc:35.1% {'cnn_width': 55, 'cnn_depth': 1}
# 10 TrialState.COMPLETE  acc:93.4% {'cnn_width': 224, 'cnn_depth': 5}
# 11 TrialState.COMPLETE  acc:92.2% {'cnn_width': 207, 'cnn_depth': 5}
# 12 TrialState.PRUNED    acc:79.6% {'cnn_width': 242, 'cnn_depth': 6}
# 13 TrialState.PRUNED    acc:79.0% {'cnn_width': 81, 'cnn_depth': 3}
# 14 TrialState.COMPLETE  acc:96.3% {'cnn_width': 235, 'cnn_depth': 4}
# 15 TrialState.PRUNED    acc:69.0% {'cnn_width': 117, 'cnn_depth': 2}
# 16 TrialState.COMPLETE  acc:91.7% {'cnn_width': 39, 'cnn_depth': 4}
# 17 TrialState.PRUNED    acc:70.9% {'cnn_width': 253, 'cnn_depth': 2}
# 18 TrialState.PRUNED    acc:81.2% {'cnn_width': 96, 'cnn_depth': 6}
# 19 TrialState.COMPLETE  acc:96.2% {'cnn_width': 161, 'cnn_depth': 4}
# 20 TrialState.PRUNED    acc:60.7% {'cnn_width': 24, 'cnn_depth': 2}
# 21 TrialState.COMPLETE  acc:95.8% {'cnn_width': 149, 'cnn_depth': 4}
# 22 TrialState.COMPLETE  acc:96.7% {'cnn_width': 177, 'cnn_depth': 4}
# 23 TrialState.PRUNED    acc:81.8% {'cnn_width': 188, 'cnn_depth': 3}
# 24 TrialState.PRUNED    acc:77.9% {'cnn_width': 50, 'cnn_depth': 6}
# 25 TrialState.PRUNED    acc:81.6% {'cnn_width': 82, 'cnn_depth': 5}
# 26 TrialState.COMPLETE  acc:95.9% {'cnn_width': 180, 'cnn_depth': 4}
# 27 TrialState.PRUNED    acc:82.6% {'cnn_width': 92, 'cnn_depth': 3}
# 28 TrialState.PRUNED    acc:80.1% {'cnn_width': 172, 'cnn_depth': 5}
# 29 TrialState.PRUNED    acc:67.2% {'cnn_width': 117, 'cnn_depth': 2}
# 30 TrialState.PRUNED    acc:82.8% {'cnn_width': 251, 'cnn_depth': 3}
# 31 TrialState.COMPLETE  acc:96.1% {'cnn_width': 171, 'cnn_depth': 4}
# 32 TrialState.COMPLETE  acc:96.7% {'cnn_width': 254, 'cnn_depth': 4}
# 33 TrialState.PRUNED    acc:79.8% {'cnn_width': 133, 'cnn_depth': 3}
# 34 TrialState.COMPLETE  acc:96.9% {'cnn_width': 242, 'cnn_depth': 4}
# 35 TrialState.PRUNED    acc:82.1% {'cnn_width': 255, 'cnn_depth': 5}


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

    import mel.lib.ellipsespace
    import mel.lib.fs
    import mel.rotomap.identifynn
    import mel.rotomap.moles

    melroot = mel.lib.fs.find_melroot()

    study_name = "mel-rotomap-identify-train-tune"
    pruner = optuna.pruners.MedianPruner()
    study_path = pathlib.Path(f"{study_name}.pickle")
    if study_path.exists():
        with study_path.open("rb") as f:
            study = pickle.load(f)
        print("Loaded:", study_path)
        report_study(study)
    else:
        study = optuna.create_study(
            direction="maximize", pruner=pruner, study_name=study_name
        )

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
        _,
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

    while True:
        trial = study.ask()
        values = None
        state = optuna.trial.TrialState.COMPLETE
        try:
            values = objective(trial)
        except KeyboardInterrupt:
            print("Interrupted.")
            break
        except optuna.exceptions.TrialPruned:
            state = optuna.trial.TrialState.PRUNED
        # except Exception:
        #     state = TrialState.FAIL

        study.tell(trial, values=values, state=state)

        with study_path.open("wb") as f:
            pickle.dump(study, f)
        report_study(study)
        print("Wrote:", study_path)

    report_study(study)


def report_study(study):
    print("Number of finished trials: {}".format(len(study.trials)))
    for trial in study.trials:
        print(
            f"{trial.number:3}",
            f"{trial.state:20}",
            f"acc:{trial.value:5.1%}",
            trial.params,
        )

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

    cnn_width = trial.suggest_int("cnn_width", 4, 256, log=True)
    cnn_depth = trial.suggest_int("cnn_depth", 1, 8)
    num_cnns = 3
    channels_in = 2
    epochs = 100
    # epochs = 1

    print("cnn_width:", cnn_width)
    print("cnn_depth:", cnn_depth)

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

    if trainer.interrupted:
        raise KeyboardInterrupt()

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
