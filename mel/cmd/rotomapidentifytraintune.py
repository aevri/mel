"""Find the best hyper-parameters for identify-train."""

import pathlib
import pickle
import warnings

  # 0 TrialState.COMPLETE  acc:65.7% {'cnn_width': 76, 'cnn_depth': 1}
  # 1 TrialState.COMPLETE  acc:91.6% {'cnn_width': 54, 'cnn_depth': 6}
  # 2 TrialState.COMPLETE  acc:88.5% {'cnn_width': 133, 'cnn_depth': 2}
  # 3 TrialState.COMPLETE  acc:37.9% {'cnn_width': 5, 'cnn_depth': 4}
  # 4 TrialState.COMPLETE  acc:96.4% {'cnn_width': 246, 'cnn_depth': 5}
  # 6 TrialState.COMPLETE  acc:93.3% {'cnn_width': 68, 'cnn_depth': 6}
  # 7 TrialState.COMPLETE  acc:94.0% {'cnn_width': 227, 'cnn_depth': 6}
 # 11 TrialState.COMPLETE  acc:96.9% {'cnn_width': 246, 'cnn_depth': 5}
 # 12 TrialState.COMPLETE  acc:97.6% {'cnn_width': 224, 'cnn_depth': 4}
 # 13 TrialState.COMPLETE  acc:98.0% {'cnn_width': 154, 'cnn_depth': 4}
 # 14 TrialState.COMPLETE  acc:97.8% {'cnn_width': 141, 'cnn_depth': 4}
 # 20 TrialState.COMPLETE  acc:97.3% {'cnn_width': 101, 'cnn_depth': 4}
 # 21 TrialState.COMPLETE  acc:97.5% {'cnn_width': 196, 'cnn_depth': 4}
 # 22 TrialState.COMPLETE  acc:96.1% {'cnn_width': 167, 'cnn_depth': 5}
 # 23 TrialState.COMPLETE  acc:96.0% {'cnn_width': 108, 'cnn_depth': 4}
 # 31 TrialState.COMPLETE  acc:97.8% {'cnn_width': 191, 'cnn_depth': 4}
 # 32 TrialState.COMPLETE  acc:97.4% {'cnn_width': 130, 'cnn_depth': 4}
 # 34 TrialState.COMPLETE  acc:96.9% {'cnn_width': 169, 'cnn_depth': 4}
 # 41 TrialState.COMPLETE  acc:98.2% {'cnn_width': 195, 'cnn_depth': 4}
 # 42 TrialState.COMPLETE  acc:98.0% {'cnn_width': 197, 'cnn_depth': 4}
 # 46 TrialState.COMPLETE  acc:97.4% {'cnn_width': 239, 'cnn_depth': 4}
 # 50 TrialState.COMPLETE  acc:97.1% {'cnn_width': 154, 'cnn_depth': 4}
 # 51 TrialState.COMPLETE  acc:98.1% {'cnn_width': 220, 'cnn_depth': 4}
 # 52 TrialState.COMPLETE  acc:98.0% {'cnn_width': 184, 'cnn_depth': 4}
 # 53 TrialState.COMPLETE  acc:97.0% {'cnn_width': 253, 'cnn_depth': 4}
 # 60 TrialState.COMPLETE  acc:98.5% {'cnn_width': 255, 'cnn_depth': 4}
 # 61 TrialState.COMPLETE  acc:97.6% {'cnn_width': 222, 'cnn_depth': 4}
 # 72 TrialState.COMPLETE  acc:97.6% {'cnn_width': 186, 'cnn_depth': 4}
 # 75 TrialState.COMPLETE  acc:97.5% {'cnn_width': 254, 'cnn_depth': 4}
 # 81 TrialState.COMPLETE  acc:97.5% {'cnn_width': 199, 'cnn_depth': 4}
 # 88 TrialState.COMPLETE  acc:97.8% {'cnn_width': 158, 'cnn_depth': 4}
 # 92 TrialState.COMPLETE  acc:96.8% {'cnn_width': 162, 'cnn_depth': 4}
 # 94 TrialState.COMPLETE  acc:98.0% {'cnn_width': 252, 'cnn_depth': 4}
 # 95 TrialState.COMPLETE  acc:97.1% {'cnn_width': 239, 'cnn_depth': 4}
 # 96 TrialState.COMPLETE  acc:98.1% {'cnn_width': 185, 'cnn_depth': 4}
# 104 TrialState.COMPLETE  acc:97.6% {'cnn_width': 229, 'cnn_depth': 4}

  # 0 TrialState.COMPLETE  acc:65.7% {'cnn_width': 76, 'cnn_depth': 1}
  # 1 TrialState.COMPLETE  acc:91.6% {'cnn_width': 54, 'cnn_depth': 6}
  # 2 TrialState.COMPLETE  acc:88.5% {'cnn_width': 133, 'cnn_depth': 2}
  # 3 TrialState.COMPLETE  acc:37.9% {'cnn_width': 5, 'cnn_depth': 4}
  # 4 TrialState.COMPLETE  acc:96.4% {'cnn_width': 246, 'cnn_depth': 5}
  # 5 TrialState.PRUNED    acc:31.1% {'cnn_width': 5, 'cnn_depth': 2}
  # 6 TrialState.COMPLETE  acc:93.3% {'cnn_width': 68, 'cnn_depth': 6}
  # 7 TrialState.COMPLETE  acc:94.0% {'cnn_width': 227, 'cnn_depth': 6}
  # 8 TrialState.PRUNED    acc:25.9% {'cnn_width': 8, 'cnn_depth': 1}
  # 9 TrialState.PRUNED    acc:33.3% {'cnn_width': 6, 'cnn_depth': 2}
 # 10 TrialState.PRUNED    acc:61.1% {'cnn_width': 16, 'cnn_depth': 8}
 # 11 TrialState.COMPLETE  acc:96.9% {'cnn_width': 246, 'cnn_depth': 5}
 # 12 TrialState.COMPLETE  acc:97.6% {'cnn_width': 224, 'cnn_depth': 4}
 # 13 TrialState.COMPLETE  acc:98.0% {'cnn_width': 154, 'cnn_depth': 4}
 # 14 TrialState.COMPLETE  acc:97.8% {'cnn_width': 141, 'cnn_depth': 4}
 # 15 TrialState.PRUNED    acc:66.5% {'cnn_width': 32, 'cnn_depth': 3}
 # 16 TrialState.PRUNED    acc:78.9% {'cnn_width': 107, 'cnn_depth': 3}
 # 17 TrialState.PRUNED    acc:80.2% {'cnn_width': 148, 'cnn_depth': 8}
 # 18 TrialState.PRUNED    acc:65.8% {'cnn_width': 37, 'cnn_depth': 3}
 # 19 TrialState.PRUNED    acc:61.1% {'cnn_width': 18, 'cnn_depth': 7}
 # 20 TrialState.COMPLETE  acc:97.3% {'cnn_width': 101, 'cnn_depth': 4}
 # 21 TrialState.COMPLETE  acc:97.5% {'cnn_width': 196, 'cnn_depth': 4}
 # 22 TrialState.COMPLETE  acc:96.1% {'cnn_width': 167, 'cnn_depth': 5}
 # 23 TrialState.COMPLETE  acc:96.0% {'cnn_width': 108, 'cnn_depth': 4}
 # 24 TrialState.PRUNED    acc:84.6% {'cnn_width': 254, 'cnn_depth': 3}
 # 25 TrialState.PRUNED    acc:85.3% {'cnn_width': 49, 'cnn_depth': 5}
 # 26 TrialState.PRUNED    acc:81.5% {'cnn_width': 76, 'cnn_depth': 4}
 # 27 TrialState.PRUNED    acc:83.1% {'cnn_width': 153, 'cnn_depth': 3}
 # 28 TrialState.PRUNED    acc:72.3% {'cnn_width': 20, 'cnn_depth': 5}
 # 29 TrialState.PRUNED    acc:29.9% {'cnn_width': 94, 'cnn_depth': 1}
 # 30 TrialState.PRUNED    acc:82.1% {'cnn_width': 193, 'cnn_depth': 6}
 # 31 TrialState.COMPLETE  acc:97.8% {'cnn_width': 191, 'cnn_depth': 4}
 # 32 TrialState.COMPLETE  acc:97.4% {'cnn_width': 130, 'cnn_depth': 4}
 # 33 TrialState.PRUNED    acc:74.6% {'cnn_width': 64, 'cnn_depth': 3}
 # 34 TrialState.COMPLETE  acc:96.9% {'cnn_width': 169, 'cnn_depth': 4}
 # 35 TrialState.PRUNED    acc:89.4% {'cnn_width': 126, 'cnn_depth': 5}
 # 36 TrialState.PRUNED    acc:68.3% {'cnn_width': 203, 'cnn_depth': 2}
 # 37 TrialState.PRUNED    acc:84.4% {'cnn_width': 50, 'cnn_depth': 4}
 # 38 TrialState.PRUNED    acc:85.9% {'cnn_width': 254, 'cnn_depth': 5}
 # 39 TrialState.PRUNED    acc:71.3% {'cnn_width': 79, 'cnn_depth': 3}
 # 40 TrialState.PRUNED    acc:61.8% {'cnn_width': 125, 'cnn_depth': 2}
 # 41 TrialState.COMPLETE  acc:98.2% {'cnn_width': 195, 'cnn_depth': 4}
 # 42 TrialState.COMPLETE  acc:98.0% {'cnn_width': 197, 'cnn_depth': 4}
 # 43 TrialState.PRUNED    acc:81.3% {'cnn_width': 177, 'cnn_depth': 6}
 # 44 TrialState.PRUNED    acc:90.2% {'cnn_width': 86, 'cnn_depth': 5}
 # 45 TrialState.PRUNED    acc:91.8% {'cnn_width': 139, 'cnn_depth': 4}
 # 46 TrialState.COMPLETE  acc:97.4% {'cnn_width': 239, 'cnn_depth': 4}
 # 47 TrialState.PRUNED    acc:80.9% {'cnn_width': 116, 'cnn_depth': 3}
 # 48 TrialState.PRUNED    acc:83.2% {'cnn_width': 61, 'cnn_depth': 5}
 # 49 TrialState.PRUNED    acc:83.0% {'cnn_width': 213, 'cnn_depth': 3}
 # 50 TrialState.COMPLETE  acc:97.1% {'cnn_width': 154, 'cnn_depth': 4}
 # 51 TrialState.COMPLETE  acc:98.1% {'cnn_width': 220, 'cnn_depth': 4}
 # 52 TrialState.COMPLETE  acc:98.0% {'cnn_width': 184, 'cnn_depth': 4}
 # 53 TrialState.COMPLETE  acc:97.0% {'cnn_width': 253, 'cnn_depth': 4}
 # 54 TrialState.PRUNED    acc:83.8% {'cnn_width': 148, 'cnn_depth': 3}
 # 55 TrialState.PRUNED    acc:78.4% {'cnn_width': 10, 'cnn_depth': 5}
 # 56 TrialState.PRUNED    acc:92.3% {'cnn_width': 220, 'cnn_depth': 4}
 # 57 TrialState.PRUNED    acc:92.7% {'cnn_width': 166, 'cnn_depth': 4}
 # 58 TrialState.PRUNED    acc:85.8% {'cnn_width': 101, 'cnn_depth': 5}
 # 59 TrialState.PRUNED    acc:84.6% {'cnn_width': 183, 'cnn_depth': 3}
 # 60 TrialState.COMPLETE  acc:98.5% {'cnn_width': 255, 'cnn_depth': 4}
 # 61 TrialState.COMPLETE  acc:97.6% {'cnn_width': 222, 'cnn_depth': 4}
 # 62 TrialState.PRUNED    acc:92.6% {'cnn_width': 242, 'cnn_depth': 4}
 # 63 TrialState.PRUNED    acc:90.1% {'cnn_width': 138, 'cnn_depth': 4}
 # 64 TrialState.PRUNED    acc:86.8% {'cnn_width': 198, 'cnn_depth': 5}
 # 65 TrialState.PRUNED    acc:81.7% {'cnn_width': 173, 'cnn_depth': 3}
 # 66 TrialState.PRUNED    acc:90.2% {'cnn_width': 118, 'cnn_depth': 4}
 # 67 TrialState.PRUNED    acc:87.6% {'cnn_width': 253, 'cnn_depth': 5}
 # 68 TrialState.PRUNED    acc:92.6% {'cnn_width': 157, 'cnn_depth': 4}
 # 69 TrialState.PRUNED    acc:77.5% {'cnn_width': 93, 'cnn_depth': 3}
 # 70 TrialState.PRUNED    acc:87.8% {'cnn_width': 217, 'cnn_depth': 5}
 # 71 TrialState.PRUNED    acc:93.0% {'cnn_width': 192, 'cnn_depth': 4}
 # 72 TrialState.COMPLETE  acc:97.6% {'cnn_width': 186, 'cnn_depth': 4}
 # 73 TrialState.PRUNED    acc:88.3% {'cnn_width': 139, 'cnn_depth': 4}
 # 74 TrialState.PRUNED    acc:79.7% {'cnn_width': 113, 'cnn_depth': 3}
 # 75 TrialState.COMPLETE  acc:97.5% {'cnn_width': 254, 'cnn_depth': 4}
 # 76 TrialState.PRUNED    acc:92.2% {'cnn_width': 212, 'cnn_depth': 4}
 # 77 TrialState.PRUNED    acc:43.1% {'cnn_width': 4, 'cnn_depth': 4}
 # 78 TrialState.PRUNED    acc:81.4% {'cnn_width': 168, 'cnn_depth': 3}
 # 79 TrialState.PRUNED    acc:76.6% {'cnn_width': 23, 'cnn_depth': 5}
 # 80 TrialState.PRUNED    acc:85.6% {'cnn_width': 41, 'cnn_depth': 4}
 # 81 TrialState.COMPLETE  acc:97.5% {'cnn_width': 199, 'cnn_depth': 4}
 # 82 TrialState.PRUNED    acc:92.8% {'cnn_width': 228, 'cnn_depth': 4}
 # 83 TrialState.PRUNED    acc:92.7% {'cnn_width': 149, 'cnn_depth': 4}
 # 84 TrialState.PRUNED    acc:83.5% {'cnn_width': 174, 'cnn_depth': 3}
 # 85 TrialState.PRUNED    acc:86.6% {'cnn_width': 128, 'cnn_depth': 5}
 # 86 TrialState.PRUNED    acc:90.6% {'cnn_width': 231, 'cnn_depth': 4}
 # 87 TrialState.PRUNED    acc:84.3% {'cnn_width': 187, 'cnn_depth': 3}
 # 88 TrialState.COMPLETE  acc:97.8% {'cnn_width': 158, 'cnn_depth': 4}
 # 89 TrialState.PRUNED    acc:92.8% {'cnn_width': 153, 'cnn_depth': 4}
 # 90 TrialState.PRUNED    acc:88.7% {'cnn_width': 106, 'cnn_depth': 4}
 # 91 TrialState.PRUNED    acc:90.6% {'cnn_width': 207, 'cnn_depth': 4}
 # 92 TrialState.COMPLETE  acc:96.8% {'cnn_width': 162, 'cnn_depth': 4}
 # 93 TrialState.PRUNED    acc:86.7% {'cnn_width': 136, 'cnn_depth': 5}
 # 94 TrialState.COMPLETE  acc:98.0% {'cnn_width': 252, 'cnn_depth': 4}
 # 95 TrialState.COMPLETE  acc:97.1% {'cnn_width': 239, 'cnn_depth': 4}
 # 96 TrialState.COMPLETE  acc:98.1% {'cnn_width': 185, 'cnn_depth': 4}
 # 97 TrialState.PRUNED    acc:85.5% {'cnn_width': 174, 'cnn_depth': 5}
 # 98 TrialState.PRUNED    acc:92.3% {'cnn_width': 122, 'cnn_depth': 4}
 # 99 TrialState.PRUNED    acc:83.7% {'cnn_width': 254, 'cnn_depth': 3}
# 100 TrialState.PRUNED    acc:88.5% {'cnn_width': 146, 'cnn_depth': 4}
# 101 TrialState.PRUNED    acc:89.8% {'cnn_width': 205, 'cnn_depth': 4}
# 102 TrialState.PRUNED    acc:92.2% {'cnn_width': 185, 'cnn_depth': 4}
# 103 TrialState.PRUNED    acc:90.5% {'cnn_width': 161, 'cnn_depth': 4}
# 104 TrialState.COMPLETE  acc:97.6% {'cnn_width': 229, 'cnn_depth': 4}
# 105 TrialState.PRUNED    acc:85.5% {'cnn_width': 196, 'cnn_depth': 5}

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
