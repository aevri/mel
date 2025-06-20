"""Train to guess where moles are in a rotomap image."""

import os
import warnings


def setup_parser(parser):
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=10,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="Limit the number of training batches, useful for debugging.",
    )
    parser.add_argument(
        "--limit-valid-batches",
        type=int,
        default=None,
        help="Limit the number of validation batches, useful for debugging.",
    )
    parser.add_argument(
        "--wandb",
        nargs=2,
        metavar=("project", "run_name"),
        help="Use a https://wandb.ai/ logger.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=4,
        help="Number of images to load at a time.",
    )
    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=0,
        help="Number of workers to load batches in parallel.",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate, choose carefully.",
    )
    parser.add_argument(
        "--just-validate",
        action="store_true",
        help="No training, just calc validation score.",
    )
    parser.add_argument(
        "--no-post-validate",
        action="store_true",
        help="No full validation at the end, useful for debugging.",
    )
    parser.add_argument(
        "--min-session", help="e.g. '2020_' to exclude anything pre-2020."
    )


def process_args(args):
    # Some of are expensive imports, so to keep program start-up time lower,
    # import them only when necessary.
    import pytorch_lightning as pl
    import torch

    import mel.lib.ellipsespace
    import mel.lib.fs
    import mel.rotomap.automarknn
    import mel.rotomap.moles

    melroot = mel.lib.fs.find_melroot()
    model_dir = melroot / mel.lib.fs.DEFAULT_CLASSIFIER_PATH
    model_path = model_dir / "detect.pth"

    if model_path.exists():
        if not os.access(model_path, os.W_OK):
            print("No permission to write to", model_path)
            return 1

        print(f"Will fine-tune {model_path}")
        model = mel.rotomap.automarknn.PlModule(model_path)
    else:
        print(f"Will save to {model_path}")
        model = mel.rotomap.automarknn.PlModule()

    model.lr = args.learning_rate

    if not model_dir.exists():
        model_dir.mkdir()

    warnings.filterwarnings(
        "ignore",
        message=(
            r"The dataloader, \w+ dataloader( 0)?, "
            "does not have many workers which may be a bottleneck."
        ),
    )

    print("Making data ..")
    (
        train_images,
        valid_images,
        train_sessions,
        valid_sessions,
    ) = mel.rotomap.automarknn.list_train_valid_images(min_session=args.min_session)
    train_images = mel.rotomap.automarknn.drop_paths_without_moles(train_images)
    valid_images = mel.rotomap.automarknn.drop_paths_without_moles(valid_images)

    def print_sessions(kind, sessions):
        print(f"{kind} image sessions:")
        print()
        for session in sessions:
            print(" ", session)
        print()

    print_sessions("Training", train_sessions)
    print_sessions("Validation", valid_sessions)
    print(f"There are {len(train_images):,} training images.")
    print(f"There are {len(valid_images):,} validation images.")

    train_dataset = mel.rotomap.automarknn.MoleImageBoxesDataset(train_images)
    valid_dataset = mel.rotomap.automarknn.MoleImageBoxesDataset(valid_images)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=mel.rotomap.automarknn.collate_fn,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        collate_fn=mel.rotomap.automarknn.collate_fn,
        shuffle=True,
        num_workers=args.num_workers,
    )

    trainer_kwargs = {
        "log_every_n_steps": 5,
        "enable_checkpointing": False,
        "accelerator": "auto",
        # "accumulate_grad_batches": args.accumulate_grad_batches,
        "max_epochs": args.epochs,
        # "max_epochs": 1,
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_valid_batches,
        "val_check_interval": 50 if len(train_loader) > 50 else None,
        # "auto_lr_find": True,
    }

    if args.wandb:
        wandb_project, wandb_run_name = args.wandb
        trainer_kwargs |= {
            "logger": pl.loggers.WandbLogger(project=wandb_project, name=wandb_run_name)
        }

    if not args.just_validate:
        trainer = pl.Trainer(**trainer_kwargs)

        # model.train()
        # trainer.tune(model, train_loader)
        print(f"Learning rate: {model.lr:0.8f}")

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )

        torch.save(model.model.state_dict(), model_path)
        print(f"Saved {model_path}")

    if not args.no_post_validate:
        pl.Trainer(accelerator="auto").validate(
            model,
            torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=args.batch_size,
                collate_fn=mel.rotomap.automarknn.collate_fn,
                num_workers=args.num_workers,
            ),
        )

    if args.wandb:
        import wandb

        wandb.finish()
        return None
    return None


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
