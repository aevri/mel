"""Train to automatically mark moles on rotomap images."""

import json
import math
import sys

import tqdm

import torch

import mel.lib.math
import mel.rotomap.detectmolesnn
import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        "IMAGES",
        nargs="+",
        help="A list of paths to images to learn to automark.",
    )
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
        default=1800,
        help="Number of epochs to train for.",
    )


def process_args(args):
    try:
        melroot = mel.lib.fs.find_melroot()
    except mel.lib.fs.NoMelrootError:
        print("Not in a mel repo, could not find melroot", file=sys.stderr)
        return 1

    cpu_device = torch.device("cpu")
    device = torch.device("cuda") if torch.cuda.is_available() else cpu_device

    print("Will train on", device)

    model_dir = melroot / mel.lib.fs.DEFAULT_CLASSIFIER_PATH
    model_path = model_dir / "detectmoles.pth"
    metadata_path = model_dir / "detectmoles.json"

    if model_path.exists():
        if not metadata_path.exists():
            raise Exception(
                f"Metadata for model does not exist: {metadata_path}"
            )
        print(f"Will fine-tune {model_path}")
        print(f"           and {metadata_path}")
        with open(metadata_path) as f:
            init_dict = json.load(f)
        model = mel.rotomap.detectmolesnn.DenseUnet(**init_dict)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Will save to {model_path}")
        print(f"         and {metadata_path}")

        model = mel.rotomap.detectmolesnn.DenseUnet(
            channels_in=3, channels_per_layer=16, num_classes=1
        )

    training_images = args.IMAGES

    print(f"Got {len(training_images)} images to train on.")

    batch_size = 40
    num_epochs = args.epochs
    max_lr = 0.01
    tile_size = 512
    max_dist = 16
    loss_magnitude = 10

    train(
        model,
        device,
        training_images,
        batch_size,
        tile_size,
        num_epochs,
        max_lr,
        max_dist,
        loss_magnitude,
    )

    # with open(metadata_path, "w") as f:
    #     json.dump(model.init_dict(), f)
    # print(f"Saved {metadata_path}")
    # torch.save(model.state_dict(), model_path)
    # print(f"Saved {model_path}")

    # dt_string = mel.lib.datetime.make_datetime_string(
    #     datetime.datetime.utcnow()
    # )
    # log_path = model_dir / f"{dt_string}_detectmoles.log.json"
    # with open(log_path, "w") as f:
    #     json.dump(train_log_dict, f)
    # print(f"Saved {log_path}")


def train(
    model,
    device,
    training_images,
    batch_size,
    tile_size,
    num_epochs,
    max_lr,
    max_dist,
    loss_magnitude,
):
    model.to(device)

    frame_dataset = mel.rotomap.detectmolesnn.IterableFrameDataset(
        training_images,
        tile_size=tile_size,
        num_repeats=40,
        cache_size=40,
        max_dist=max_dist,
        num_searches=10,
    )

    dataloader = torch.utils.data.DataLoader(
        frame_dataset, batch_size=batch_size,
    )

    num_batches = math.ceil(frame_dataset.len() / batch_size)

    # optimizer = torch.optim.AdamW(model.parameters())
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     steps_per_epoch=num_batches,
    #     max_lr=max_lr,
    #     epochs=num_epochs,
    # )

    def loss_func(output, target):
        return mel.rotomap.detectmolesnn.image_loss_max_dist(
            output, target, max_dist, loss_magnitude,
        )

    loss = mel.rotomap.detectmolesnn.validate_epoch(
        device,
        model,
        dataloader,
        loss_func,
        ["image"],
        ["expected_image"],
        num_batches,
    )
    print(f"Loss: {float(loss):.4g}")

    # with tqdm.auto.tqdm(range(num_epochs)) as bar:
    #     for epoch in bar:
    #         loss = mel.rotomap.detectmolesnn.train_epoch(
    #             device,
    #             model,
    #             dataloader,
    #             loss_func,
    #             optimizer,
    #             scheduler,
    #             ["image"],
    #             ["expected_image"],
    #             num_batches,
    #         )
    #         bar.set_description(f"Loss: {float(loss):.4g}")
    #         if not (epoch % 8):
    #             tqdm.tqdm.write(f"Epoch {epoch}: loss {float(loss):.4g}")
    #     tqdm.tqdm.write(f"Epoch {epoch}: loss {float(loss):.4g}")


# -----------------------------------------------------------------------------
# Copyright (C) 2020 Angelos Evripiotis.
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
