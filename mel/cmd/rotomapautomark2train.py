"""Train to automatically mark moles on rotomap images."""

import json
import sys

import torch

import mel.lib.math
import mel.rotomap.detectmolesnn
import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print information about the processing.",
    )


def process_args(args):
    try:
        melroot = mel.lib.fs.find_melroot()
    except mel.lib.fs.NoMelrootError:
        print("Not in a mel repo, could not find melroot", file=sys.stderr)
        return 1

    model_dir = melroot / mel.lib.fs.DEFAULT_CLASSIFIER_PATH
    model_path = model_dir / "detectmoles.pth"
    metadata_path = model_dir / "detectmoles.json"
    print(f"Will save to {model_path}")
    print(f"         and {metadata_path}")

    batch_size = 512
    max_lr = 0.01
    num_epochs = 50

    parts_path = melroot / mel.lib.fs.ROTOMAPS_PATH / "parts"

    # all_images = list(parts_path.glob("LeftLeg/*/*/*.jpg"))
    all_images = [
        path
        for path in parts_path.glob("*/*/*/*.jpg")
        if ("Left" in str(path) or "Right" in str(path))
        and ("Lower" in str(path) or "Upper" in str(path))
        and "2015" not in str(path)
        and "2016" not in str(path)
    ]
    # all_images = [
    #     path
    #     for path in parts_path.glob("*/*/*/*.jpg")
    #     if ("Left" in str(path) or "Right" in str(path))
    #     and ("Lower" in str(path) or "Upper" in str(path))
    # ]

    all_parts = sorted(
        {mel.rotomap.detectmolesnn.image_path_to_part(i) for i in all_images}
    )
    part_to_id = {part: i for i, part in enumerate(all_parts)}

    training_images = [path for path in all_images if not "2019_" in str(path)]
    training_dataloader = load_dataset(training_images, batch_size)
    validation_images = [path for path in all_images if "2019_" in str(path)]
    validation_dataloader = load_dataset(validation_images, batch_size)

    # resnet18_num_features = 512
    # resnet18_num_features = 7680
    # resnet18_num_features = 69120
    # resnet50_num_features = 2048
    # resnet_num_features = resnet18_num_features
    num_features = None
    for batch in training_dataloader:
        activations_batch = batch[1]
        assert len(activations_batch.shape) == 2
        num_features = len(activations_batch[0])
        break
    num_intermediate = 80
    num_layers = 2
    # model = mel.rotomap.detectmolesnn.NeighboursLinearSigmoidModel2(
    model = mel.rotomap.detectmolesnn.LinearSigmoidModel2(
        part_to_id, num_features, num_intermediate, num_layers
    )

    mel.rotomap.detectmolesnn.train(
        model,
        training_dataloader,
        validation_dataloader,
        loss_func,
        max_lr,
        num_epochs,
    )

    with open(metadata_path, "w") as f:
        json.dump(model.init_dict(), f)
    print(f"Saved {metadata_path}")
    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_path}")


def loss_func(in_, target):
    scale = target[:, 0].unsqueeze(1)
    assert scale.shape == (len(in_), 1)
    scale1 = (scale * 10) + 1
    # scale1 = scale + 1
    # loss1 = torch.nn.functional.mse_loss(in_[:, 0], target[:, 0])
    loss1 = torch.nn.functional.mse_loss(
        in_[:, 0] + scale1, target[:, 0] + scale1
    )

    pos_diff = in_[:, 1:] - target[:, 1:]
    pos_diff_sq = pos_diff ** 2
    dist_sq = (pos_diff_sq[:, 0] + pos_diff_sq[:, 1]).unsqueeze(1)
    assert dist_sq.shape == (len(in_), 1), f"{dist_sq.shape}"

    loss2 = torch.nn.functional.mse_loss(
        dist_sq * scale, torch.zeros(len(in_), 1)
    )

    # loss2 = torch.nn.functional.mse_loss(
    #     in_[:, 1:] * scale, target[:, 1:] * scale
    # )

    return loss1 + loss2


def load_dataset(images, batch_size):
    print(f"Will load from {len(images)} images.")
    dataset = mel.rotomap.detectmolesnn.TileDataset(images, 32)
    print(f"Loaded {len(dataset)} tiles.")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
    )
    return dataloader
    # neighbours_dataset = mel.rotomap.detectmolesnn.NeighboursDataset(dataset)
    # print(f"Got {len(neighbours_dataset)} 3x3 tiles.")
    # neighbours_dataloader = torch.utils.data.DataLoader(
    #     neighbours_dataset, batch_size=batch_size, shuffle=True,
    # )
    # return neighbours_dataloader


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
