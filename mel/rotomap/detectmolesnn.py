"""Detect moles in an image, using deep neural nets."""


import gzip
import io
import sys

import torch
import torchvision
import tqdm

import pytorch_lightning as pl
from torch.nn import functional as F

# import PIL
# import wandb


import mel.lib.math
import mel.rotomap.moles

X_OFFSET = 1
Y_OFFSET = 1
TILE_MAGNIFICATION = 1


to_tensor = torchvision.transforms.ToTensor()


# ismole
# photo_red
# photo_green
# photo_blue
# mask
# blur64_red
# blur64_green
# blur64_blue
# blur64_mask


def dice_loss(prediction, target):
    images = [prediction, target]
    # for img in images:
    #     if "NCHW" != "".join(img.names):
    #         raise ValueError("Image names must be NCHW, got:", img.names)
    if not all(len(img.shape) == 4 for img in images):
        raise ValueError(
            "Images must be of rank 4.",
            [img.shape for img in images],
        )
    if not all(img.shape[0] == images[0].shape[0] for img in images):
        raise ValueError(
            "Images must have the same number of fragments.",
            [img.shape for img in images],
        )
    if not all(img.shape[2:4] == (1, 1) for img in images):
        raise ValueError(
            "Images must be 1x1 tiles.",
            [img.shape for img in images],
        )
    if any((img > 1).any() or (img < 0).any() for img in images):
        raise ValueError("Pixel value must be [0, 1].")

    intersection = (prediction * target).sum()
    total = prediction.sum() + target.sum()
    loss = 1 - ((2 * intersection) / total)
    assert loss >= 0, loss
    assert loss <= 1, loss
    return loss


def precision_ish(prediction, target):
    images = [prediction, target]
    if not all(len(img.shape) == 4 for img in images):
        raise ValueError(
            "Images must be of rank 4.",
            [img.shape for img in images],
        )
    if not all(img.shape[0] == images[0].shape[0] for img in images):
        raise ValueError(
            "Images must have the same number of fragments.",
            [img.shape for img in images],
        )
    if not all(img.shape[2:4] == (1, 1) for img in images):
        raise ValueError(
            "Images must be 1x1 tiles.",
            [img.shape for img in images],
        )
    if any((img > 1).any() or (img < 0).any() for img in images):
        raise ValueError("Pixel value must be [0, 1].")

    result = (prediction * target).sum() / prediction.sum()
    assert result >= 0, result
    assert result <= 1, result
    return result


def recall_ish(prediction, target):
    images = [prediction, target]
    if not all(len(img.shape) == 4 for img in images):
        raise ValueError(
            "Images must be of rank 4.",
            [img.shape for img in images],
        )
    if not all(img.shape[0] == images[0].shape[0] for img in images):
        raise ValueError(
            "Images must have the same number of fragments.",
            [img.shape for img in images],
        )
    if not all(img.shape[2:4] == (1, 1) for img in images):
        raise ValueError(
            "Images must be 1x1 tiles.",
            [img.shape for img in images],
        )
    if any((img > 1).any() or (img < 0).any() for img in images):
        raise ValueError("Pixel value must be [0, 1].")

    result = (prediction * target).sum() / target.sum()
    assert result >= 0, result
    assert result <= 1, result
    return result


def sorted_unique_images_sync(image_a, image_b):
    images = [image_a, image_b]
    for img in images:
        if "NCHW" != "".join(img.names):
            raise ValueError("Image names must be NCHW, got:", img.names)
    if not all(img.shape[0] == images[0].shape[0] for img in images):
        raise ValueError(
            "Images must have the same number of fragments.",
            [img.shape for img in images],
        )
    if not all(img.shape[2:4] == (1, 1) for img in images):
        raise ValueError(
            "Images must be 1x1 tiles.",
            [img.shape for img in images],
        )

    temp = torch.cat([image_a.rename(None), image_b.rename(None)], dim=1)
    assert temp.shape == (
        image_a.shape[0],
        image_a.shape[1] + image_b.shape[1],
        image_a.shape[2],
        image_a.shape[3],
    )
    temp = torch.unique(temp, dim=0)
    assert temp.shape[1:] == (
        image_a.shape[1] + image_b.shape[1],
        image_a.shape[2],
        image_a.shape[3],
    )

    image_a, image_b = torch.split(
        temp, [image_a.shape[1], image_b.shape[1]], dim=1
    )

    return image_a.rename(*list("NCHW")), image_b.rename(*list("NCHW"))


def shuffled_images_sync(*images):
    if not images:
        raise ValueError("No images supplied")
    for img in images:
        if "NCHW" != "".join(img.names):
            raise ValueError("Image names must be NCHW, got:", img.names)
    if not all(img.shape[0] == images[0].shape[0] for img in images):
        raise ValueError(
            "Images must have the same number of fragments.",
            [img.shape for img in images],
        )
    if not all(img.shape[2:4] == (1, 1) for img in images):
        raise ValueError(
            "Images must be 1x1 tiles.",
            [img.shape for img in images],
        )
    indices = torch.randperm(images[0].shape[0])
    return [
        img.rename(None).index_select(0, indices).rename(*list("NCHW"))
        for img in images
    ]


def select_not_masked(image, mask):
    if "NCHW" != "".join(image.names):
        raise ValueError("Image names must be NCHW, got:", image.names)
    if "NCHW" != "".join(mask.names):
        raise ValueError("Mask names must be NCHW, got:", mask.names)
    if 1 != mask.shape[1]:
        raise ValueError("Mask must have one channel, got shape:", mask.shape)
    mask_indices = mask.rename(None).nonzero()[:, 0]
    return (
        image.rename(None).index_select(0, mask_indices).rename(*list("NCHW"))
    )


def pixelise(image):
    """Convert a CHW image into NCHW NxCx1x1 tiles.

    Examples:

        >>> import torch
        >>> t = torch.tensor([[[1, 2, 3]]], names=list("CHW"))
        >>> p = torch.tensor([[[[1]]], [[[2]]], [[[3]]]], names=list("NCHW"))
        >>> torch.equal(pixelise(t), p)
        True

    """
    if "CHW" != "".join(image.names):
        raise ValueError("Tensor names must be CHW, got:", image.names)
    num_channels = image.shape[0]
    p_image = (
        image.rename(None).reshape([num_channels, -1, 1, 1]).movedim(0, 1)
    )
    return p_image.rename(*list("NCHW"))


class ConstantModel(torch.nn.Module):
    def __init__(self, constant_value):
        super().__init__()
        self.constant_value = constant_value

    def init_dict(self):
        return {}

    def forward(self, images):
        assert len(images.shape) == 4
        shape = list(images.shape)
        shape[1] = 1
        result = torch.empty(shape)
        result[:, :, :, :] = self.constant_value
        result = result.to(images.device)
        return result


def image_path_to_part(image_path):
    subpart_path = image_path.parent.parent
    part_path = subpart_path.parent
    return f"{part_path.name}, {subpart_path.name}"


def locations_image(moles, image_width, image_height):
    image = mel.lib.common.new_image(image_height, image_width)

    mole_points = [
        (m["x"], m["y"])
        for m in moles
        if "looks_like" not in m or m["looks_like"] == "mole"
    ]

    for x, y in mole_points:
        mel.lib.common.draw_circle(image, x, y, 32, (0, 0, 255))

    return image


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def mean_l1(model):
    l1_sum = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return l1_sum / p_count


def mean(*args):
    return sum(args) / len(args)


class Model(pl.LightningModule):
    def __init__(self, total_steps):
        super().__init__()
        self.learning_rate = 0.075
        self.total_steps = total_steps

    def training_step(self, batch, batch_nb):
        x, y = batch
        result = self(x)
        target = y
        assert result.shape == target.shape, (result.shape, target.shape)
        # loss = dice_loss(result, target)
        # loss = F.mse_loss(result, target) * 0.999 + mean_l1(self) * 0.001
        loss = F.mse_loss(result, target)
        self.log("train/loss", loss.detach())
        return {
            "loss": loss,
            "dice": dice_loss(result, target),
            "pres": precision_ish(result, target),
            "rec": recall_ish(result, target),
            "mse": F.mse_loss(result, target),
            "mul1": mean_l1(self),
        }

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(), self.learning_rate
        )

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=self.total_steps,
        )

        sched = {
            "scheduler": self.scheduler,
            "interval": "step",
        }
        return [self.optimizer], [sched]

    def print_details(self):
        print(self)
        print()
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)


class Dense1x1(Model):
    def __init__(self, total_steps):
        super().__init__(total_steps)
        self.l1_bn = torch.nn.BatchNorm2d(13)
        self.l2_cnn = torch.nn.Conv2d(
            in_channels=13, out_channels=3, kernel_size=1, padding=0
        )
        self.l3_swish = Swish()
        self.l4_bn = torch.nn.BatchNorm2d(3)
        self.l5_cnn = torch.nn.Conv2d(
            in_channels=16, out_channels=3, kernel_size=1, padding=0
        )
        self.l6_swish = Swish()
        self.l7_bn = torch.nn.BatchNorm2d(3)
        self.l8_cnn = torch.nn.Conv2d(
            in_channels=19, out_channels=1, kernel_size=1, padding=0
        )
        self.l9_sigmoid = torch.nn.Sigmoid()

        self.frame = 0

    def forward(self, x):
        x1_out = self.l1_bn(x)
        x4_out = self.l4_bn(self.l3_swish(self.l2_cnn(x1_out)))
        x7_in = torch.cat([x1_out, x4_out], dim=1)
        x7_out = self.l7_bn(self.l6_swish(self.l5_cnn(x7_in)))
        x8_in = torch.cat([x7_in, x7_out], dim=1)
        return self.l9_sigmoid(self.l8_cnn(x8_in))

    def print_details(self):
        super().print_details()

        channel_names = [
            "photo_B",
            "photo_G",
            "photo_R",
            "photo_hsv_H",
            "photo_hsv_S",
            "photo_hsv_V",
            "blur_photo_B",
            "blur_photo_G",
            "blur_photo_R",
            "blur_photo_hsv_H",
            "blur_photo_hsv_S",
            "blur_photo_hsv_V",
            "blur_mask",
        ]

        for cnn in [self.l2_cnn, self.l5_cnn, self.l8_cnn]:
            print()
            for i, name in enumerate(channel_names):
                print(f"{name:20} ", end="")
                for output in cnn.weight:
                    print(f"{output[i][0][0].item(): .3f}  ", end="")
                print()


class Dense1x1HueSatMask(Model):
    def __init__(self, total_steps):
        super().__init__(total_steps)
        image_channels = 5
        self.l1_bn = torch.nn.BatchNorm2d(image_channels)
        self.l2_cnn = torch.nn.Conv2d(
            in_channels=image_channels,
            out_channels=3,
            kernel_size=1,
            padding=0,
        )
        self.l3_swish = Swish()
        self.l4_bn = torch.nn.BatchNorm2d(3)
        self.l5_cnn = torch.nn.Conv2d(
            in_channels=image_channels + 3,
            out_channels=3,
            kernel_size=1,
            padding=0,
        )
        self.l6_swish = Swish()
        self.l7_bn = torch.nn.BatchNorm2d(3)
        self.l8_cnn = torch.nn.Conv2d(
            in_channels=image_channels + 6,
            out_channels=1,
            kernel_size=1,
            padding=0,
        )
        self.l9_sigmoid = torch.nn.Sigmoid()

        self.frame = 0

    def forward(self, orig_x):
        sat_channel = 4
        hue_channel = 3
        blur_hue_channel = 9
        blur_sat_channel = 10
        blur_mask_channel = 12
        x = orig_x[
            :,
            [
                sat_channel,
                hue_channel,
                blur_hue_channel,
                blur_sat_channel,
                blur_mask_channel,
            ],
            :,
            :,
        ]
        x1_out = self.l1_bn(x)
        x4_out = self.l4_bn(self.l3_swish(self.l2_cnn(x1_out)))
        x7_in = torch.cat([x1_out, x4_out], dim=1)
        x7_out = self.l7_bn(self.l6_swish(self.l5_cnn(x7_in)))
        x8_in = torch.cat([x7_in, x7_out], dim=1)
        return self.l9_sigmoid(self.l8_cnn(x8_in))

    def print_details(self):
        super().print_details()

        channel_names = [
            "photo_hsv_H",
            "photo_hsv_S",
            "blur_photo_hsv_H",
            "blur_photo_hsv_S",
            "blur_mask",
        ]

        for cnn in [self.l2_cnn, self.l5_cnn, self.l8_cnn]:
            print()
            for i, name in enumerate(channel_names):
                print(f"{name:20} ", end="")
                for output in cnn.weight:
                    print(f"{output[i][0][0].item(): .3f}  ", end="")
                print()


class Conv1x1HueSatMask(Model):
    def __init__(self, total_steps):
        super().__init__(total_steps)
        image_channels = 5
        width = 10
        self.cnn = torch.nn.Sequence(
            [
                torch.nn.BatchNorm2d(image_channels),
                torch.nn.Conv2d(
                    in_channels=image_channels,
                    out_channels=width,
                    kernel_size=1,
                    padding=0,
                ),
                Swish(),
                torch.nn.BatchNorm2d(3),
                torch.nn.Conv2d(
                    in_channels=width,
                    out_channels=width,
                    kernel_size=1,
                    padding=0,
                ),
                Swish(),
                torch.nn.BatchNorm2d(3),
                torch.nn.Conv2d(
                    in_channels=width,
                    out_channels=1,
                    kernel_size=1,
                    padding=0,
                ),
                torch.nn.Sigmoid(),
            ]
        )

    def forward(self, orig_x):
        sat_channel = 4
        hue_channel = 3
        blur_hue_channel = 9
        blur_sat_channel = 10
        blur_mask_channel = 12
        x = orig_x[
            :,
            [
                sat_channel,
                hue_channel,
                blur_hue_channel,
                blur_sat_channel,
                blur_mask_channel,
            ],
            :,
            :,
        ]
        return self.cnn(x)

    def print_details(self):
        super().print_details()

        channel_names = [
            "photo_hsv_H",
            "photo_hsv_S",
            "blur_photo_hsv_H",
            "blur_photo_hsv_S",
            "blur_mask",
        ]

        for i, name in enumerate(channel_names):
            print(f"{name:20} ", end="")
            for output in self.cnn[1].weight:
                print(f"{output[i][0][0].item(): .3f}  ", end="")
            print()


class Threshold1x1(Model):
    def __init__(self, total_steps):
        super().__init__(total_steps)
        self.min_sat = torch.nn.Parameter(torch.tensor(0.0))
        self.max_sat = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError("Expected NCHW.")

        sat_channel = 4

        y = x[:, sat_channel : sat_channel + 1, :, :] - self.min_sat
        y /= self.max_sat
        y = torch.sigmoid(y)

        assert len(y.shape) == len(x.shape), (x.shape, y.shape)

        return y


class CackModel(Conv1x1HueSatMask):
    pass


def collate(pretrained_list):
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torch

    data_list = []

    for path in pretrained_list:
        zipin = gzip.open(path, "rb")
        # See https://github.com/pytorch/pytorch/issues/55777
        # Oddly much faster to load from BytesIO instead of zip directly.
        bytesin = io.BytesIO(zipin.read())
        data_list.append(torch.load(bytesin))

    return (
        torch.cat([d["x_data"] for d in data_list]),
        torch.cat([d["y_data"] for d in data_list]),
    )


class GlobalProgressBar(pl.callbacks.progress.ProgressBarBase):
    def __init__(self, process_position: int = 0):
        super().__init__()
        self._process_position = process_position
        self._enabled = True
        self.main_progress_bar = None

    def __getstate__(self):
        # can't pickle the tqdm objects
        state = self.__dict__.copy()
        state["main_progress_bar"] = None
        return state

    @property
    def process_position(self) -> int:
        return self._process_position

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.main_progress_bar = tqdm.tqdm(
            desc="Total Steps",
            initial=0,
            total=trainer.max_steps,
            position=(2 * self.process_position),
            disable=False,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )

    def on_train_end(self, trainer, pl_module):
        self.main_progress_bar.close()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        desc = " ".join(
            f"{name}:{val.item():.3}" for name, val in outputs.items()
        )
        self.main_progress_bar.set_description(desc)
        self.main_progress_bar.update(1)


# -----------------------------------------------------------------------------
# Copyright (C) 2022 Angelos Evripiotis.
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
