"""Detect moles in an image, using deep neural nets."""


import gzip
import io
import sys

import cv2
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
    if any((img > 1).any() or (img < 0).any() for img in images):
        raise ValueError("Pixel value must be [0, 1].")

    intersection = (prediction * target).sum()
    total = prediction.sum() + target.sum()
    return (2 * intersection) / total


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


class Dense1x1(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.075
        self.total_steps = 600
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

    # @staticmethod
    # def images_to_data(photo, mask):
    #     photo_hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
    #     blur_photo = cv2.blur(photo, (64, 64))
    #     blur_photo_hsv = cv2.cvtColor(blur_photo, cv2.COLOR_BGR2HSV)
    #     blur_mask = cv2.blur(mask, (64, 64))
    #     return torch.vstack(
    #         [
    #             to_tensor(photo),
    #             to_tensor(photo_hsv),
    #             to_tensor(mask),
    #             to_tensor(blur_photo),
    #             to_tensor(blur_photo_hsv),
    #             to_tensor(blur_mask),
    #         ]
    #     )

    def forward(self, x):
        x1_out = self.l1_bn(x)
        x4_out = self.l4_bn(self.l3_swish(self.l2_cnn(x1_out)))
        x7_in = torch.cat([x1_out, x4_out], dim=1)
        x7_out = self.l7_bn(self.l6_swish(self.l5_cnn(x7_in)))
        x8_in = torch.cat([x7_in, x7_out], dim=1)
        return self.l9_sigmoid(self.l8_cnn(x8_in))

    def training_step(self, batch, batch_nb):
        x, y = batch
        result = self(x)
        cv2.imwrite(
            f"model_001_{self.frame:04}.png",
            result.detach().numpy()[0][0] * 255,
        )
        self.frame += 1
        # target = y[:, 2:3]
        target = y
        assert result.shape == target.shape, (result.shape, target.shape)
        # loss = F.cross_entropy(result, target)
        loss = F.mse_loss(result, target)
        self.log("train/loss", loss)
        # wandb.log({"train/loss": loss})
        return loss

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


class Threshold1x1(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.075
        self.total_steps = 600

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

    def training_step(self, batch, batch_nb):
        x, y = batch
        result = self(x)
        target = y
        assert result.shape == target.shape, (result.shape, target.shape)
        loss = F.mse_loss(result, target)
        self.log("train/loss", loss.detach())
        return loss

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


class CackModel(Threshold1x1):
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

    def on_train_batch_end(self, *_args, **_kwargs):
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
