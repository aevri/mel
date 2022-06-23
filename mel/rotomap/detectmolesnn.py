"""Detect moles in an image, using deep neural nets."""

import cv2
import torch
import torchvision

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


class CackModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.075
        self.epochs = 600
        self.l1_bn = torch.nn.BatchNorm2d(14)
        self.l2_cnn = torch.nn.Conv2d(
            in_channels=14, out_channels=3, kernel_size=1, padding=0
        )
        self.l3_swish = Swish()
        self.l4_bn = torch.nn.BatchNorm2d(3)
        self.l5_cnn = torch.nn.Conv2d(
            in_channels=17, out_channels=3, kernel_size=1, padding=0
        )
        self.l6_swish = Swish()
        self.l7_bn = torch.nn.BatchNorm2d(3)
        self.l8_cnn = torch.nn.Conv2d(
            in_channels=20, out_channels=1, kernel_size=1, padding=0
        )
        self.l9_sigmoid = torch.nn.Sigmoid()

        self.frame = 0

    @staticmethod
    def images_to_data(photo, mask):
        photo_hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
        blur_photo = cv2.blur(photo, (64, 64))
        blur_photo_hsv = cv2.cvtColor(blur_photo, cv2.COLOR_BGR2HSV)
        blur_mask = cv2.blur(mask, (64, 64))
        return torch.vstack(
            [
                to_tensor(photo),
                to_tensor(photo_hsv),
                to_tensor(mask),
                to_tensor(blur_photo),
                to_tensor(blur_photo_hsv),
                to_tensor(blur_mask),
            ]
        )

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
        target = y[:, 2:3]
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
            steps_per_epoch=1,
            epochs=self.epochs,
        )

        sched = {
            "scheduler": self.scheduler,
            "interval": "step",
        }
        return [self.optimizer], [sched]


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
