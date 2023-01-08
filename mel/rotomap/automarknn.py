"""Automatically mark moles on rotomap images."""

import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import wandb

import mel.lib.common
import mel.lib.fs
import mel.rotomap.moles


def make_detector():
    melroot = mel.lib.fs.find_melroot()
    model_dir = melroot / mel.lib.fs.DEFAULT_CLASSIFIER_PATH
    model_path = model_dir / "detect.pth"
    return MoleDetector(model_path)


class MoleDetector:
    def __init__(self, model_path):
        self.model = make_model(model_path)
        self.image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )

    def get_moles(self, frame):
        image = load_image(frame.path)
        image = self.image_transform(image)

        self.model.eval()
        with torch.no_grad():
            boxes = self.model(image.unsqueeze(0))[0]["boxes"]
        poslist = boxes_to_poslist(boxes)
        moles = []
        for x, y in poslist:
            mel.rotomap.moles.add_mole(moles, int(x), int(y))
        return moles


def make_model(model_path=None):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"
    )
    # set_parameter_no_grad(model)
    num_classes = 2  # 1 class + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_image(image_path):
    # flags = cv2.IMREAD_UNCHANGED + cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR
    flags = cv2.IMREAD_COLOR
    try:
        original_image = cv2.imread(str(image_path), flags)
        if original_image is None:
            raise OSError(f"File not recognized by opencv: {image_path}")
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise OSError(f"Error handling image at: {image_path}") from e

    mask = mel.rotomap.mask.load(image_path)
    green = np.zeros(original_image.shape, np.uint8)
    green[:, :, 1] = 255
    image = cv2.bitwise_and(original_image, original_image, mask=mask)
    not_mask = cv2.bitwise_not(mask)
    green = cv2.bitwise_and(green, green, mask=not_mask)
    image = cv2.bitwise_or(image, green)
    return image


def boxes_to_poslist(boxes):
    poslist = [
        [int(0.5 * (xmin + xmax)), int(0.5 * (ymin + ymax))]
        for xmin, ymin, xmax, ymax in boxes
    ]
    return np.array(poslist)
