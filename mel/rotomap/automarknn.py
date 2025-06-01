"""Automatically mark moles on rotomap images."""

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision

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
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
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


def calc_precision_recall(target_poslist, poslist, error_distance=5):
    if not len(poslist):
        return 0, 0
    vec_matches, vec_missing, vec_added = mel.rotomap.automark.match_pos_vecs(
        target_poslist, poslist, error_distance
    )
    precision = len(vec_matches) / (len(vec_matches) + len(vec_added))
    recall = len(vec_matches) / (len(vec_matches) + len(vec_missing))
    return precision, recall


class PlModule(pl.LightningModule):
    def __init__(self, model_path=None):
        super().__init__()
        # self.lr = 0.0000229  # As determined by pl auto_lr_find. Not good.
        # self.lr = 0.01  # Too high.
        # self.lr = 0.0001  # Too low? Validation worse than '0.001'.
        self.lr = 0.001
        self.model = make_model(model_path)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        self.model.train()  # Oddly this seemst to be necessary.
        assert self.model.training
        loss_dict = self.model(x, y)
        if not isinstance(loss_dict, dict):
            raise ValueError(f"Expected dict, got: {loss_dict}")
        losses = sum(loss for loss in loss_dict.values())
        self.log("train_loss", losses.detach())
        return losses

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        # self.model.train()
        result = self.model(x, y)
        precision_list = []
        recall_list = []
        for y_item, result_item in zip(y, result):
            (
                item_precision,
                item_recall,
            ) = calc_precision_recall(
                target_poslist=boxes_to_poslist(y_item["boxes"]),
                poslist=boxes_to_poslist(result_item["boxes"]),
            )
            precision_list.append(item_precision)
            recall_list.append(item_recall)
        precision = sum(precision_list) / len(precision_list)
        recall = sum(recall_list) / len(recall_list)
        self.log("val_prec", precision, prog_bar=True)
        self.log("val_reca", recall, prog_bar=True)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.model.backbone.parameters(),
                    "lr": self.lr * 0.01,
                },
                {
                    "params": self.model.rpn.parameters(),
                    "lr": self.lr * 1.0,
                },
                {
                    "params": self.model.roi_heads.parameters(),
                    "lr": self.lr * 1.0,
                },
            ],
            lr=self.lr,
        )
        return optimizer


class MoleImageBoxesDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                     std=[0.229, 0.224, 0.225])
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]

        image = load_image(path)
        image = self.image_transform(image)

        moles = mel.rotomap.moles.load_image_moles(path)
        if not moles:
            raise ValueError("Mole list must not be empty.")
        fr = 10
        boxes = [[m["x"] - fr, m["y"] - fr, m["x"] + fr, m["y"] + fr] for m in moles]

        target = {}
        target["labels"] = torch.ones((len(boxes),), dtype=torch.int64)
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        return image, target


# This is useful for debugging sometimes.
#
# def rgb_tensor_to_image(tensor):
#     image = tensor.detach().numpy() * 255
#     image = np.uint8(image)
#     image = image.transpose((1, 2, 0))
#     return image


def list_train_valid_images(min_session=None):
    melroot = mel.lib.fs.find_melroot()
    parts_path = melroot / mel.lib.fs.ROTOMAPS_PATH / "parts"
    exclude_parts = [
        "LeftArm/Hand",
        "RightArm/Hand",
        "LeftLeg/Foot",
        "RightLeg/Foot",
        "Trunk/Lower",
        "Trunk/Back",
    ]
    session_images = mel.lib.fs.list_rotomap_images_by_session(
        parts_path, exclude_parts=exclude_parts
    )
    sessions = sorted(session_images.keys())
    if min_session:
        sessions = [s for s in sessions if s > min_session]
    train_sessions = sessions[:-1]
    valid_sessions = sessions[-1:]
    train_images = [img for sess in train_sessions for img in session_images[sess]]
    valid_images = [img for sess in valid_sessions for img in session_images[sess]]
    return train_images, valid_images, train_sessions, valid_sessions


def drop_paths_without_moles(path_list):
    return [path for path in path_list if mel.rotomap.moles.load_image_moles(path)]


# See https://github.com/pytorch/vision/blob/59ec1dfd550652a493cb99d5704dcddae832a204/references/detection/utils.py#L203
def collate_fn(batch):
    return tuple(zip(*batch))
