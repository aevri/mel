"""Automatically mark moles on rotomap images."""

import pathlib

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
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"
    )
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
        loss_dict = self.model(x, y)
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


class TileHandler:
    def __init__(self, image_shape, tile_size=800, min_overlap=0.1):
        """
        Initialize a TileHandler object.

        >>> th = TileHandler((1600, 1000), tile_size=800, min_overlap=0.1)
        >>> th.tile_counts
        array([3, 2])
        >>> th.overlaps
        array([400, 600])

        >>> th = TileHandler((1601, 999), tile_size=800, min_overlap=0.1)
        >>> th.tile_counts
        array([3, 2])
        >>> th.overlaps
        array([400, 601])

        >>> th = TileHandler((4032, 3024), tile_size=800, min_overlap=0.1)
        >>> th.tile_counts
        array([6, 5])
        >>> th.tile((6 * 5) - 1)
        (3230, 4030, 2224, 3024)
        >>> th.overlaps
        array([154, 244])

        """
        self.image_shape = np.array(image_shape)
        self.tile_size = tile_size
        self.min_overlap = min_overlap
        self.tile_counts, self.overlaps = self._calculate_num_tiles()
        self.num_tiles = self.tile_counts.prod()

    def _calculate_num_tiles(self):
        """
        Calculate the number of tiles in x and y directions and the overlaps.
        """
        effective_tile_size = self.tile_size - np.floor(
            self.tile_size * self.min_overlap
        )
        tile_counts = np.ceil(self.image_shape / effective_tile_size).astype(
            int
        )
        overlaps = np.ceil(
            (tile_counts * self.tile_size - self.image_shape)
            / (tile_counts - 1)
        ).astype(int)

        return tile_counts, overlaps

    def tile(self, tile_index):
        """
        Get the coordinates (y_start, y_end, x_start, x_end) for a tile.

        >>> th = TileHandler((1600, 1000), tile_size=800, min_overlap=0.1)
        >>> th.tile(0)
        (0, 800, 0, 800)
        >>> th.tile(1)
        (0, 800, 200, 1000)
        >>> th.tile(2)
        (400, 1200, 0, 800)
        >>> th.tile(3)
        (400, 1200, 200, 1000)
        >>> th.tile(4)
        (800, 1600, 0, 800)
        >>> th.tile(5)
        (800, 1600, 200, 1000)
        >>> th.tile(-1)
        Traceback (most recent call last):
          ...
        IndexError: ('Out of bounds', -1)
        >>> th.tile(6)
        Traceback (most recent call last):
          ...
        IndexError: ('Out of bounds', 6)

        """
        if tile_index < 0:
            raise IndexError("Out of bounds", tile_index)
        if tile_index >= self.num_tiles:
            raise IndexError("Out of bounds", tile_index)
        tile_row_col = np.array(
            [
                tile_index // self.tile_counts[1],
                tile_index % self.tile_counts[1],
            ]
        )
        starts = np.maximum(
            0, tile_row_col * self.tile_size - tile_row_col * self.overlaps
        )
        ends = np.minimum(
            self.image_shape,
            (tile_row_col + 1) * self.tile_size - tile_row_col * self.overlaps,
        )

        return starts[0], ends[0], starts[1], ends[1]


def clip_boxes(boxes, x_start, y_start, x_end, y_end):
    """Drop boxes that are fully outside the bounds of a tile.

    >>> clip_boxes([[0, 0, 5, 5], [5, 5, 7, 7]], 0, 0, 10, 10)
    [[0, 0, 5, 5], [5, 5, 7, 7]]
    >>> clip_boxes([[0, 0, 5, 5], [5, 5, 7, 7]], 0, 0, 4, 4)
    [[0, 0, 5, 5]]
    >>> clip_boxes([[0, 0, 5, 5], [5, 5, 7, 7]], 10, 10, 14, 14)
    []

    """
    clipped_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        if xmax < x_start or xmin > x_end or ymax < y_start or ymin > y_end:
            continue
        clipped_boxes.append(box)
    return clipped_boxes


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
        boxes = [
            [m["x"] - fr, m["y"] - fr, m["x"] + fr, m["y"] + fr] for m in moles
        ]

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


def list_train_valid_images():
    parts_path = pathlib.Path(
        "~/angelos_mel/angelos_mel/rotomaps/parts"
    ).expanduser()
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
    sessions = [s for s in sorted(session_images.keys()) if s > "2020_"]
    train_sessions = sessions[:-1]
    valid_sessions = sessions[-1:]
    train_images = [
        img for sess in train_sessions for img in session_images[sess]
    ]
    valid_images = [
        img for sess in valid_sessions for img in session_images[sess]
    ]
    return train_images, valid_images, train_sessions, valid_sessions


def drop_paths_without_moles(path_list):
    return [
        path for path in path_list if mel.rotomap.moles.load_image_moles(path)
    ]


# See https://github.com/pytorch/vision/blob/59ec1dfd550652a493cb99d5704dcddae832a204/references/detection/utils.py#L203
def collate_fn(batch):
    return tuple(zip(*batch))
