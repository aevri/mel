# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pathlib
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
# -

import wandb

# Make it possible to view images within the notebook.
# %matplotlib inline

# If we're working on mel, we're probably going to want to reload changed
# modules.
# %load_ext autoreload
# %autoreload 2

import mel.lib.common
import mel.lib.fs
import mel.rotomap.moles
import mel.rotomap.detectmolesnn2

parts_path = pathlib.Path("~/angelos_mel/angelos_mel/rotomaps/parts").expanduser()


# +
def list_train_valid_images():
    parts_path = pathlib.Path("~/angelos_mel/angelos_mel/rotomaps/parts").expanduser()
    exclude_parts = ["LeftArm/Hand", "RightArm/Hand", "LeftLeg/Foot", "RightLeg/Foot", "Trunk/Lower", "Trunk/Back"]
    session_images = mel.lib.fs.list_rotomap_images_by_session(parts_path, exclude_parts=exclude_parts)
    sessions = [s for s in sorted(session_images.keys()) if s > "2020_"]
    train_sessions = sessions[:-1]
    valid_sessions = sessions[-1:]
    train_images = [img for sess in train_sessions for img in session_images[sess]]
    valid_images = [img for sess in valid_sessions for img in session_images[sess]]
    return train_images, valid_images

train_images, valid_images = list_train_valid_images()
print(f"There are {len(train_images)} training images.")
print(f"There are {len(valid_images)} validation images.")


# +
def drop_paths_without_moles(path_list):
    return [path for path in path_list if mel.rotomap.moles.load_image_moles(path)]

train_images = drop_paths_without_moles(train_images)
valid_images = drop_paths_without_moles(valid_images)

print(f"There are {len(train_images)} training images.")
print(f"There are {len(valid_images)} validation images.")


# +
def load_image(image_path):
    #flags = cv2.IMREAD_UNCHANGED + cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR
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

plt.figure(figsize=(20, 20))
plt.imshow(load_image(train_images[0]))


# -

class MoleImageBoxesDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.image_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        path = self.image_paths[index]
        
        image = load_image(path)
        to_tensor = torchvision.transforms.ToTensor()
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


def rgb_tensor_to_image(tensor):
    image = tensor.detach().numpy() * 255
    image = np.uint8(image)
    image = image.transpose((1, 2, 0))
    return image


train_dataset = MoleImageBoxesDataset(train_images)
valid_dataset = MoleImageBoxesDataset(valid_images)

image, target = train_dataset[0]
plt.imshow(rgb_tensor_to_image(image))

target

# +
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.eval()
# -

to_tensor = torchvision.transforms.ToTensor()
model.eval()
model(image.unsqueeze(0))

# +
# See https://github.com/pytorch/vision/blob/59ec1dfd550652a493cb99d5704dcddae832a204/references/detection/engine.py#L12

# +
import pytorch_lightning as pl

def set_parameter_no_grad(model):
    for param in model.parameters():
        param.requires_grad = False

def set_parameter_yes_grad(model):
    for param in model.parameters():
        param.requires_grad = True
        
class PlModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lr = 0.0001
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        #set_parameter_no_grad(model)
        num_classes = 2  # 1 class + background
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.eval()
        
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss_dict = self.model(x, y)
        losses = sum(loss for loss in loss_dict.values())
        self.log("train_loss", losses.detach())
        return losses
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        #self.model.train()
        result = self.model(x, y)
        precision_list = []
        recall_list = []
        for y_item, result_item in zip(y, result):
            item_precision, item_recall = mel.rotomap.detectmolesnn2.calc_precision_recall(
                target_poslist=mel.rotomap.detectmolesnn2.boxes_to_poslist(y_item["boxes"]),
                poslist=mel.rotomap.detectmolesnn2.boxes_to_poslist(result_item["boxes"]),
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
model = PlModule()


# -

# See https://github.com/pytorch/vision/blob/59ec1dfd550652a493cb99d5704dcddae832a204/references/detection/utils.py#L203
def collate_fn(batch):
    return tuple(zip(*batch))


# + active=""
# import gc
# torch.cuda.empty_cache()
# gc.collect()
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, collate_fn=collate_fn, shuffle=True)
# valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=10, collate_fn=collate_fn, shuffle=True)

# + active=""
# trainer = pl.Trainer(limit_train_batches=10, max_epochs=1, accelerator="auto")
# trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
# -

import gc
torch.cuda.empty_cache()
gc.collect()

import gc
torch.cuda.empty_cache()
gc.collect()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)

# +
experiment_name = "base-lr.0001"
def setup_wandb_logger():
    wandb_logger = pl.loggers.WandbLogger(
        project="mel-faster-rcnn", name=experiment_name
    )
    wandb_logger.watch(model, log="all")
    return wandb_logger

trainer_kwargs = {
    "log_every_n_steps": 1,
    "enable_checkpointing": False,
    "accelerator": "auto",
    #"accumulate_grad_batches": args.accumulate_grad_batches,
    #"max_epochs": 10,
    "max_epochs": 1,
    "limit_train_batches": 10,
    "limit_val_batches": 10,
    #"val_check_interval": 50,
    # "auto_lr_find": True,
    #"logger": setup_wandb_logger(),
}
trainer = pl.Trainer(**trainer_kwargs)

#set_parameter_yes_grad(model)
#trainer = pl.Trainer(max_epochs=1, accelerator="auto", limit_val_batches=10, val_check_interval=50)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

# +
image, target = valid_dataset[60]
model.eval()
with torch.no_grad():
    boxes = model(image.unsqueeze(0))[0]["boxes"]

def draw_result(image, boxes):
    image = rgb_tensor_to_image(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print(boxes)
    for xmin, ymin, xmax, ymax in target["boxes"]:
        x = 0.5 * (xmin + xmax)
        y = 0.5 * (ymin + ymax)
        mel.lib.common.draw_circle(image, int(x), int(y), 20, (128, 0, 128))
    for xmin, ymin, xmax, ymax in boxes:
        x = 0.5 * (xmin + xmax)
        y = 0.5 * (ymin + ymax)
        mel.lib.common.draw_circle(image, int(x), int(y), 10, (256, 256, 0))
    return image

plt.figure(figsize=(20, 20))
plt.imshow(draw_result(image, boxes))
# -

mel.rotomap.detectmolesnn2.boxes_to_poslist(target["boxes"])

mel.rotomap.detectmolesnn2.boxes_to_poslist(boxes)

mel.rotomap.detectmolesnn2.calc_precision_recall(
    target_poslist=mel.rotomap.detectmolesnn2.boxes_to_poslist(target["boxes"]),
    poslist=mel.rotomap.detectmolesnn2.boxes_to_poslist(boxes),
)
