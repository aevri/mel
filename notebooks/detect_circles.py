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
import torch
# -

# Make it possible to view images within the notebook.
# %matplotlib inline

# If we're working on mel, we're probably going to want to reload changed
# modules.
# %load_ext autoreload
# %autoreload 2

import mel.lib.common


# +
def gen_image():
    image_width, image_height = (1024, 800)
    image = mel.lib.common.new_image(image_height, image_width)
    points = [
        [random.randrange(image_width), random.randrange(image_height), random.randrange(8, 12)]
        for _ in range(random.randrange(1, 20))
    ]
    boxes = [
        [x - r, y - r, x + r, y + r] for x, y, r in points
    ]
    fr = 10
    boxes = [
        [x - fr, y - fr, x + fr, y + fr] for x, y, r in points
    ]
    # From https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    # ...
    # boxes.append([xmin, ymin, xmax, ymax])
    # ...
    # # convert everything into a torch.Tensor
    # boxes = torch.as_tensor(boxes, dtype=torch.float32)
    # # there is only one class
    # labels = torch.ones((num_objs,), dtype=torch.int64)
    # ...
    # target["boxes"] = boxes
    # target["labels"] = labels
    # ...
    for x, y, radius in points:
       mel.lib.common.draw_circle(image, x, y, radius, (0, 0, 255))
    target = {}
    target["labels"] = torch.ones((len(points),), dtype=torch.int64)
    target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
    return image, target

def torch_gen_image():
    image, target = gen_image()
    to_tensor = torchvision.transforms.ToTensor()
    return to_tensor(image), target

image, target = gen_image()
plt.imshow(image)
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
model(to_tensor(image).unsqueeze(0))

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
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        set_parameter_no_grad(model)
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
        result = self.model(x)
        nobj_offset = 0
        for y_item, result_item in zip(y, result):
            nobj_offset += len(result_item["labels"]) - len(y_item["labels"])
        self.log("valid_nobj_offset", float(nobj_offset), prog_bar=True)
        return result
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
model = PlModule()


# +
# See https://github.com/pytorch/vision/blob/59ec1dfd550652a493cb99d5704dcddae832a204/references/detection/utils.py#L203
def collate_fn(batch):
    return tuple(zip(*batch))

# Note, need to normalize the images,
# e.g. https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#load-data
train_dataset = [torch_gen_image() for _ in range(1000)]
valid_dataset = [torch_gen_image() for _ in range(10)]
# -

import gc
torch.cuda.empty_cache()
gc.collect()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, collate_fn=collate_fn, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=10, collate_fn=collate_fn)

# + active=""
# for x, y in train_loader:
#     print(y)
#     break
# -

trainer = pl.Trainer(max_epochs=1, accelerator="auto")
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

import gc
torch.cuda.empty_cache()
gc.collect()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, collate_fn=collate_fn)
set_parameter_yes_grad(model)
trainer = pl.Trainer(max_epochs=1, accelerator="auto")
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

# +
image, target = gen_image()
to_tensor = torchvision.transforms.ToTensor()
torch_image = to_tensor(image)
model.eval()
with torch.no_grad():
    boxes = model(torch_image.unsqueeze(0))[0]["boxes"]

def draw_result(image, boxes):
    image = image.copy()
    print(boxes)
    for xmin, ymin, xmax, ymax in boxes:
        mel.lib.common.draw_circle(image, int(xmin), int(ymin), 2, (255, 255, 255))
        mel.lib.common.draw_circle(image, int(xmax), int(ymax), 2, (255, 255, 255))
    return image

plt.imshow(draw_result(image, boxes))
