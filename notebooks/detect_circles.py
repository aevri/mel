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
# -

# Make it possible to view images within the notebook.
# %matplotlib inline

# If we're working on mel, we're probably going to want to reload changed
# modules.
# %load_ext autoreload
# %autoreload 2

import mel.lib.common

# +
image_width, image_height = (1024, 800)
image = mel.lib.common.new_image(image_height, image_width)

points = [
    [100, 150],
    [150, 50],
]

points = [
    [random.randrange(image_width), random.randrange(image_height), random.randrange(8, 12)]
    for _ in range(random.randrange(1, 20))
]

for x, y, radius in points:
   mel.lib.common.draw_circle(image, x, y, radius, (0, 0, 255))

plt.imshow(image)

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
