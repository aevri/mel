# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
# -

# Make it possible to view images within the notebook.
# %matplotlib inline

# If we're working on mel, we're probably going to want to reload changed
# modules.
# %load_ext autoreload
# %autoreload 2

from mel.rotomap import moles
from mel.rotomap import detectmolesnn

# # Setup data

data_path = pathlib.Path("/Volumes/angelos-mel2/angelos_mel/rotomaps/parts")
assert data_path.exists()

path = data_path / 'LeftArm/Upper/2022_07_29/01.jpg'

num_train_moles = 10

half_size = 64
x_data = [
    detectmolesnn.pick_one_mole(path, border_size=64, index=index)
    for index in range(num_train_moles)
]
plt.imshow(
    detectmolesnn.rgb_tensor_to_cv2_image(
        x_data[0]
    )
)

scaleup = 16
image_width = x_data[0].shape[2] // scaleup
image_height = x_data[0].shape[1] // scaleup

y_data = detectmolesnn.pointvec_to_vexy_y_tensor(
    np.array([[half_size, half_size]]), image_width, image_height, scaleup,
)
plt.imshow(
    detectmolesnn.rgb_tensor_to_cv2_image(
        y_data
    )
)

# # Training

datamodule = detectmolesnn.CenteredImageMoles(path, batch_size=2)

experiment_name = "VexyConvVal"
model = detectmolesnn.VexyConv(total_steps=10000)
wandb_logger = pl.loggers.WandbLogger(
    project="mel-automark5", name=experiment_name
)
wandb_logger.watch(model, log="all")
trainer_kwargs = {
    "max_steps": model.total_steps,
    "log_every_n_steps": 1,
    "callbacks": [detectmolesnn.GlobalProgressBar()],
    "enable_checkpointing": False,
    "accelerator": "auto",
    #"accumulate_grad_batches": args.accumulate_grad_batches,
    #"val_check_interval": 5,
    # "auto_lr_find": True,
}
trainer_kwargs["logger"] = wandb_logger
trainer = pl.Trainer(**trainer_kwargs)

trainer.fit(model, datamodule)
wandb.finish()

# +
# MSE training loss on 10 moles:
# VexyConv5 0.00005
# VexyConv4 0.0001
# VexyConv3 0.0005
# VexyConv2 0.36
# VexyConv  0.47
# -

model.eval()
yy_data = model(x_data[0].unsqueeze(0))
plt.imshow(
    detectmolesnn.rgb_tensor_to_cv2_image(
        yy_data[0]
    )
)
torch.nn.functional.mse_loss(yy_data[0].detach(), y_data)

detectmolesnn.vexy_y_tensor_to_position_counter(y_data, scaleup)

detectmolesnn.vexy_y_tensor_to_position_counter(yy_data[0], scaleup)

# # Test

half_size = 64
x2_data = detectmolesnn.pick_one_mole(path, border_size=64, index=num_train_moles)
plt.imshow(
    detectmolesnn.rgb_tensor_to_cv2_image(
        x2_data
    )
)

yy2_data = model(x2_data.unsqueeze(0))
plt.imshow(
    detectmolesnn.rgb_tensor_to_cv2_image(
        yy2_data[0]
    )
)

detectmolesnn.vexy_y_tensor_to_position_counter(yy2_data[0], scaleup)
