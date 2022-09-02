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

half_size = 64
x_data = detectmolesnn.pick_one_mole(path, border_size=64, index=0)
plt.imshow(
    detectmolesnn.rgb_tensor_to_cv2_image(
        x_data
    )
)

scaleup = 16
image_width = x_data.shape[2] // scaleup
image_height = x_data.shape[1] // scaleup

y_data = detectmolesnn.pointvec_to_vexy_y_tensor(
    np.array([[half_size, half_size]]), image_width, image_height, scaleup,
)
plt.imshow(
    detectmolesnn.rgb_tensor_to_cv2_image(
        y_data
    )
)

# # Training

train_dl = torch.utils.data.DataLoader(
    [{"x_data": x_data, "y_data": y_data}],
    batch_size=1,
    shuffle=True,
)
model = detectmolesnn.VexyConv(total_steps=10000)
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
trainer = pl.Trainer(**trainer_kwargs)

trainer.fit(model, train_dl)

y2_data = model(x_data.unsqueeze(0))
plt.imshow(
    detectmolesnn.rgb_tensor_to_cv2_image(
        y2_data[0]
    )
)

detectmolesnn.vexy_y_tensor_to_position_counter(y_data, scaleup)

detectmolesnn.vexy_y_tensor_to_position_counter(y2_data[0], scaleup)
