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
# -

# Make it possible to view images within the notebook.
# %matplotlib inline

# If we're working on mel, we're probably going to want to reload changed
# modules.
# %load_ext autoreload
# %autoreload 2

from mel.rotomap import moles
from mel.rotomap import detectmolesnn

data_path = pathlib.Path("/Volumes/angelos-mel2/angelos_mel/rotomaps/parts")
assert data_path.exists()

path = data_path / 'LeftArm/Upper/2022_07_29/01.jpg'

half_size = 64
x_data = detectmolesnn.pick_one_mole(path, border_size=64, index=1)
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

model = detectmolesnn.VexyConv(total_steps=100)

pos_counter = mel.rotomap.detectmolesnn.vexy_y_tensor_to_position_counter(y_data, scaleup)

pos_list = mel.rotomap.detectmolesnn.position_counter_to_position_list(pos_counter, threshold=10)

moles = mel.rotomap.moles.load_image_moles(path)

mel.rotomap.detectmolesnn.compare_position_list_to_moles(moles, pos_list, 0)
