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

import mel.rotomap.moles
import mel.rotomap.detectmolesnn

# +
data_path = pathlib.Path("/Volumes/angelos-mel2/angelos_mel/rotomaps/parts")
assert data_path.exists()

def show_image_from_rotodir(path):
    rotodir = mel.rotomap.moles.RotomapDirectory(path)
    frame = next(rotodir.yield_frames())
    image = frame.load_image()
    mask = frame.load_mask()
    # Make sure the image is nice and large in the notebook.
    plt.figure(figsize=(20, 20))

    # OpenCV images are BGR, whereas matplotlib expects RGB.
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
show_image_from_rotodir(data_path / 'LeftArm/Upper/2022_07_29/')
# -

path = data_path / 'LeftArm/Upper/2022_07_29/01.jpg'
(
    image_width,
    image_height,
    scaleup,
) = mel.rotomap.detectmolesnn.calc_vexy_shapewh_scaleup(path)
y_data = mel.rotomap.detectmolesnn.rotoimage_to_vexy_y_tensor(
    path, image_width, image_height, scaleup,
)

# +
# NEXT: try to get the multiplier to be a whole number.
# -

image = y_data.detach().numpy() * 255
image = np.uint8(image)
image = image.transpose((1, 2, 0))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# +
# Convert back to list of moles and positions.
# -

pos_counter = mel.rotomap.detectmolesnn.vexy_y_tensor_to_position_counter(y_data, scaleup)

pos_list = mel.rotomap.detectmolesnn.position_counter_to_position_list(pos_counter, threshold=10)

moles = mel.rotomap.moles.load_image_moles(path)

mel.rotomap.detectmolesnn.compare_position_list_to_moles(moles, pos_list, 0)
