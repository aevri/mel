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


# +
def show_target_image_from_path(path):
    (
        image_width,
        image_height,
        scale_x,
        scale_y,
    ) = mel.rotomap.detectmolesnn.calc_mxy_shapewh_scalexy(path)
    y = mel.rotomap.detectmolesnn.rotoimage_to_vexy_y_tensor(
        path, image_width, image_height, scale_x, scale_y
    )
    
    image = y.detach().numpy() * 255
    image = np.uint8(image)
    image = image.transpose((1, 2, 0))
    #print(image[image[:,:,0] != 0])
    
    # OpenCV images are BGR, whereas matplotlib expects RGB.
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    
show_target_image_from_path(data_path / 'LeftArm/Upper/2022_07_29/01.jpg')
# -

path = data_path / 'LeftArm/Upper/2022_07_29/01.jpg'
(
    image_width,
    image_height,
    scale_x,
    scale_y,
) = mel.rotomap.detectmolesnn.calc_mxy_shapewh_scalexy(path)
y_data = mel.rotomap.detectmolesnn.rotoimage_to_vexy_y_tensor(
    path, image_width, image_height, scale_x, scale_y
)

image = y_data.detach().numpy() * 255
image = np.uint8(image)
image = image.transpose((1, 2, 0))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

pos_image = mel.rotomap.detectmolesnn.vexy_y_tensor_to_position_image(y_data, 16)
plt.figure(figsize=(20, 20))
plt.imshow(pos_image.numpy())

# +
# Convert back to list of moles and positions.
# -

pos_list = mel.rotomap.detectmolesnn.position_image_to_position_list(pos_image, 1)
sorted(pos_list)

moles = mel.rotomap.moles.load_image_moles(path)

sorted(mel.rotomap.moles.mole_list_to_pointvec(moles).tolist())

mel.rotomap.detectmolesnn.compare_position_list_to_moles(moles, pos_list, 16)
