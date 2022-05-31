# +
import pathlib

import cv2
import matplotlib.pyplot as plt
# -

# Make it possible to view images within the notebook.
# %matplotlib inline

# If we're working on mel, we're probably going to want to reload changed
# modules.
# %load_ext autoreload
# %autoreload 2

import mel.rotomap.moles

data_path = pathlib.Path("/Volumes/angelos-mel/angelos_mel/rotomaps/parts")
assert data_path.exists()
d = mel.rotomap.moles.RotomapDirectory(data_path / 'LeftArm/Upper/2017_04_19/')
f = next(d.yield_frames())
i = f.load_image()

plt.figure(figsize=(20, 20))

# +
# Make sure the image is nice and large in the notebook.
plt.figure(figsize=(20, 20))

# OpenCV images are BGR, whereas matplotlib expects RGB.
plt.imshow(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
# -

import mel.lib.common

i.shape

import mel.rotomap.detectmolesnn
image_height, image_width = i.shape[0:2]
image = mel.rotomap.detectmolesnn.locations_image(f.moles, image_width, image_height)
plt.imshow(cv2.cvtColor(image // 2 + i // 2, cv2.COLOR_BGR2RGB))

# + active=""
# plt.imshow(cv2.cvtColor(i, cv2.COLOR_BGR2HSV))

# +
# Model, data, loss fn, optimizer.
import pytorch_lightning as pl
import torchvision
import torch

model = mel.rotomap.detectmolesnn.CackModel()
to_tensor = torchvision.transforms.ToTensor()
# -

train_dl = torch.utils.data.DataLoader([(to_tensor(i), to_tensor(image))], batch_size=1)
trainer = pl.Trainer(
    max_epochs=30,
)

trainer.fit(model, train_dl)

result = model(to_tensor(i).unsqueeze(0))
print(result.shape)
plt.imshow(result.detach().numpy()[0][0])

plt.imshow(image[:, :, 2])
