Getting Started with mel in Jupyter Notebooks
=============================================

[Jupyter Notebooks](http://jupyter.org/) are a great way to interact with data,
it's often very quick to hack something together there when functionality is
lacking in mel's CLI.  When developing new features, it usually makes sense to
iterate on them in Jupyter before making them more permanent.

Setting Up
----------

Assuming you've installed Python 3, OpenCV, Numpy, matplotlib, and Jupyter,
here's a snippet to start off a notebook to hack on Mel:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Make it possible to view images within the notebook.
%matplotlib inline

# If we're working on mel, we're probably going to want to reload changed
# modules.
%load_ext autoreload
%autoreload 1
```

Next, load an image from a rotomap and display it inline:

```python
%aimport mel.rotomap.moles

d = mel.rotomap.moles.RotomapDirectory('data/LeftArm/Upper/2017_04_19/')
f = next(d.yield_frames())
i = f.load_image()

# Make sure the image is nice and large in the notebook.
plt.figure(figsize=(20, 20))

# OpenCV images are BGR, whereas matplotlib expects RGB.
plt.imshow(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
```
