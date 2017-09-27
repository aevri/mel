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

```python
%aimport mel.rotomap.moles
%aimport mel.rotomap.mask
%aimport mel.lib.moleimaging

import itertools
import pathlib

bodypart = pathlib.Path('data/LeftArm/Upper')
rotomaps = [mel.rotomap.moles.RotomapDirectory(x) for x in bodypart.iterdir() if x.is_dir()]

known_frames = list(itertools.chain.from_iterable(x.yield_frames() for x in rotomaps[:-1]))
unk_frames = list(rotomaps[-1].yield_frames())

len(known_frames), len(unk_frames)
```

```python
%aimport mel.jupyter.utils
mj = mel.jupyter.utils

import pandas

import collections

uuid_target = '4bd26787cf1a4995bc58ec3d19a3192a'
uuid_target = '274224c9138c4d82a477402cb9c71490'
uuid_target = 'cc67302a0dcf4173918ac6f171012cf9'

uuid_to_offsets = collections.defaultdict(list)
for frame in itertools.chain.from_iterable(x.yield_frames() for x in rotomaps[-4:-1]):
    mask = frame.load_mask()
    contour = mel.jupyter.utils.biggest_contour(mask)
    ellipse = cv2.fitEllipse(contour)
    elspace = mj.EllipseSpace(ellipse)
    uuid_pos = {
        uuid_: np.array(elspace.to_space(pos))
        for uuid_, pos in frame.moledata.uuid_points.items()
    }

    if uuid_target not in uuid_pos:
        continue

    center = uuid_pos[uuid_target]
    uuid_pos = {
        uuid_: pos - center
        for uuid_, pos in uuid_pos.items()
        if uuid_ != uuid_target
    }

    for uuid_, pos in uuid_pos.items():
        uuid_to_offsets[uuid_].append(pos)

next(iter(uuid_to_offsets.items()))[1][0].shape
```

```python
%%time
%aimport mel.jupyter.utils
mj = mel.jupyter.utils

uuid_to_frameposlist = mj.frames_to_uuid_frameposlist(
    itertools.chain.from_iterable(
        x.yield_frames() for x in rotomaps[-4:-2]))

c = mj.MoleClassifier(uuid_to_frameposlist)
```

```python
dframes = [
    pandas.DataFrame({uuid_[:3]: [x for x, y in offsetlist]})
    for uuid_, offsetlist in uuid_to_offsets.items()
    if len(offsetlist) > 5
]

df = pandas.concat(dframes, axis=1)
df.plot.density(figsize=(12, 12))
```
