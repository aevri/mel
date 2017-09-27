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

def show_image(image):
    plt.figure(figsize=(20, 20))
    plt.imshow(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

%%time
%aimport mel.rotomap.moles
%aimport mel.rotomap.mask
%aimport mel.lib.moleimaging

import itertools
import pathlib

bodypart = pathlib.Path('data/LeftArm/Upper')
rotomaps = [mel.rotomap.moles.RotomapDirectory(x) for x in bodypart.iterdir() if x.is_dir()]

known_frames = list(itertools.chain.from_iterable(x.yield_frames() for x in rotomaps[:-2]))
unk_frames = list(rotomaps[-2].yield_frames())

len(known_frames), len(unk_frames)
```

```python
%%time
%aimport mel.jupyter.utils
mj = mel.jupyter.utils

uuid_to_frameposlist = mj.frames_to_uuid_frameposlist(
    itertools.chain.from_iterable(
        x.yield_frames() for x in rotomaps[-4:-2]))

unk_uuid_to_frameposlist = mj.frames_to_uuid_frameposlist(rotomaps[-2].yield_frames())
```

```python
%%time
%aimport mel.jupyter.utils
%aimport mel.cmd.rotomapmontagesingle
mj = mel.jupyter.utils

import collections
import pprint

classifier = mj.MoleClassifier(uuid_to_frameposlist)

for frame in unk_frames[:-1]:
    uuid_to_pos = mj.frame_to_uuid_to_pos(frame)
    guesser = mj.Guesser(uuid_to_pos, classifier)

    best = mj.best_match_combination(guesser)
    #pprint.pprint(best)
    print('matched', sum(1 for a, b in best[1].items() if a == b), 'of', len(best[1]))
```

```python
%%time
%aimport mel.jupyter.utils
%aimport mel.cmd.rotomapmontagesingle
mj = mel.jupyter.utils

import collections
import pprint

classifier = mj.MoleClassifier(uuid_to_frameposlist)

match_index_counts = collections.Counter()

for frame in unk_frames[:-1]:
    uuid_to_pos = mj.frame_to_uuid_to_pos(frame)
    for uuid_, pos in uuid_to_pos.items():
        for uuid2, pos2 in uuid_to_pos.items():
            if uuid_ == uuid2:
                continue
            results = classifier.guesses_from_known_mole(uuid_, pos, pos2)
            matched = False
            for i, r in enumerate(reversed(sorted(results, key=lambda x: x[2]))):
                uuid3, p, q = r
                if uuid2 == uuid3:
                    #print('ref', uuid_, 'match', uuid3, i)
                    matched = True
                    quartile = int(4 * i / len(results))
                    match_index_counts[quartile] += 1
            if not matched:
                print('No match for', uuid_, uuid2)

            #pprint.pprint(results)

total_match_index_counts = sum(match_index_counts.values())
for index, count in sorted(match_index_counts.items()):
    print(index, count / total_match_index_counts)
```
