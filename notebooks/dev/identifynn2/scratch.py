# +
import pathlib
import subprocess

import cv2
import numpy as np
import torch
import pandas as pd
from tqdm.auto import tqdm

import mel.rotomap.moles
import mel.lib.common
import mel.rotomap.dataset as mel_dataset
import mel.rotomap.identifynn2 as identifynn2

# If we're working on mel, we're probably going to want to reload changed
# modules.
# %load_ext autoreload
# %autoreload 2
# -

# Generate fake data if not already present.
temp_path = pathlib.Path("temp-data")
if not temp_path.exists():
    subprocess.run(f"mel-debug gen-repo {temp_path} --num-rotomaps 10 --num-parts 10", shell=True)
    subprocess.run(f"find {temp_path} -iname '*.jpg' | xargs mel rotomap automask", shell=True)
    subprocess.run(f"find {temp_path} -iname '*.jpg' | xargs mel rotomap calc-space", shell=True)

# +
pathdict = mel_dataset.make_pathdict(temp_path)
pathdict = mel_dataset.drop_empty_paths(pathdict)
train, valid = mel_dataset.split_train_valid_last(pathdict)
partnames_uuids = mel_dataset.make_partnames_uuids(pathdict)

def process_dataset(pathdict, name):
    d = mel_dataset.listify_pathdict(pathdict)
    d = mel_dataset.yield_imagemoles_from_pathlist(d)
    d = list(d)
    print(f"There are {len(d)} {name} items.")
    return d

train = process_dataset(train, "training")
valid = process_dataset(valid, "validation")
# -

criterion = torch.nn.CrossEntropyLoss()

#model = identifynn2.RandomChooser(partnames_uuids)
#model = identifynn2.SelfposOnly(partnames_uuids)
model = identifynn2.SelfposOnlyVec(partnames_uuids)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
trainer = identifynn2.Trainer(model, criterion, optimizer, train, valid)
trainer.validate()

for _ in tqdm(range(100)):
    trainer.train(10)
    trainer.validate()

trainer.plot()

# + active=""
# identifynn2.infer_uuids(model, x1)
