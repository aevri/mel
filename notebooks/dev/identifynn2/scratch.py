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

model = identifynn2.RandomChooser(partnames_uuids)

model = identifynn2.SelfposOnly(partnames_uuids)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train_loss = []
train_acc = []
valid_loss = []
valid_acc = []
valid_step = []


# +
def do_valid():
    with torch.no_grad():
        loss, acc = identifynn2.eval_step(model, criterion, optimizer, valid)
    valid_loss.append(float(loss))
    valid_acc.append(float(acc))
    valid_step.append(len(train_loss))

def do_train(num_iter):
    for _ in tqdm(range(num_iter), leave=False):
        loss, acc = identifynn2.train_step(model, criterion, optimizer, train)
        train_loss.append(float(loss))
        train_acc.append(float(acc))


# -

do_valid()

for _ in tqdm(range(10)):
    do_train(10)
    do_valid()

train_df = pd.DataFrame({"train loss": train_loss, "train accuracy": train_acc})
valid_df = pd.DataFrame({"valid loss": valid_loss, "valid accuracy": valid_acc}, index=valid_step)
ax = train_df.plot(y="train loss")
ax = valid_df.plot(y="valid loss", ax=ax)
ax = train_df.plot(y="train accuracy", secondary_y=True, ax=ax)
ax = valid_df.plot(y="valid accuracy", secondary_y=True, ax=ax)

# + active=""
# identifynn2.infer_uuids(model, x1)
