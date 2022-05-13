"""Automatically remove marked regions that are probably not moles."""

import collections
import contextlib
import json
import os
import pathlib
import pickle

import cv2
import numpy
from tqdm import tqdm

import mel.lib.fs
import mel.rotomap.automark
import mel.rotomap.mask
import mel.rotomap.moles

_HALF_IMAGE_SIZE = 16

DEFAULT_BATCH_SIZE = 128

_PRETRAINED_SUFFIX = ".jpg.efficientnet_b0.pt"


@contextlib.contextmanager
def record_input_context(module_to_record):
    activations = []

    def record_response(module, input_):
        nonlocal activations
        activations.append(input_)

    hook = module_to_record.register_forward_pre_hook(record_response)
    with contextlib.ExitStack() as stack:
        stack.callback(hook.remove)
        yield activations


def get_model_weights_version():
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torchvision

    model_url = torchvision.models.efficientnet.model_urls["efficientnet_b0"]
    weights_version = model_url.split("/")[-1]
    return weights_version


def make_model_and_transform():
    # After experimentation, it seems that we can get away with using this
    # model instead of deeper models. This doesn't seem to make much difference
    # to the quality of the results. It does improve the running time somewhat.

    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torchvision

    model = torchvision.models.efficientnet_b0(pretrained=True)
    model.eval()
    num_features = 1280
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )
    return model, num_features, transform


def images_to_features(images, batch_size):
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torch.utils.data

    if not images:
        # pylint: disable=not-callable
        return torch.tensor([])
        # pylint: enable=not-callable

    model, num_features, transform = make_model_and_transform()
    batcher = torch.utils.data.DataLoader(
        [transform(i) for i in images], batch_size=batch_size
    )

    with record_input_context(model.classifier[1]) as fc_in:
        with torch.no_grad():
            for batch in batcher:
                model(batch)

    features = torch.cat([batch[0].flatten(1) for batch in fc_in])
    assert features.shape == (len(images), num_features), features.shape
    return features


def pretrain_image(image_path, moles, batch_size):
    image_path = pathlib.Path(image_path)
    image, mask = open_image_for_classifier(image_path)

    loaded_moles = moles
    guessed_moles = mel.rotomap.detectmoles.moles(image, mask)

    moles_and_marks = mel.rotomap.automark.merge_in_radiuses(
        loaded_moles,
        radii_sources=guessed_moles,
        error_distance=10,
        only_merge=False,
    )

    moles = select_moles(moles_and_marks)
    marks = select_marks(moles_and_marks)

    metadata = moles + marks
    images = [
        get_item_image(image, mask, item, _HALF_IMAGE_SIZE)
        for item in metadata
    ]
    is_mole = [True] * len(moles) + [False] * len(marks)
    is_mole = [item for i, item in enumerate(is_mole) if images[i] is not None]
    metadata = [
        item for i, item in enumerate(metadata) if images[i] is not None
    ]
    images = [item for item in images if item is not None]
    assert len(images) == len(is_mole)
    assert len(images) == len(metadata)
    features = images_to_features(images, batch_size)

    weights_version = get_model_weights_version()

    pretrained_path = image_path.with_suffix(_PRETRAINED_SUFFIX)
    pretrained_path.write_bytes(
        pickle.dumps(
            {
                "features": features,
                "is_mole": is_mole,
                "metadata": metadata,
                "path": image_path,
                "weights_version": weights_version,
            }
        )
    )


def get_item_image(image, mask, item, size):
    x = item["x"]
    y = item["y"]

    if not mask[y, x]:
        return None

    item_image = image[y - size : y + size, x - size : x + size]

    if item_image.shape != (size * 2, size * 2, 3):
        return None

    return item_image


def train(
    epochs,
    train_dataloader,
    valid_dataloader,
    model,
    opt,
    loss_func,
    scheduler,
    evaluators,
):
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torch

    with tqdm(
        total=len(train_dataloader) * epochs + len(valid_dataloader),
        smoothing=0,
    ) as pbar:
        model.train()
        for _ in range(epochs):
            for data in train_dataloader:
                opt.zero_grad()
                out = model(data["features"])
                loss = loss_func(out, data["is_mole"])

                loss.backward()
                opt.step()
                if scheduler is not None:
                    scheduler.step()

                pbar.update(1)

        model.eval()
        with torch.no_grad():
            for data in valid_dataloader:
                out = model(data["features"])
                loss = loss_func(out, data["is_mole"])
                for evaluator in evaluators:
                    evaluator.update(out, data)
                pbar.update(1)


class Evaluator:
    def __init__(self, threshold):
        # Import this as lazily as possible as it takes a while to import, so
        # that we only pay the import cost when we use it.
        import torch

        self.threshold = threshold
        self.num_moles = 0
        self.num_predicted_moles = 0
        self.num_moles_correct = 0

        self.softmax = torch.nn.Softmax(dim=1)

    def update(self, out, data):
        predictions = self.softmax(out)[:, 1] > self.threshold
        self.num_predicted_moles += predictions.sum()
        self.num_moles_correct += ((data["is_mole"] > 0) & predictions).sum()
        self.num_moles += (data["is_mole"] > 0).sum()

    def precision(self):
        if not self.num_predicted_moles:
            raise ValueError("No predicted moles.")
        return (
            100
            * self.num_moles_correct.item()
            / self.num_predicted_moles.item()
        )

    def recall(self):
        if not self.num_moles:
            raise ValueError("No moles.")
        return 100 * self.num_moles_correct.item() / self.num_moles.item()


def make_model(num_features):
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torch

    # After experimenting with a 3-layer head, it seems that this simpler
    # option is just as good. Wit the 3-layer head it becomes possible to
    # over-fit, which doesn't seem to be so much the case with this single
    # layer.
    return torch.nn.Linear(num_features, 2)


def prepare_data(pretrained_data, sessions):
    image_dicts = (
        data for session in sessions for data in pretrained_data[session]
    )
    return [
        {
            "features": features,
            "is_mole": int(is_mole),
        }
        for image_data in image_dicts
        for features, is_mole in zip(
            image_data["features"], image_data["is_mole"]
        )
    ]


def split_data(pretrained_data, training_split=0.8):
    sessions = list(pretrained_data.keys())
    num_sessions = len(sessions)
    if training_split != 1 and num_sessions < 2:
        raise ValueError("Must have at least two sessions in order to split")
    num_training_sessions = int(num_sessions * training_split)
    num_validation_sessions = num_sessions - num_training_sessions
    if training_split != 1:
        assert num_validation_sessions
    training_data = prepare_data(
        pretrained_data, sessions[:num_training_sessions]
    )
    validation_data = prepare_data(
        pretrained_data, sessions[num_training_sessions:]
    )
    return training_data, validation_data


def load_pretrained(pretrained_paths):
    work_items = [
        (session, path)
        for session, path_list in pretrained_paths.items()
        for path in path_list
    ]
    pretrained_data = collections.defaultdict(list)
    current_weights_version = get_model_weights_version()
    for session, path in work_items:
        loaded_data = pickle.loads(path.read_bytes())
        pretrained_weights_version = loaded_data["weights_version"]
        if pretrained_weights_version != current_weights_version:
            raise Exception(
                (
                    "Pretrained weights version mismatch."
                    "Please pretrain again.\n"
                    f"Pretrained: {pretrained_weights_version}\n"
                    f"Current: {current_weights_version}"
                )
            )
        pretrained_data[session].append(loaded_data)
    return pretrained_data


def find_pretrained(melroot):
    parts_path = melroot / "rotomaps" / "parts"
    all_sessions = collections.defaultdict(list)
    for part in parts_path.iterdir():
        for subpart in part.iterdir():
            subpart_paths = sorted(p for p in subpart.iterdir())
            for session in subpart_paths:
                for pretrained_file in session.glob("*" + _PRETRAINED_SUFFIX):
                    all_sessions[f"{session.stem}"].append(pretrained_file)
    return all_sessions


def make_model_and_fit(
    training_data,
    validation_data,
    evaluators,
    batch_size,
    num_epochs,
    learning_rate,
):
    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torch

    training_batcher = torch.utils.data.DataLoader(
        training_data, batch_size=batch_size, shuffle=True
    )

    for batch in training_batcher:
        pass

    validation_batcher = torch.utils.data.DataLoader(
        validation_data, batch_size=batch_size
    )

    assert len(training_data[0]["features"].shape) == 1
    num_features = training_data[0]["features"].shape[0]
    model = make_model(num_features)
    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(params=model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(training_data),
    )

    train(
        num_epochs,
        training_batcher,
        validation_batcher,
        model,
        opt,
        loss_func,
        scheduler,
        evaluators,
    )

    return model


def open_image_for_classifier(image_path):
    if not os.path.exists(image_path):
        raise OSError("No such file or directory: {}".format(image_path))
    elif os.path.isdir(image_path):
        raise OSError("Is a directory: {}".format(image_path))

    flags = cv2.IMREAD_UNCHANGED + cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR
    try:
        original_image = cv2.imread(str(image_path), flags)
        if original_image is None:
            raise OSError(f"File not recognized by opencv: {image_path}")
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise OSError(f"Error handling image at: {image_path}") from e

    mask = mel.rotomap.mask.load(image_path)
    green = numpy.zeros(original_image.shape, numpy.uint8)
    green[:, :, 1] = 255
    image = cv2.bitwise_and(original_image, original_image, mask=mask)
    not_mask = cv2.bitwise_not(mask)
    green = cv2.bitwise_and(green, green, mask=not_mask)
    image = cv2.bitwise_or(image, green)
    return image, mask


def select_moles(moles_and_marks):
    moles = []

    for item in moles_and_marks:
        kind = item.get("kind", None)
        looks_like = item.get("looks_like", None)
        if looks_like == "unsure":
            continue
        elif kind == "mole" and looks_like == "non-mole":
            continue
        elif kind == "non-mole" and looks_like == "mole":
            pass
        elif not item[mel.rotomap.moles.KEY_IS_CONFIRMED]:
            continue
        moles.append(item)

    return moles


def select_marks(moles_and_marks):
    marks = []

    for item in moles_and_marks:
        kind = item.get("kind", None)
        looks_like = item.get("looks_like", None)
        if looks_like == "unsure":
            continue
        elif kind == "non-mole" and looks_like == "mole":
            continue
        elif kind == "mole":
            # Even if this looks_like a non-mole, exclude it from the dataset.
            # Even humans may find those cases ambiguous, perhaps it's better
            # to stick to unambiguous cases for training and evaluating.
            # There seem to be plenty of real marks to consider instead.
            continue
        elif item[mel.rotomap.moles.KEY_IS_CONFIRMED]:
            continue
        marks.append(item)

    return marks


def filter_marks(is_mole, image, moles, include_canonical):
    """Return a list of moles with the unlikely ones filtered out."""

    filtered_moles = []
    for m in moles:
        r = _HALF_IMAGE_SIZE

        x = m["x"]
        y = m["y"]

        # Don't touch canonical moles.
        if not include_canonical and m[mel.rotomap.moles.KEY_IS_CONFIRMED]:
            filtered_moles.append(m)
            continue

        image_fragment = image[y - r : y + r, x - r : x + r]

        # TODO: decide what to do about fragments that overlap the image
        # boundary.
        if not all(image_fragment.shape):
            filtered_moles.append(m)
            continue

        if is_mole(image_fragment):
            filtered_moles.append(m)

    return filtered_moles


def make_is_mole_func(metadata_dir, model_fname, softmax_threshold):

    # These imports can be very expensive, so we delay them as late as
    # possible.
    #
    # Also, they introduce a significant amount of extra dependencies. At
    # this point it is only experimental functionality, so importing late
    # like this allows folks not using it to avoid the extra burden.

    # Import this as lazily as possible as it takes a while to import, so that
    # we only pay the import cost when we use it.
    import torch

    model, num_features, transform = make_model_and_transform()
    head = make_model(num_features)
    head.load_state_dict(torch.load(metadata_dir / model_fname))

    metadata_path = (metadata_dir / model_fname).with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())

    trained_version = metadata["model_weights_version"]
    current_version = get_model_weights_version()

    if trained_version != current_version:
        raise Exception(
            (
                "Pretrained weights version mismatch."
                "Please pretrain again.\n"
                f"Pretrained: {trained_version}\n"
                f"Current: {current_version}"
            )
        )

    model.classifier[1] = head

    softmax = torch.nn.Softmax(dim=1)

    def is_mole(image):

        batcher = torch.utils.data.DataLoader([transform(image)], batch_size=1)

        with torch.no_grad():
            for i, batch in enumerate(batcher):
                assert i == 0
                class_scores = softmax(model(batch))

        assert class_scores.shape == (1, 2)

        is_mole_detected = class_scores[0][1] > softmax_threshold

        return is_mole_detected

    return is_mole


# -----------------------------------------------------------------------------
# Copyright (C) 2022 Angelos Evripiotis.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------ END-OF-FILE ----------------------------------
