"""Identify which moles are which, using neural nets."""
import collections
import contextlib

import time

import numpy
import torch.utils.data
import torchvision
import tqdm

import mel.lib.ellipsespace
import mel.rotomap.moles
import json
import mel.lib.fs
import mel.rotomap.identifynn


def make_identifier():
    melroot = mel.lib.fs.find_melroot()
    model_dir = melroot / mel.lib.fs.DEFAULT_CLASSIFIER_PATH
    model_path = model_dir / "identify.pth"
    metadata_path = model_dir / "identify.json"
    return mel.rotomap.identifynn.MoleIdentifier(metadata_path, model_path)


class MoleIdentifier:
    def __init__(self, metadata_path, model_path):
        # Some of these imports are expensive, so to keep program start-up time
        # lower, import them only when necessary.
        import torch
        import mel.rotomap.identifynn

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        model_args = self.metadata["model_args"]
        self.part_to_index = self.metadata["part_to_index"]
        self.classes = self.metadata["classes"]
        self.class_to_index = {cls: i for i, cls in enumerate(self.classes)}

        self.in_fields = ["part_index"]
        self.in_fields.extend(
            ["molemap", "molemap_detail_2", "molemap_detail_4"]
        )
        self.out_fields = ["uuid_index", "mole_count"]

        self.model = mel.rotomap.identifynn.Model(**model_args)
        self.model.load_state_dict(torch.load(model_path))

    def get_new_moles(self, frame):
        import torch
        import torch.utils.data

        import mel.rotomap.identifynn

        class_to_index2 = self.class_to_index.copy()
        for m in frame.moles:
            uuid_ = m["uuid"]
            if uuid_ not in class_to_index2:
                class_to_index2[uuid_] = -1

        datadict = collections.defaultdict(list)
        mel.rotomap.identifynn.extend_dataset_by_frame(
            dataset=datadict,
            frame=frame,
            image_size=self.metadata["image_size"],
            part_to_index=self.part_to_index,
            do_channels=False,
            channel_cache=None,
            class_to_index=class_to_index2,
            escale=1.0,
            etranslate=0.0,
        )

        dataset = mel.rotomap.identifynn.RotomapsDataset(
            datadict,
            classes=self.classes,
            class_to_index=class_to_index2,
            in_fields=self.in_fields,
            out_fields=self.out_fields,
        )

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

        new_moles = list(frame.moles)
        self.model.eval()
        with torch.no_grad():
            for i, xb, _ in dataloader:
                if new_moles[i][mel.rotomap.moles.KEY_IS_CONFIRMED]:
                    continue
                out = self.model(xb)
                preds = torch.argmax(out[0], dim=1)
                new_moles[i]["uuid"] = self.classes[preds]
        return new_moles


def fit(
    epochs,
    train_dataloader,
    valid_dataloader,
    model,
    train_record,
    valid_record,
    *,
    opt=None,
    loss_func=None,
    scheduler=None,
):

    if opt is None:
        opt = torch.optim.Adam(model.parameters())

    if loss_func is None:
        loss_func = torch.nn.CrossEntropyLoss()

    iters_per_epoch = len(train_dataloader) * 2 + len(valid_dataloader)

    with tqdm.tqdm(total=iters_per_epoch * epochs, smoothing=0) as pbar:
        for _ in range(epochs):
            model.train()
            for i, xb, yb in train_dataloader:
                opt.zero_grad()
                out = model(xb)
                loss = loss_func(out, yb)

                loss.backward()
                opt.step()
                if scheduler is not None:
                    scheduler.step()

                pbar.update(1)

            model.eval()
            with torch.no_grad():
                with train_record.batch_ctx():
                    for i, xb, yb in train_dataloader:
                        out = model(xb)
                        loss = loss_func(out, yb)
                        train_record.batch(i, yb, out, loss)
                        pbar.update(1)

            model.eval()
            with torch.no_grad():
                with valid_record.batch_ctx():
                    for i, xb, yb in valid_dataloader:
                        out = model(xb)
                        loss = loss_func(out, yb)
                        valid_record.batch(i, yb, out, loss)
                        pbar.update(1)

            if valid_dataloader:
                pbar.set_description(
                    f"valid: {int(valid_record.acc[-1] * 100)}%"
                )


def make_convnet2d(width, depth, channels_in):
    return torch.nn.Sequential(
        torch.nn.BatchNorm2d(channels_in),
        make_cnn_layer(channels_in, width),
        *[make_cnn_layer(width, width) for _ in range(depth - 1)],
        torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        Lambda(lambda x: x.view(x.size(0), -1)),
    )


def make_cnn_layer(in_width, out_width):
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            in_width, out_width, kernel_size=3, stride=2, padding=1, bias=False
        ),
        torch.nn.BatchNorm2d(out_width),
        torch.nn.ReLU(inplace=True),
    )


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class FitRecord:
    def __init__(self):
        self.loss = []
        self.acc = []
        self.most_confused = collections.Counter()
        self._in_batch = False
        self._reset_batch()

    def _reset_batch(self):
        assert not self._in_batch
        self._batch_len = 0
        self._batch_total_loss = 0
        self._batch_total_correct = 0
        self._batch_total_items = 0
        self._batch_most_confused = collections.Counter()

    def to_dict(self):
        return {
            "loss": self.loss,
            "acc": self.acc,
            "most_confused": self.most_confused,
        }

    @staticmethod
    def from_dict(d):
        result = FitRecord()
        result.loss = d["loss"]
        result.acc = d["acc"]
        result.most_confused = d["most_confused"]
        return result

    @contextlib.contextmanager
    def batch_ctx(self):
        self._reset_batch()
        self._in_batch = True
        try:
            yield
        finally:
            self._in_batch = False
        if self._batch_len:
            self.record(
                self._batch_total_loss / self._batch_len,
                self._batch_total_correct / self._batch_total_items,
                self._batch_most_confused,
            )

    def batch(self, i, yb, out, loss):
        if not self._in_batch:
            raise ValueError("Must be called within a batch_ctx.")
        self._batch_total_loss += loss.item()
        preds = torch.argmax(out[0], dim=1)
        correct = (preds == yb[0]).float().sum()
        self._batch_most_confused += collections.Counter(
            tuple(sorted((p.item(), label.item())))
            for p, label in zip(preds, yb[0])
            if p != label
        )
        self._batch_total_correct += correct
        self._batch_total_items += len(i)
        self._batch_len += 1

    def record(self, loss, acc, most_confused):
        self.loss.append(float(loss))
        self.acc.append(float(acc))
        self.most_confused = most_confused


class RotomapsClassMapping:
    def __init__(self, rotomap_dict):
        all_uuids = set()
        for rotomap_list in rotomap_dict.values():
            for rotomap in rotomap_list:
                for _, moles in rotomap.yield_mole_lists():
                    for m in moles:
                        all_uuids.add(m["uuid"])

        self.classes = sorted(list(all_uuids))
        self.class_to_index = {cls: i for i, cls in enumerate(self.classes)}


def resize(image, image_size):
    return torchvision.transforms.Resize(image_size)(image)


def make_model_and_fit(
    train_dataset,
    valid_dataset,
    train_dataloader,
    valid_dataloader,
    part_to_index,
    model_config,
    train_config,
    old_results=None,
):

    train_fit_record = FitRecord()
    valid_fit_record = FitRecord()

    model_args = dict(
        cnn_width=model_config["cnn_width"],
        cnn_depth=model_config["cnn_depth"],
        num_parts=len(part_to_index),
        num_classes=len(train_dataset.classes),
        num_cnns=3,
        channels_in=2,
    )

    if old_results is not None:
        model = old_results["model"]
        old_part_to_index = old_results["part_to_index"]
        old_classes = old_results["classes"]
        old_model_args = old_results["model_args"]
        if (
            old_part_to_index != part_to_index
            or old_classes != train_dataset.classes
        ):
            # TODO: support copying over embeddings and other bits that are the
            # same, and don't need to be re-learned.
            old_model_args["num_parts"] = model_args["num_parts"]
            old_model_args["num_classes"] = model_args["num_classes"]
            model.reset_num_parts_classes(
                new_num_parts=old_model_args["num_parts"],
                new_num_classes=old_model_args["num_classes"],
            )
        if old_model_args != model_args:
            raise Exception(
                f"Old model args not compatible.\n"
                f"old: {old_model_args}\n"
                f"new: {model_args}\n"
            )
    else:
        model = Model(**model_args)

    timer = Timer()

    def loss_func(model_out, out_data):
        assert len(out_data) == 2
        f = torch.nn.functional
        return (
            f.cross_entropy(model_out[0], out_data[0])
            # + f.mse_loss(model_out[1], out_data[1])
            # + f.mse_loss(model_out[2] / 8, out_data[2] / 8)
        )

    params = [
        {"params": model.embedding.parameters()},
        {"params": model.fc.parameters()},
    ]

    # Note that the static analysis tool 'vulture' doesn't seem to be happy
    # with setting requires_grad but not reading it here. The only workaround
    # appears to be ignoring the whole file.

    if train_config["train_conv"]:
        for p in model.conv.parameters():
            p.requires_grad = True
        params.append({"params": model.conv.parameters()})
    else:
        for p in model.conv.parameters():
            p.requires_grad = False

    opt = torch.optim.AdamW(
        params=params,
        weight_decay=train_config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=train_config["learning_rate"],
        epochs=train_config["epochs"],
        steps_per_epoch=len(train_dataloader),
    )

    fit(
        train_config["epochs"],
        train_dataloader,
        valid_dataloader,
        model,
        train_fit_record,
        valid_fit_record,
        opt=opt,
        loss_func=loss_func,
        scheduler=scheduler,
    )

    results = {
        "elapsed": timer.elapsed(),
        "train_fit_record": train_fit_record.to_dict(),
        "valid_fit_record": valid_fit_record.to_dict(),
        "model": model,
        "model_args": model_args,
        "part_to_index": part_to_index,
        "classes": train_dataset.classes,
    }

    return results


def yield_frame_mole_maps_detail(
    frame, final_image_size, zoom, escale, etranslate
):
    ellipse = frame.metadata["ellipse"]
    elspace = mel.lib.ellipsespace.Transform(ellipse)

    image_size = final_image_size * zoom

    frame_map = torch.zeros(1, image_size, image_size)
    for uuid_, pos in frame.moledata.uuid_points.items():
        epos = elspace.to_space(pos)
        ipos = numpy.array(epos)
        ipos *= image_size * 0.3 * escale
        ipos += image_size * 0.5
        ipos += etranslate
        splat5(frame_map[0], ipos[0], ipos[1])

    max_point = image_size - final_image_size
    half_final_image_size = final_image_size // 2

    for uuid_, pos in frame.moledata.uuid_points.items():
        mole_mark = torch.zeros(1, image_size, image_size)
        epos = elspace.to_space(pos)
        ipos = numpy.array(epos)
        ipos *= image_size * 0.3 * escale
        ipos += image_size * 0.5
        ipos += etranslate
        splat5(mole_mark[0], ipos[0], ipos[1])

        x, y = [int(p) for p in ipos]
        left = max(0, min(max_point, x - half_final_image_size))
        right = left + final_image_size
        top = max(0, min(max_point, y - half_final_image_size))
        bottom = top + final_image_size

        result = torch.cat(
            (
                frame_map[:, top:bottom, left:right],
                mole_mark[:, top:bottom, left:right],
            )
        )
        yield uuid_, result


def yield_frame_mole_maps(frame, image_size, escale, etranslate):
    ellipse = frame.metadata["ellipse"]
    elspace = mel.lib.ellipsespace.Transform(ellipse)

    frame_map = torch.zeros(1, image_size, image_size)
    for uuid_, pos in frame.moledata.uuid_points.items():
        epos = elspace.to_space(pos)
        ipos = numpy.array(epos)
        ipos *= image_size * 0.3 * escale
        ipos += image_size * 0.5
        ipos += etranslate
        splat5(frame_map[0], ipos[0], ipos[1])

    for uuid_, pos in frame.moledata.uuid_points.items():
        mole_mark = torch.zeros(1, image_size, image_size)
        epos = elspace.to_space(pos)
        ipos = numpy.array(epos)
        ipos *= image_size * 0.3 * escale
        ipos += image_size * 0.5
        ipos += etranslate
        splat5(mole_mark[0], ipos[0], ipos[1])

        yield uuid_, torch.cat((frame_map, mole_mark))


def yield_frame_part_index(frame, part_to_index):
    part_name = frame_to_part_name(frame)
    part_index = part_to_index[part_name]
    for uuid_, pos in frame.moledata.uuid_points.items():
        yield uuid_, part_index


def frame_to_part_name(frame):
    return f"{frame.path.parents[2].stem}/{frame.path.parents[1].stem}"


def unzip_dataset_part(uuid_list, dataset_generator):
    dataset_part = list(dataset_generator)
    data_list = []
    for uuid_, item in zip(uuid_list, dataset_part):
        item_uuid, data = item
        if item_uuid != uuid_:
            raise ValueError("uuids don't match")
        data_list.append(data)
    assert len(data_list) == len(dataset_part)
    return data_list


def make_dataset(
    rotomaps,
    image_size,
    part_to_index,
    do_channels,
    channel_cache,
    class_mapping,
    augmentations,
):

    if augmentations is None:
        augmentations = [(1, numpy.array([0, 0]))]

    total_frames = 0
    for rotomap in rotomaps:
        total_frames += len(rotomap.image_paths)

    dataset = collections.defaultdict(list)
    with tqdm.tqdm(total=total_frames * len(augmentations)) as pbar:
        for rotomap in rotomaps:
            for frame in rotomap.yield_frames():
                for escale, etranslate in augmentations:
                    extend_dataset_by_frame(
                        dataset,
                        frame,
                        image_size,
                        part_to_index,
                        do_channels,
                        channel_cache,
                        class_mapping.class_to_index,
                        escale,
                        etranslate,
                    )
                    pbar.update(1)

    return dataset


def make_data(repo_path, data_config, channel_cache=None):

    parts_path = repo_path / "rotomaps" / "parts"

    if data_config["rotomaps"][0] == "subpart":
        part, subpart = data_config["rotomaps"][1:]
        rotomaps = get_subpart_rotomap(parts_path, part, subpart)
    elif data_config["rotomaps"] == "lowerlimbs":
        rotomaps = get_lower_limb_rotomaps(parts_path)
    elif data_config["rotomaps"] == "limbs":
        rotomaps = get_limb_rotomaps(parts_path)
    elif data_config["rotomaps"] == "all":
        rotomaps = get_all_rotomaps(parts_path)
    else:
        raise Exception("Unhandled rotomap type")

    train_rotomaps, valid_rotomaps = split_train_valid(
        rotomaps, data_config["train_proportion"]
    )

    if data_config["do_augmentation"]:
        assert False

    image_size = data_config["image_size"]
    do_channels = data_config["do_channels"]

    part_to_index = {p: i for i, p in enumerate(sorted(rotomaps.keys()))}

    class_mapping = RotomapsClassMapping(rotomaps)

    in_fields = ["part_index"]
    if do_channels:
        in_fields.append("channels")
    else:
        in_fields.extend(["molemap", "molemap_detail_2", "molemap_detail_4"])

    out_fields = ["uuid_index", "mole_count"]

    augmentations = [
        (scale, numpy.array([x, y]))
        for scale in [1, 0.99, 0.97, 0.95, 0.93]
        for x in [-0.01, 0, 0.01]
        for y in [-0.01, 0, 0.01]
    ]
    augmentations = None

    train_dataset = RotomapsDataset(
        make_dataset(
            train_rotomaps,
            image_size,
            part_to_index,
            do_channels,
            channel_cache,
            class_mapping,
            augmentations=augmentations,
        ),
        classes=class_mapping.classes,
        class_to_index=class_mapping.class_to_index,
        in_fields=in_fields,
        out_fields=out_fields,
    )
    valid_dataset = RotomapsDataset(
        make_dataset(
            valid_rotomaps,
            image_size,
            part_to_index,
            do_channels,
            channel_cache,
            class_mapping,
            augmentations=None,
        ),
        classes=class_mapping.classes,
        class_to_index=class_mapping.class_to_index,
        in_fields=in_fields,
        out_fields=out_fields,
    )

    if not train_dataset:
        raise Exception(
            f"No data in training dataset. "
            f"Tried these rotomaps: {train_rotomaps}"
        )

    if not valid_dataset and data_config["train_proportion"] != 1:
        raise Exception(
            f"No data in validation dataset. "
            f"Tried these rotomaps: {valid_rotomaps}"
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=data_config["batch_size"], shuffle=True
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=data_config["batch_size"]
    )

    return (
        train_dataset,
        valid_dataset,
        train_dataloader,
        valid_dataloader,
        part_to_index,
    )


def splat5(tensor, x, y, alpha=1.0):
    splat(tensor, x, y, alpha)
    splat(tensor, x + 1, y, alpha)
    splat(tensor, x - 1, y, alpha)
    splat(tensor, x, y + 1, alpha)
    splat(tensor, x, y - 1, alpha)


def splat(tensor, x, y, alpha=1.0):
    intx, inty = int(x), int(y)
    partx, party = x - intx, y - inty
    if partx <= 0.5:
        x1, x2 = intx - 1, intx
    else:
        x1, x2 = intx, intx + 1
    if party <= 0.5:
        y1, y2 = inty - 1, inty
    else:
        y1, y2 = inty, inty + 1
    cx1, cx2 = 1 - partx, partx
    cy1, cy2 = 1 - party, party

    draw_add(tensor, x1, y1, cy1 * cx1 * alpha)
    draw_add(tensor, x2, y1, cy1 * cx2 * alpha)
    draw_add(tensor, x1, y2, cy2 * cx1 * alpha)
    draw_add(tensor, x2, y2, cy2 * cx2 * alpha)


def draw_add(tensor, x, y, value):
    if x < 0 or y < 0:
        return
    if x >= tensor.shape[1] or y >= tensor.shape[0]:
        return
    tensor[y][x] += value


def split_train_valid(rotomaps, train_split=0.8):
    train_rotomaps = []
    valid_rotomaps = []
    for part, rotomap_list in rotomaps.items():
        empty_rotomaps = [
            r
            for r in rotomap_list
            if all(("ellipse" not in f.metadata) for f in r.yield_frames())
        ]
        nonempty_rotomaps = [
            r for r in rotomap_list if r not in empty_rotomaps
        ]
        num_train_rotomaps = int(len(nonempty_rotomaps) * train_split)
        num_valid_rotomaps = len(nonempty_rotomaps) - num_train_rotomaps
        if train_split != 1:
            assert num_valid_rotomaps
        train_rotomaps.extend(nonempty_rotomaps[:num_train_rotomaps])
        valid_rotomaps.extend(nonempty_rotomaps[num_train_rotomaps:])
        # train_rotomaps.extend(rotomap_list[num_valid_rotomaps:])
        # valid_rotomaps.extend(rotomap_list[:num_valid_rotomaps])
    return train_rotomaps, valid_rotomaps


def get_lower_limb_rotomaps(parts_path):
    parts = {
        parts_path
        / "LeftLeg": [
            parts_path / "LeftLeg" / "Lower",
            # parts_path / "LeftLeg" / "Upper",
        ],
        parts_path
        / "RightLeg": [
            parts_path / "RightLeg" / "Lower",
            # parts_path / "RightLeg" / "Upper",
        ],
        parts_path
        / "LeftArm": [
            parts_path / "LeftArm" / "Lower",
            # parts_path / "LeftArm" / "Upper",
        ],
        parts_path
        / "RightArm": [
            parts_path / "RightArm" / "Lower",
            # parts_path / "RightArm" / "Upper",
        ],
    }
    all_rotomaps = collections.defaultdict(list)
    for part, subpart_list in parts.items():
        for subpart in subpart_list:
            for p in sorted(subpart.iterdir()):
                all_rotomaps[f"{part.stem}:{subpart.stem}"].append(
                    mel.rotomap.moles.RotomapDirectory(p)
                )
    return all_rotomaps


def get_subpart_rotomap(parts_path, part, subpart):
    parts = {parts_path / part: [parts_path / part / subpart]}
    all_rotomaps = collections.defaultdict(list)
    for part, subpart_list in parts.items():
        for subpart in subpart_list:
            for p in sorted(subpart.iterdir()):
                all_rotomaps[f"{part.stem}/{subpart.stem}"].append(
                    mel.rotomap.moles.RotomapDirectory(p)
                )
    return all_rotomaps


def get_limb_rotomaps(parts_path):
    bits = [
        parts_path / "LeftArm",
        parts_path / "RightArm",
        parts_path / "LeftLeg",
        parts_path / "RightLeg",
    ]

    all_rotomaps = collections.defaultdict(list)
    for part in bits:
        for subpart in part.iterdir():
            for p in subpart.iterdir():
                all_rotomaps[f"{part.stem}/{subpart.stem}"].append(
                    mel.rotomap.moles.RotomapDirectory(p)
                )

    return all_rotomaps


def get_all_rotomaps(parts_path):
    all_rotomaps = collections.defaultdict(list)
    for part in parts_path.iterdir():
        for subpart in part.iterdir():
            subpart_paths = sorted(p for p in subpart.iterdir())
            for p in subpart_paths:
                all_rotomaps[f"{part.stem}/{subpart.stem}"].append(
                    mel.rotomap.moles.RotomapDirectory(p)
                )
    return all_rotomaps


class Timer:
    def __init__(self):
        self.then = time.time()

    def elapsed(self):
        now = time.time()
        return now - self.then


class RotomapsDataset:
    def __init__(self, data, classes, class_to_index, in_fields, out_fields):
        self._data = data
        self.classes = classes
        self.class_to_index = class_to_index
        self._in_fields, self._out_fields = in_fields, out_fields

    def __getitem__(self, index):
        in_data = [self._data[field][index] for field in self._in_fields]
        out_data = [self._data[field][index] for field in self._out_fields]

        return (index, in_data, out_data)

    def __len__(self):
        return len(self._data[self._in_fields[0]])


def extend_dataset_by_frame(
    dataset,
    frame,
    image_size,
    part_to_index,
    do_channels,
    channel_cache,
    class_to_index,
    escale,
    etranslate,
):
    if "ellipse" not in frame.metadata:
        return

    uuid_list = [uuid_ for uuid_, pos in frame.moledata.uuid_points.items()]
    dataset["uuid"].extend(uuid_list)

    dataset["pos"].extend(
        [pos for uuid_, pos in frame.moledata.uuid_points.items()]
    )

    dataset["uuid_index"].extend(
        [class_to_index[uuid_] for uuid_ in uuid_list]
    )

    # pylint: disable=not-callable
    dataset["mole_count"].extend(
        [torch.tensor([len(uuid_list)], dtype=torch.float)] * len(uuid_list)
    )
    # pylint: enable=not-callable

    def extend_dataset(field_name, dataset_part):
        dataset[field_name].extend(unzip_dataset_part(uuid_list, dataset_part))

    extend_dataset("part_index", yield_frame_part_index(frame, part_to_index))

    if do_channels:
        assert False, "Implement augmentations"
    else:
        extend_dataset(
            "molemap",
            yield_frame_mole_maps(frame, image_size, escale, etranslate),
        )
        extend_dataset(
            "molemap_detail_2",
            yield_frame_mole_maps_detail(
                frame, image_size, 2, escale, etranslate
            ),
        )
        extend_dataset(
            "molemap_detail_4",
            yield_frame_mole_maps_detail(
                frame, image_size, 4, escale, etranslate
            ),
        )


class Model(torch.nn.Module):
    def __init__(
        self,
        cnn_width,
        cnn_depth,
        num_parts,
        num_classes,
        num_cnns,
        channels_in=2,
    ):
        super().__init__()

        self.embedding_len = num_parts // 2
        self.embedding = None
        if num_parts:
            self.embedding = torch.nn.Embedding(num_parts, num_parts // 2)

        self.conv = make_convnet2d(
            cnn_width, cnn_depth, channels_in=channels_in
        )

        self._num_cnns = num_cnns
        self.end_width = (cnn_width * num_cnns) + self.embedding_len

        self.fc = None
        if num_classes:
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(self.end_width, num_classes),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(num_features=num_classes),
                torch.nn.Linear(num_classes, num_classes),
            )

    def forward(self, data):
        part, *rest = data
        part_embedding = self.embedding(part)

        convs_out = []
        for i, image in enumerate(rest):
            if i == self._num_cnns:
                break
            convs_out.append(self.conv(image))

        combined = torch.cat((*convs_out, part_embedding), 1)

        result = [self.fc(combined)]

        return result

    def reset_num_parts_classes(self, new_num_parts, new_num_classes):
        self.end_width -= self.embedding_len
        self.embedding_len = new_num_parts // 2
        self.end_width += self.embedding_len
        self.embedding = torch.nn.Embedding(new_num_parts, new_num_parts // 2)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.end_width, new_num_classes),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=new_num_classes),
            torch.nn.Linear(new_num_classes, new_num_classes),
        )

    def clear_non_cnn(self):
        self.end_width -= self.embedding_len
        self.embedding_len = 0
        self.embedding = None
        self.fc = None


# -----------------------------------------------------------------------------
# Copyright (C) 2019 Angelos Evripiotis.
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
