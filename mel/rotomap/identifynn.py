"""Identify which moles are which, using neural nets."""
import collections
import contextlib
import random
import time

import cv2
import numpy
import torch.utils.data
import torchvision
import tqdm

import mel.lib.ellipsespace
import mel.rotomap.moles


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
):

    if opt is None:
        opt = torch.optim.Adam(model.parameters())

    if loss_func is None:
        loss_func = torch.nn.CrossEntropyLoss()

    iters_per_epoch = len(train_dataloader) * 2 + len(valid_dataloader)

    with tqdm.tqdm(total=iters_per_epoch * epochs, smoothing=0) as pbar:
        for epoch in range(epochs):
            model.train()
            for i, xb, yb in train_dataloader:
                opt.zero_grad()
                out = model(xb)
                loss = loss_func(out, yb)

                loss.backward()
                opt.step()

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

            pbar.set_description(
                f"valid: {int(valid_record.acc[-1] * 100)}%",
                # refresh=False,
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


def make_random_shift_crop(magnitude):
    def no_shift(t):
        return t

    def random_shift_crop(tensor_image):
        v_directions = (shift_crop_up, shift_crop_down)
        h_directions = (shift_crop_left, shift_crop_right)
        v_dir = random.choice(v_directions)
        h_dir = random.choice(h_directions)
        shifts = (no_shift, v_dir, h_dir)
        for _ in range(magnitude):
            tensor_image = random.choice(shifts)(tensor_image)
        return tensor_image

    return random_shift_crop


def get_sub_image(image, top, left, bottom, right):
    width = image.shape[1]
    height = image.shape[0]
    topc, leftc, bottomc, rightc = numpy.clip(
        [top, left, bottom, right],
        numpy.array([0, 0, 0, 0]),
        numpy.array([height, width, height, width]),
    )
    sub_image = image[topc:bottomc, leftc:rightc]
    vsize = bottom - top
    hsize = right - left
    if sub_image.shape != (bottom - top, right - left, 3):
        new_image = numpy.zeros((vsize, hsize, 3), dtype=image.dtype)
        new_image[:, :, 1] = 255
        clipped_hsize = rightc - leftc
        clipped_vsize = bottomc - topc
        new_image[:clipped_vsize, :clipped_hsize, :] = sub_image[:, :, :]
        sub_image = new_image
    return sub_image


def to_tensor(image):
    return torchvision.transforms.ToTensor()(image)


def resize(image, image_size):
    return torchvision.transforms.Resize(image_size)(image)


def to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def mask_image(original_image, mask):
    green = numpy.zeros(original_image.shape, numpy.uint8)
    green[:, :, 1] = 255
    image = cv2.bitwise_and(original_image, original_image, mask=mask)
    not_mask = cv2.bitwise_not(mask)
    green = cv2.bitwise_and(green, green, mask=not_mask)
    image = cv2.bitwise_or(image, green)
    return image


def shift_crop_right(tensor_image):
    # Delete the right column, add a column of zeros down on the left.
    new_image = torch.cat((new_col(tensor_image), tensor_image), 2)
    return new_image[:, :, :-1]


def shift_crop_left(tensor_image):
    # Delete the left column, add a column of zeros down on the right.
    new_image = torch.cat((tensor_image, new_col(tensor_image)), 2)
    return new_image[:, :, 1:]


def shift_crop_down(tensor_image):
    # Shift a row of zeros down from the top
    new_image = torch.cat((new_row(tensor_image), tensor_image), 1)
    return new_image[:, :-1, :]


def shift_crop_up(tensor_image):
    # Shift a row of zeros up from the bottom
    new_image = torch.cat((tensor_image, new_row(tensor_image)), 1)
    return new_image[:, 1:, :]


def new_col(tensor_image):
    channels = tensor_image.shape[0]
    rows = tensor_image.shape[1]
    new_row_shape = channels, rows, 1
    return torch.zeros(new_row_shape)


def new_row(tensor_image):
    channels = tensor_image.shape[0]
    cols = tensor_image.shape[2]
    new_row_shape = channels, 1, cols
    return torch.zeros(new_row_shape)


def make_model_and_fit(
    train_dataset,
    valid_dataset,
    train_dataloader,
    valid_dataloader,
    part_to_index,
    model_config,
    train_config,
):

    train_fit_record = FitRecord()
    valid_fit_record = FitRecord()

    model_args = dict(
        cnn_width=model_config["cnn_width"],
        cnn_depth=model_config["cnn_depth"],
        num_parts=len(part_to_index),
        num_classes=len(train_dataset.classes),
        use_pos=model_config["use_pos"],
        use_photo=model_config["use_photo"],
        num_cnns=3,
        channels_in=2,
        predict_pos=False,
        predict_count=False,
    )

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

    opt = torch.optim.SGD(
        params=model.parameters(),
        lr=train_config["learning_rate"],
        momentum=train_config["momentum"],
        weight_decay=train_config["weight_decay"],
    )

    # opt = AdamW(
    #     params=model.parameters(),
    #     lr=train_config["learning_rate"],
    #     weight_decay=train_config["weight_decay"],
    # )

    fit(
        train_config["epochs"],
        train_dataloader,
        valid_dataloader,
        model,
        train_fit_record,
        valid_fit_record,
        opt=opt,
        loss_func=loss_func,
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


def append_frame_data(
    i,
    frame,
    image_size,
    photo_size,
    do_photo,
    part_to_index,
    mid_point,
    quarter_point,
    num_frames,
    image_list,
):
    assert not do_photo
    # half_photo_size = photo_size // 2
    # rgb_masked = to_rgb(mask_image(frame.load_image(), frame.load_mask()))
    if do_photo:
        assert False
    else:
        frame_image = 0
    part_name = frame_to_part_name(frame)
    part_index = part_to_index[part_name]
    ellipse = frame.metadata["ellipse"]
    elspace = mel.lib.ellipsespace.Transform(ellipse)
    a = abs(i - mid_point) / mid_point
    b = abs((i - quarter_point) % num_frames - mid_point) / mid_point
    for uuid_target, target_pos in frame.moledata.uuid_points.items():
        if do_photo:
            assert False
        else:
            mole_image = 0
        image = torch.zeros(2, image_size, image_size)
        for uuid_, pos in frame.moledata.uuid_points.items():
            # if canonical_only and uuid_ not in
            # frame.moledata.canonical_uuids:
            #    continue
            epos = elspace.to_space(pos)
            ipos = numpy.array(epos)
            ipos *= image_size * 0.3
            ipos += image_size * 0.5
            splat4(image[0], ipos[0], ipos[1])
            if uuid_ == uuid_target:
                splat4(image[1], ipos[0], ipos[1])
        image_list.append(
            (
                (frame.path, uuid_target),
                (
                    image,
                    torch.tensor([a, b, epos[0], epos[1]]),
                    (frame_image, mole_image),
                    part_index,
                ),
                uuid_target,
            )
        )


def record_inputs(model, to_record, dataset):
    activations = None

    def record_response(module, input_):
        nonlocal activations
        activations = catted(activations, input_[0][0].clone())

    model.eval()
    hook = to_record.register_forward_pre_hook(record_response)
    with contextlib.ExitStack() as stack:
        stack.callback(hook.remove)
        with torch.no_grad():
            for data in tqdm.tqdm(dataset):
                model(data[1].unsqueeze(0))

    return activations


class FakePartRotomap:
    def __init__(self, num_moles, space_width):
        self.space_width = space_width
        self.positions = torch.empty(num_moles, 2).uniform_(-1, 1)
        self.positions[:, 0] *= self.space_width

    def _yield_masked_points_transformed(self, mask, transform):
        for i, v in enumerate(mask):
            if not v:
                continue
            x, y = transform(self.positions[i])
            yield i, x, y

    def yield_frame_images(self, left, image_size, warp_factor):
        top = -1
        bottom = 1
        right = left + 2

        visible = (
            (self.positions[:, 0] >= left)
            * (self.positions[:, 0] <= right)
            * (self.positions[:, 1] >= top)
            * (self.positions[:, 1] <= bottom)
            * 1
        )

        half_size = image_size // 2

        map_image = torch.zeros(image_size, image_size)

        def transform(point):
            translated = point - torch.tensor((left + 1, 0), dtype=torch.float)
            warped_x = (torch.sigmoid(translated[0] * warp_factor) - 0.5) * 2.0
            warped = translated
            warped[0] = warped_x
            return warped * half_size + half_size

        for i, x, y in self._yield_masked_points_transformed(
            visible, transform
        ):
            splat5(map_image, x, y)

        for i, x, y in self._yield_masked_points_transformed(
            visible, transform
        ):
            mole_image = torch.zeros(image_size, image_size)
            splat5(mole_image, x, y)
            yield torch.cat(
                (map_image.unsqueeze(0), mole_image.unsqueeze(0))
            ), i


class FakeBodyDataset:
    def __init__(self, parts_list, image_size):
        self._parts = parts_list
        self._image_size = image_size

        self._part_ident_offsets = [0]
        for part in parts_list:
            self._part_ident_offsets.append(
                self._part_ident_offsets[-1] + len(part.positions)
            )

        self._image_lefts = []

        self._images = None
        self._images_parts = []
        self._labels = []

        self.regen_images()

    def _ident_to_class(self, part_index, ident):
        return ident + self._part_ident_offsets[part_index]

    def regen_images(self):
        self._image_lefts = []
        for i, part in enumerate(self._parts):
            num_images = random.randint(6, 20)
            offset = part.space_width / num_images
            lefts = torch.arange(0, part.space_width, offset)
            self._image_lefts.extend([i, l] for l in lefts)

        self._images = None
        self._images_parts = []
        self._labels = []
        # TODO: randomize warp factor
        for part_index, left in self._image_lefts:
            warp_factor = random.uniform(2, 5)
            part = self._parts[part_index]
            for image, ident in part.yield_frame_images(
                left, self._image_size, warp_factor
            ):
                self._images = catted(self._images, image)
                self._images_parts.append(part_index)
                self._labels.append(self._ident_to_class(part_index, ident))

    def __getitem__(self, i):
        return i, (self._images[i], self._images_parts[i]), self._labels[i]

    def __len__(self):
        return len(self._labels)

    def num_classes(self):
        return self._part_ident_offsets[-1]

    def num_parts(self):
        return len(self._parts)


def catted(list_tensor, element_tensor):
    if list_tensor is None:
        return element_tensor.unsqueeze(0)
    return torch.cat((list_tensor, element_tensor.unsqueeze(0)))


def make_body_of_fake_parts():
    space_width = 4
    return (
        *[
            FakePartRotomap(random.randint(30, 50), space_width)
            for _ in range(10)
        ],
        *[
            FakePartRotomap(random.randint(20, 40), space_width)
            for _ in range(10)
        ],
    )


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


def yield_transformed_pos(frame):
    ellipse = frame.metadata["ellipse"]
    elspace = mel.lib.ellipsespace.Transform(ellipse)
    for uuid_, pos in frame.moledata.uuid_points.items():
        epos = elspace.to_space(pos)
        yield uuid_, torch.tensor(epos, dtype=torch.float)


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
            raise ValueError(f"uuids don't match")
        data_list.append(data)
    assert len(data_list) == len(dataset_part)
    return data_list


def make_dataset(
    rotomaps,
    image_size,
    photo_size,
    part_to_index,
    do_photo,
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
            for i, frame in enumerate(rotomap.yield_frames()):
                for escale, etranslate in augmentations:
                    extend_dataset_by_frame(
                        dataset,
                        frame,
                        image_size,
                        photo_size,
                        part_to_index,
                        do_photo,
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
    else:
        raise Exception("Unhandled rotomap type")
    # rotomaps = get_lower_limb_rotomaps(parts_path)

    train_rotomaps, valid_rotomaps = split_train_valid(
        rotomaps, data_config["train_proportion"]
    )

    if data_config["do_augmentation"]:
        assert False

    image_size = data_config["image_size"]
    photo_size = data_config["photo_size"]
    do_photo = data_config["do_photo"]
    do_channels = data_config["do_channels"]

    part_to_index = {p: i for i, p in enumerate(sorted(rotomaps.keys()))}

    class_mapping = RotomapsClassMapping(rotomaps)

    in_fields = ["part_index"]
    if do_channels:
        in_fields.append("channels")
    else:
        in_fields.extend(["molemap", "molemap_detail_2", "molemap_detail_4"])
    if do_photo:
        in_fields.extend(["frame_photo", "mole_photo"])

    out_fields = ["uuid_index", "mole_count"]
    # out_fields = ["uuid_index", "transformed_pos", "mole_count"]

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
            photo_size,
            part_to_index,
            do_photo,
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
            photo_size,
            part_to_index,
            do_photo,
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


def splat4(tensor, x, y):
    splat(tensor, x, y)
    splat(tensor, x + 1, y)
    splat(tensor, x, y + 1)
    splat(tensor, x + 1, y + 1)


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
        num_train_rotomaps = int(len(rotomap_list) * train_split)
        num_valid_rotomaps = len(rotomap_list) - num_train_rotomaps
        assert num_valid_rotomaps
        train_rotomaps.extend(rotomap_list[:num_train_rotomaps])
        valid_rotomaps.extend(rotomap_list[num_train_rotomaps:])
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
            for p in subpart.iterdir():
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
    photo_size,
    part_to_index,
    do_photo,
    do_channels,
    channel_cache,
    class_to_index,
    escale,
    etranslate,
):

    uuid_list = [uuid_ for uuid_, pos in frame.moledata.uuid_points.items()]
    dataset["uuid"].extend(uuid_list)

    dataset["pos"].extend(
        [pos for uuid_, pos in frame.moledata.uuid_points.items()]
    )

    dataset["uuid_index"].extend(
        [class_to_index[uuid_] for uuid_ in uuid_list]
    )

    dataset["mole_count"].extend(
        [torch.tensor([len(uuid_list)], dtype=torch.float)] * len(uuid_list)
    )

    def extend_dataset(field_name, dataset_part):
        dataset[field_name].extend(unzip_dataset_part(uuid_list, dataset_part))

    extend_dataset("part_index", yield_frame_part_index(frame, part_to_index))

    # TODO: Implement augmentations
    # extend_dataset(
    #     "transformed_pos",
    #     yield_transformed_pos(frame),
    # )

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
        use_pos,
        use_photo,
        num_cnns,
        channels_in=2,
        predict_pos=False,
        predict_count=False,
    ):
        super().__init__()

        self.embedding_len = num_parts // 2
        self.embedding = torch.nn.Embedding(num_parts, num_parts // 2)

        self.conv = make_convnet2d(
            cnn_width, cnn_depth, channels_in=channels_in
        )

        self._use_pos = use_pos
        assert not self._use_pos

        self._num_cnns = num_cnns
        end_width = (cnn_width * num_cnns) + self.embedding_len

        self._predict_pos = predict_pos
        if self._predict_pos:
            self.predict_pos = torch.nn.Sequential(
                torch.nn.Linear(cnn_width, 32),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(num_features=32),
                torch.nn.Linear(32, 2),
            )

        self._predict_count = predict_count
        if self._predict_count:
            self.predict_count = torch.nn.Sequential(
                torch.nn.Linear(cnn_width, 32),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(num_features=32),
                torch.nn.Linear(32, 1),
            )

        self._use_photo = use_photo
        if self._use_photo:
            end_width += 512

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(end_width, num_classes),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=num_classes),
            torch.nn.Linear(num_classes, num_classes),
        )

    def forward(self, data):
        # image, table, photo, part, rn18 = data
        # image, table, photo, part = data

        part, *rest = data
        part_embedding = self.embedding(part)

        convs_out = []
        # assert len(rest) == self._num_cnns
        for i, image in enumerate(rest):
            if i == self._num_cnns:
                break
            convs_out.append(self.conv(image))

        if self._use_pos:
            assert not self._use_pos
            assert not self._use_photo
        elif self._use_photo:
            assert not self._use_photo
        else:
            combined = torch.cat((*convs_out, part_embedding), 1)

        result = [self.fc(combined)]
        if self._predict_pos:
            result.append(self.predict_pos(convs_out[0]))
        if self._predict_count:
            result.append(self.predict_count(convs_out[0]))
        return result


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
