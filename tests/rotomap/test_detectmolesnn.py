"""Test suite for `mel.rotomap.detectmolesnn`."""

import pytest
import torch

import mel.rotomap.detectmolesnn


def test_sorted_unique_images_sync():
    a = torch.tensor(
        [
            [[[1]]],
            [[[0]]],
            [[[0]]],
            [[[1]]],
            [[[1]]],
        ],
        names=list("NCHW"),
    )
    b = torch.tensor(
        [
            [[[1]]],
            [[[0]]],
            [[[1]]],
            [[[0]]],
            [[[1]]],
        ],
        names=list("NCHW"),
    )
    a2 = torch.tensor(
        [
            [[[0]]],
            [[[0]]],
            [[[1]]],
            [[[1]]],
        ],
        names=list("NCHW"),
    )
    b2 = torch.tensor(
        [
            [[[0]]],
            [[[1]]],
            [[[0]]],
            [[[1]]],
        ],
        names=list("NCHW"),
    )
    a_out, b_out = mel.rotomap.detectmolesnn.sorted_unique_images_sync(a, b)
    print(a_out)
    assert torch.equal(a_out, a2)
    assert torch.equal(b_out, b2)


def test_dice_loss_happy():
    prediction = torch.tensor(
        [
            [[[1]]],
            [[[0]]],
            [[[0]]],
            [[[1]]],
        ],
        names=list("NCHW"),
    )
    target = torch.tensor(
        [
            [[[1]]],
            [[[0]]],
            [[[1]]],
            [[[0]]],
        ],
        names=list("NCHW"),
    )
    loss = mel.rotomap.detectmolesnn.dice_loss(prediction, target)
    assert loss == (2 * 1) / (2 + 2)


def test_dice_loss_domain():
    prediction = torch.tensor(
        [
            [[[1.1]]],
            [[[0]]],
            [[[0]]],
            [[[-0.1]]],
        ],
        names=list("NCHW"),
    )
    target = torch.tensor(
        [
            [[[1]]],
            [[[0]]],
            [[[1]]],
            [[[0]]],
        ],
        names=list("NCHW"),
    )
    with pytest.raises(ValueError):
        mel.rotomap.detectmolesnn.dice_loss(prediction, target)


def test_shuffled_images_sync_3x1x1x1():
    p = torch.tensor(
        [
            [[[1]]],
            [[[2]]],
            [[[3]]],
        ],
        names=list("NCHW"),
    )
    p2 = mel.rotomap.detectmolesnn.shuffled_images_sync(p)[0]
    assert p2.shape == p.shape
    assert torch.equal(
        p2.rename(None).flatten().sort().values, torch.tensor([1, 2, 3])
    )


def test_pixelise_2x2x3():
    t = torch.tensor(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[11, 12, 13], [14, 15, 16]],
        ],
        names=list("CHW"),
    )
    p = torch.tensor(
        [
            [[[1]], [[11]]],
            [[[2]], [[12]]],
            [[[3]], [[13]]],
            [[[4]], [[14]]],
            [[[5]], [[15]]],
            [[[6]], [[16]]],
        ],
        names=list("NCHW"),
    )
    assert torch.equal(p, mel.rotomap.detectmolesnn.pixelise(t))


def make_image(num_c, num_h, num_w):
    t_py = []
    for c in range(1, num_c + 1):
        t_py.append([])
        i = 1
        for _ in range(num_h):
            t_py[-1].append([])
            for _ in range(num_w):
                t_py[-1][-1].append(i + (c * 10))
                i += 1
    return torch.tensor(t_py, names=list("CHW"))


def test_make_image():
    assert torch.equal(
        make_image(1, 1, 1), torch.tensor([[[11]]], names=list("CHW"))
    )

    t = torch.tensor(
        [
            [[11, 12, 13], [14, 15, 16]],
            [[21, 22, 23], [24, 25, 26]],
        ],
        names=list("CHW"),
    )
    assert torch.equal(make_image(2, 2, 3), t)


def make_pixelised(num_c, num_h, num_w):
    t_py = []
    i = 1
    for _ in range(num_h):
        for _ in range(num_w):
            t_py.append([])
            for c in range(1, num_c + 1):
                t_py[-1].append([[i + (c * 10)]])
            i += 1
    return torch.tensor(t_py, names=list("NCHW"))


def test_make_pixelised():
    assert torch.equal(
        make_pixelised(1, 1, 1), torch.tensor([[[[11]]]], names=list("NCHW"))
    )

    t = torch.tensor(
        [
            [[[11]], [[21]]],
            [[[12]], [[22]]],
            [[[13]], [[23]]],
            [[[14]], [[24]]],
            [[[15]], [[25]]],
            [[[16]], [[26]]],
        ],
        names=list("NCHW"),
    )
    assert t.shape == (6, 2, 1, 1)
    assert torch.equal(make_pixelised(2, 2, 3), t)


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("height", [1, 2, 3, 4])
@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_pixelise_multidim(channels, height, width):
    t = make_image(channels, height, width)
    p = make_pixelised(channels, height, width)
    assert torch.equal(p, mel.rotomap.detectmolesnn.pixelise(t))


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
