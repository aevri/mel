"""Test suite for mel.rotomap.filtermarks."""

import pathlib
import pickle

import torch

import mel.rotomap.filtermarks


def test_load_pretrained_file_torch_format(tmp_path):
    """Loading a torch-format file works directly."""
    path = tmp_path / "test.pt"
    data = {
        "features": torch.tensor([1.0, 2.0]),
        "is_mole": [True, False],
        "path": "images/test.jpg",
        "weights_version": "v1",
        "metadata": [{}],
    }
    torch.save(data, path)

    loaded = mel.rotomap.filtermarks._load_pretrained_file(path)

    assert (loaded["features"] == data["features"]).all()
    assert loaded["is_mole"] == data["is_mole"]
    assert loaded["weights_version"] == "v1"


def test_load_pretrained_file_converts_pickle_format(tmp_path, capsys):
    """Old pickle-format files are converted to torch format on load."""
    path = tmp_path / "test.pt"
    data = {
        "features": torch.tensor([3.0, 4.0]),
        "is_mole": [False, True],
        "path": pathlib.PosixPath("images/old.jpg"),
        "weights_version": "v2",
        "metadata": [{}],
    }
    path.write_bytes(pickle.dumps(data))

    loaded = mel.rotomap.filtermarks._load_pretrained_file(path)

    assert (loaded["features"] == data["features"]).all()
    assert loaded["is_mole"] == data["is_mole"]
    assert loaded["weights_version"] == "v2"
    assert loaded["path"] == "images/old.jpg"

    captured = capsys.readouterr()
    assert "Converting old pickle-format cache" in captured.err

    # Verify the file was converted in-place to torch format.
    reloaded = torch.load(path, weights_only=True)
    assert (reloaded["features"] == data["features"]).all()


# -----------------------------------------------------------------------------
# Copyright (C) 2026 Angelos Evripiotis.
# Generated with assistance from Claude Code.
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
