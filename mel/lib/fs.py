"""FileSystem helpers."""

import os
import pathlib

DEFAULT_CLASSIFIER_PATH = "classifiers"
DEFAULT_MOLE_MARK_MODEL_NAME = "mole_mark_model"
DEFAULT_MOLE_MARK_DATACONFIG_NAME = "mole_mark_dataconfig.pkl"
TIMELOG_NAME = "timelog.csv"

ROTOMAPS_PATH = pathlib.Path("rotomaps")
MICRO_PATH = pathlib.Path("micro")


def expand_dirs_to_jpegs(path_list):
    image_paths = []
    for path in path_list:
        if os.path.isdir(path):
            image_paths.extend(sorted(yield_only_jpegs_from_dir(path)))
        else:
            image_paths.append(path)
    return image_paths


def yield_only_jpegs_from_dir(path):
    for filename in os.listdir(path):
        if is_jpeg_name(filename):
            yield os.path.join(path, filename)


def is_jpeg_name(filename):
    lower_ext = os.path.splitext(filename)[1].lower()
    return lower_ext in (".jpg", ".jpeg")


class NoMelrootError(Exception):
    pass


def find_melroot():
    original_path = pathlib.Path.cwd()

    path = original_path
    while True:
        melroot = path / "melroot"
        if melroot.exists():
            return path

        parent = path.parent
        if parent == path:
            raise NoMelrootError(original_path)
        path = parent


# -----------------------------------------------------------------------------
# Copyright (C) 2018 Angelos Evripiotis.
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
