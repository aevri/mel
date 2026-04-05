"""FileSystem helpers."""

import collections
import collections.abc
import pathlib

DEFAULT_CLASSIFIER_PATH = "classifiers"
DEFAULT_MOLE_MARK_MODEL_NAME = "mole_mark_model"
DEFAULT_MOLE_MARK_DATACONFIG_NAME = "mole_mark_dataconfig.pkl"
TIMELOG_NAME = "timelog.csv"

ROTOMAPS_PATH = pathlib.Path("rotomaps")
MICRO_PATH = pathlib.Path("micro")


def expand_dirs_to_jpegs(path_list) -> list[str]:
    image_paths = []
    for path in path_list:
        if pathlib.Path(path).is_dir():
            image_paths.extend(sorted(yield_only_jpegs_from_dir(path)))
        else:
            image_paths.append(path)
    return image_paths


def yield_only_jpegs_from_dir(path) -> collections.abc.Generator[str]:
    for entry in pathlib.Path(path).iterdir():
        if is_jpeg_name(entry.name):
            yield str(entry)


def is_jpeg_name(filename) -> bool:
    lower_ext = pathlib.PurePath(filename).suffix.lower()
    return lower_ext in (".jpg", ".jpeg")


class NoMelrootError(Exception):
    pass


def find_melroot() -> pathlib.Path:
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


# This is quite useful for exploration sometimes.
# def list_rotomaps_by_part(parts_path):
#     all_rotomaps = collections.defaultdict(list)
#     for part in parts_path.iterdir():
#         for subpart in part.iterdir():
#             subpart_paths = sorted(p for p in subpart.iterdir())
#             for p in subpart_paths:
#                 all_rotomaps[f"{part.stem}/{subpart.stem}"].append(p)
#     return all_rotomaps


def list_rotomap_images_by_session(
    parts_path, *, exclude_parts=None
) -> collections.defaultdict[str, list]:
    images = collections.defaultdict(list)
    for part in sorted(parts_path.iterdir()):
        for subpart in sorted(part.iterdir()):
            if exclude_parts and f"{part.stem}/{subpart.stem}" in exclude_parts:
                continue
            for session in sorted(subpart.iterdir()):
                files = sorted(yield_only_jpegs_from_dir(session))
                for img in files:
                    images[f"{session.stem}"].append(img)
    return images


# -----------------------------------------------------------------------------
# Copyright (C) 2018-2023, 2026 Angelos Evripiotis.
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
