"""FileSystem helpers."""

import os


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
    return lower_ext in ('.jpg', '.jpeg')
# -----------------------------------------------------------------------------
# Copyright (C) 2017 Angelos Evripiotis.
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
