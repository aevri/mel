"""FileSystem helpers."""

import os


def expand_dirs_to_jpegs(path_list):
    image_paths = []
    for path in path_list:
        if os.path.isdir(path):
            image_paths.extend(list(yield_only_jpegs_from_dir(path)))
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
