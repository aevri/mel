"""Capture images from an attached microscope and add to existing moles."""

import datetime
import os

import cv2
import numpy

import mel.lib.common
import mel.lib.datetime
import mel.lib.image
import mel.lib.moleimaging
import mel.lib.ui


def setup_parser(parser):
    parser.add_argument(
        "PATH",
        nargs="+",
        type=str,
        help="Path to the mole to add new microscope images to.",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=None,
        help="Width of the preview display window.",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=None,
        help="Width of the preview display window.",
    )
    parser.add_argument(
        "--min-compare-age-days",
        type=int,
        default=None,
        help="Minimum age of the micro image to compare with, if possible.",
    )
    parser.add_argument(
        "--video-device-index",
        "-D",
        type=int,
        default=0,
        help="The index of the device to take images from.",
    )

    # From NHS 'Moles' page:
    # http://www.nhs.uk/Conditions/Moles/Pages/Introduction.aspx
    # > You should check your skin every few months for any new moles that
    # > develop (particularly after your teenage years, when new moles become
    # > less common) or any changes to existing moles. A mole can change in
    # > weeks or months.
    #
    # Compare at least 180 days back, if possible.


def get_context_image_name(path):
    # Paths should alpha-sort to recent last, pick the first jpg
    children = reversed(sorted(os.listdir(path)))
    for name in children:
        # TODO: support more than just '.jpg'
        if name.lower().endswith(".jpg"):
            return os.path.join(path, name)

    return None


def get_dirs_to_path(path_in):
    """Return a list of the intermediate paths between cwd and path.

    Raise if path is not below the current working directory (cwd).

    :returns: list of strings, includes cwd and destination path
    :path: string path
    """
    cwd = os.getcwd()
    path_abs = os.path.abspath(path_in)
    if cwd != os.path.commonprefix([cwd, path_abs]):
        raise Exception("{} is not under cwd ({})".format(path_abs, cwd))
    path_rel = os.path.relpath(path_abs, cwd)
    path_list = []
    while path_rel:
        path_rel, tail = os.path.split(path_rel)
        path_list.append(os.path.join(cwd, path_rel, tail))
    path_list.append(cwd)
    return path_list


def load_context_images(path):
    image_list = []
    path_list = get_dirs_to_path(path)
    for path in path_list:
        name = get_context_image_name(path)
        if name:
            image_list.append(mel.lib.image.load_image(name))
    return image_list


def pick_comparison_path(path_list, min_compare_age_days):
    """Return the most appropriate image path to compare with, or None."""
    path_dt_list = [
        (x, mel.lib.datetime.guess_datetime_from_path(x)) for x in path_list
    ]

    for path, dt in path_dt_list:
        if dt is None:
            raise Exception("Could not determine date", path)

    path_dt_list.sort(key=lambda x: x[1], reverse=True)

    if min_compare_age_days is not None:
        delta = datetime.timedelta(min_compare_age_days)
        appropriate_date = datetime.datetime.now() - delta

        for path, dt in path_dt_list:
            if dt <= appropriate_date:
                return path

    return path_dt_list[-1][0] if path_dt_list else None


def get_comparison_image_path(path, min_compare_age_days):

    micro_path = os.path.join(path, "__micro__")
    if not os.path.exists(micro_path):
        return None

    # List all the 'jpg' files in the micro dir
    # TODO: support more than just '.jpg'
    images = [x for x in os.listdir(micro_path) if x.lower().endswith(".jpg")]
    path = pick_comparison_path(images, min_compare_age_days)
    if path:
        return os.path.join(micro_path, path)
    else:
        return None


def load_comparison_image(path, min_compare_age_days):
    micro_path = get_comparison_image_path(path, min_compare_age_days)
    if micro_path is None:
        return None
    return micro_path, mel.lib.image.load_image(micro_path)


def process_args(args):
    cap = cv2.VideoCapture(args.video_device_index)
    if not cap.isOpened():
        raise Exception("Could not open video capture device.")

    display = mel.lib.ui.MultiImageDisplay(
        "mel micro add", args.display_width, args.display_height
    )

    mel.lib.ui.bring_python_to_front()

    for mole_path in args.PATH:
        print(mole_path)
        display.reset()
        process_path(mole_path, args.min_compare_age_days, display, cap)


def process_path(mole_path, min_compare_age_days, display, cap):
    comparison_image_data = load_comparison_image(
        mole_path, min_compare_age_days
    )

    if comparison_image_data is not None:
        comparison_path, comparison_image = comparison_image_data
        display.set_title(comparison_path)
    else:
        display.set_title(mole_path)

    context_images = load_context_images(mole_path)
    for image in context_images:
        display.add_image(image)

    if context_images:
        display.new_row()

    if comparison_image_data:
        display.add_image(comparison_image)

    # wait for confirmation
    mole_acquirer = mel.lib.moleimaging.MoleAcquirer()
    is_finished = False
    ret, frame = cap.read()
    if not ret:
        raise Exception("Could not read frame.")
    capindex = display.add_image(frame, "capture")
    while not is_finished:
        frame = capture(cap, display, capindex, mole_acquirer)
        print("Press space to save and exit, 'r' to retry, 'u' to rotate 180.")
        print("Press 'a' to abort without saving and exit with an error code.")

        is_finished = True
        for key in mel.lib.ui.yield_keys_until_quitkey(
            quit_key=" ", error_key="a"
        ):

            if key == ord("r"):
                print("Retry capture")
                is_finished = False
                break
            elif key == ord("u"):
                print("Rotated 180.")
                frame = mel.lib.image.rotated180(frame)
                display.update_image(frame, capindex)

    # write the mole image
    filename = mel.lib.datetime.make_now_datetime_string() + ".jpg"
    dirname = os.path.join(mole_path, "__micro__")
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    file_path = os.path.join(dirname, filename)
    mel.lib.common.write_image(file_path, frame)


def capture(cap, display, capindex, mole_acquirer):

    # loop until the user presses a key
    print("Press 'c' to force capture a frame, 'a' to abort.")

    centre = None
    rotation = None

    for frame, key in mel.lib.ui.yield_frames_keys(cap, error_key="a"):

        if key == ord("c"):
            print("Force capturing frame.")
            centre = None
            rotation = None
            break

        _, stats = mel.lib.moleimaging.find_mole(frame)
        asys_image = numpy.copy(frame)
        is_aligned, centre, rotation = mel.lib.moleimaging.annotate_image(
            asys_image, is_rot_sensitive=False
        )

        mole_acquirer.update(stats)

        if mole_acquirer.is_locked and is_aligned:
            break
        else:
            display.update_image(asys_image, capindex)

    normal_image = numpy.copy(frame)
    if centre is not None:
        normal_image = mel.lib.image.recentered_at(frame, centre[0], centre[1])
    if rotation is not None:
        normal_image = mel.lib.image.rotated(normal_image, rotation)

    display.update_image(normal_image, capindex)
    print("locked and aligned")

    return normal_image


# -----------------------------------------------------------------------------
# Copyright (C) 2015-2018 Angelos Evripiotis.
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
