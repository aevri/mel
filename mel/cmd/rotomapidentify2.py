"""Guess which mole is which in a rotomap image."""

import collections
import json
import os


def setup_parser(parser):
    parser.add_argument(
        "TARGET", nargs="+", help="Paths to images to identify.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print information about the processing.",
    )


def process_args(args):
    # Some of are expensive imports, so to keep program start-up time lower,
    # import them only when necessary.
    import torch
    import torch.utils.data

    import mel.lib.fs
    import mel.rotomap.moles
    import mel.rotomap.identifynn

    melroot = mel.lib.fs.find_melroot()
    model_dir = melroot / mel.lib.fs.DEFAULT_CLASSIFIER_PATH
    model_path = model_dir / "identify.pth"
    metadata_path = model_dir / "identify.json"

    with open(metadata_path) as f:
        metadata = json.load(f)

    model_args = metadata["model_args"]
    part_to_index = metadata["part_to_index"]
    classes = metadata["classes"]
    class_to_index = {cls: i for i, cls in enumerate(classes)}

    in_fields = ["part_index"]
    in_fields.extend(["molemap", "molemap_detail_2", "molemap_detail_4"])
    out_fields = ["uuid_index", "mole_count"]

    model = mel.rotomap.identifynn.Model(**model_args)
    model.load_state_dict(torch.load(model_path))

    for target in args.TARGET:
        if args.verbose:
            print("Processing", target, "..")

        # part = mel.lib.fs.get_rotomap_part_from_path(melroot, target)
        frame = mel.rotomap.moles.RotomapFrame(os.path.abspath(target))

        class_to_index2 = class_to_index.copy()
        for m in frame.moles:
            uuid_ = m["uuid"]
            if uuid_ not in class_to_index2:
                class_to_index2[uuid_] = -1

        datadict = collections.defaultdict(list)
        mel.rotomap.identifynn.extend_dataset_by_frame(
            dataset=datadict,
            frame=frame,
            image_size=metadata["image_size"],
            photo_size=None,
            part_to_index=part_to_index,
            do_photo=False,
            do_channels=False,
            channel_cache=None,
            class_to_index=class_to_index2,
            escale=1.0,
            etranslate=0.0,
        )

        dataset = mel.rotomap.identifynn.RotomapsDataset(
            datadict,
            classes=classes,
            class_to_index=class_to_index2,
            in_fields=in_fields,
            out_fields=out_fields,
        )

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

        new_moles = list(frame.moles)
        model.eval()
        with torch.no_grad():
            for i, xb, yb in dataloader:
                if new_moles[i][mel.rotomap.moles.KEY_IS_CONFIRMED]:
                    continue
                out = model(xb)
                preds = torch.argmax(out[0], dim=1)
                new_moles[i]["uuid"] = classes[preds]

        mel.rotomap.moles.save_image_moles(new_moles, str(frame.path))


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
