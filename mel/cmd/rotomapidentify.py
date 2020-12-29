"""Guess which mole is which in a rotomap image."""

import collections
import json
import os

import mel.lib.fs


def setup_parser(parser):
    parser.add_argument(
        "TARGET",
        nargs="+",
        help="Paths to images to identify.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print information about the processing.",
    )


def process_args(args):
    identifier = make_identifier()
    for target in args.TARGET:
        if args.verbose:
            print("Processing", target, "..")

        # part = mel.lib.fs.get_rotomap_part_from_path(melroot, target)
        frame = mel.rotomap.moles.RotomapFrame(os.path.abspath(target))

        new_moles = identifier.get_new_moles(frame)

        mel.rotomap.moles.save_image_moles(new_moles, str(frame.path))


def make_identifier():
    melroot = mel.lib.fs.find_melroot()
    model_dir = melroot / mel.lib.fs.DEFAULT_CLASSIFIER_PATH
    model_path = model_dir / "identify.pth"
    metadata_path = model_dir / "identify.json"
    return MoleIdentifier(metadata_path, model_path)


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
        self.in_fields.extend(["molemap", "molemap_detail_2", "molemap_detail_4"])
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
