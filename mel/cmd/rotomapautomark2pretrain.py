"""Train to automatically mark moles on rotomap images."""

import torch
import torchvision

import mel.lib.math
import mel.rotomap.detectmolesnn
import mel.rotomap.moles


def setup_parser(parser):
    parser.add_argument(
        "IMAGES", nargs="+", help="A list of paths to images to automark."
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print information about the processing.",
    )


def process_args(args):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )

    # resnet = torchvision.models.resnet18(pretrained=True)
    resnet = torchvision.models.resnet50(pretrained=True)

    for path in args.IMAGES:
        if args.verbose:
            print(path)
        frame = mel.rotomap.moles.RotomapFrame(path)
        data = mel.rotomap.detectmolesnn.get_tile_locations_activations(
            frame, transforms, resnet
        )
        if data is not None:
            # torch.save(data, path + ".resnet18.pt")
            torch.save(data, path + ".resnet50.pt")
        else:
            print("Nothing to save.")


# -----------------------------------------------------------------------------
# Copyright (C) 2020 Angelos Evripiotis.
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
