"""A global object for debug rendering into images without around."""

import cv2


class GlobalContext:
    def __init__(self):
        self._image = None

    def arrow(self, from_, to):
        if self._image is None:
            return
        cv2.arrowedLine(
            self._image,
            tuple(from_.astype(int)),
            tuple(to.astype(int)),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def circle(self, point, radius):
        if self._image is None:
            return
        cv2.circle(
            self._image,
            tuple(point.astype(int)),
            int(radius),
            (255, 255, 255),
            2,
        )


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
