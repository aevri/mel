"""User interface related things."""

import pathlib
import subprocess


class AbortKeyInterruptError(Exception):
    pass


def set_clipboard_contents(text) -> None:
    """Set the contents of the clipbaord, only works on Mac OSX.

    :returns: None
    """
    pbcopy = "/usr/bin/pbcopy"

    if not pathlib.Path(pbcopy).is_file():
        msg = f"{pbcopy} was not found, cannot write clipboard"
        raise NotImplementedError(msg)

    p = subprocess.Popen([pbcopy], stdin=subprocess.PIPE, universal_newlines=True)
    p.communicate(input=text)


# -----------------------------------------------------------------------------
# Copyright (C) 2015-2021, 2026 Angelos Evripiotis.
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
