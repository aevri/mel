"""Test suite for mel.lib.image."""
# =============================================================================
#                                   TEST PLAN
# -----------------------------------------------------------------------------
# Here we detail the things we are concerned to test and specify which tests
# cover those concerns.
#
# Concerns:
# [  ]
# -----------------------------------------------------------------------------
# Tests:
# [ A] test_a_breathing
# =============================================================================


import unittest

import mel.lib.common
import mel.lib.image


class Test(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_a_breathing(self):
        # Exercise the methods to make sure there are no unexpected exceptions
        image1 = mel.lib.common.new_image(200, 100)
        image2 = mel.lib.common.new_image(100, 200)
        montage = mel.lib.image.montage_horizontal(50, image1, image2)
        letterbox_montage = mel.lib.image.letterbox(montage, 200, 200)
        mel.lib.image.montage_vertical(50, montage, letterbox_montage)


# -----------------------------------------------------------------------------
# Copyright (C) 2015-2017 Angelos Evripiotis.
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
