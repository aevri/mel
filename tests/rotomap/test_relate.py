"""Test suite for mel.rotomap.relate."""
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

import numpy

import mel.rotomap.relate


class Test(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_a_breathing(self):
        pass

    def test_b_pick_value_from_field(self):

        value, error = mel.rotomap.relate.pick_value_from_field(
            numpy.array([0, 0]), [(numpy.array([0, 0]), [1, 2])]
        )

        self.assertEqual(0.0, error)
        self.assertTrue((value == [1.0, 2.0]).all(), True)


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
