"""Test suite for mel.rotomap.tricolour."""


import unittest

import mel.rotomap.tricolour


class Test(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_a_breathing(self):
        pass

    def test_b_rotate_bounds(self):
        self.assertListEqual(
            mel.rotomap.tricolour._list_rotated_left([1, 2, 3], 0),
            [1, 2, 3])
        self.assertListEqual(
            mel.rotomap.tricolour._list_rotated_left([1, 2, 3], 1),
            [2, 3, 1])
        self.assertListEqual(
            mel.rotomap.tricolour._list_rotated_left([1, 2, 3], 2),
            [3, 1, 2])
        self.assertListEqual(
            mel.rotomap.tricolour._list_rotated_left([1, 2, 3], 3),
            [1, 2, 3])
        with self.assertRaises(ValueError):
            mel.rotomap.tricolour._list_rotated_left([1, 2, 3], -1)
        with self.assertRaises(ValueError):
            mel.rotomap.tricolour._list_rotated_left([1, 2, 3], 4)

    def test_c_yield_triband_mapping(self):
        num_colours = 9
        mapping = list(
            mel.rotomap.tricolour.yield_triband_mapping_in_distinctive_order(
                num_colours
            )
        )

        self.assertEqual(len(mapping), num_colours ** 3)

        mapping_set = set(mapping)
        self.assertEqual(len(mapping_set), len(mapping))


# -----------------------------------------------------------------------------
# Copyright (C) 2016-2017 Angelos Evripiotis.
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
