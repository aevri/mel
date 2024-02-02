"""Test suite for mel.lib.datetime."""

# =============================================================================
#                                   TEST PLAN
# -----------------------------------------------------------------------------
# Here we detail the things we are concerned to test and specify which tests
# cover those concerns.
#
# Concerns:
# [ A] make_now_datetime_string() returns a string we can convert to datetime
# -----------------------------------------------------------------------------
# Tests:
# [ A] test_a_breathing
# =============================================================================


import unittest

import mel.lib.datetime


class Test(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_a_breathing(self):
        # [] make_now_datetime_string() returns a string we can convert to
        # datetime
        datetimestring = mel.lib.datetime.make_now_datetime_string()
        datetime_ = mel.lib.datetime.guess_datetime_from_string(datetimestring)
        assert datetime_ is not None


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
