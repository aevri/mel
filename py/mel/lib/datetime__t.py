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
# [ A] test_A_Breathing
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import mel.lib.datetime


class Test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_A_Breathing(self):
        # [] make_now_datetime_string() returns a string we can convert to
        # datetime
        datetimestring = mel.lib.datetime.make_now_datetime_string()
        datetime_ = mel.lib.datetime.guess_datetime_from_string(datetimestring)
        assert datetime_ is not None
