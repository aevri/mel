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
