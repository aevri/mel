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
            numpy.array([0, 0]),
            [(numpy.array([0, 0]), [1, 2])])

        self.assertEqual(0.0, error)
        self.assertTrue(([1.0, 2.0] == value).all(), True)
