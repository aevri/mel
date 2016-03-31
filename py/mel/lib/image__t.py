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
# [ A] test_A_Breathing
# =============================================================================


import unittest

import mel.lib.common
import mel.lib.image


class Test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_A_Breathing(self):
        # Exercise the methods to make sure there are no unexpected exceptions
        image1 = mel.lib.common.new_image(200, 100)
        image2 = mel.lib.common.new_image(100, 200)
        h = mel.lib.image.montage_horizontal(50, image1, image2)
        hh = mel.lib.image.letterbox(h, 200, 200)
        mel.lib.image.montage_vertical(50, h, hh)
