"""Test suite for mel.cmddebug.benchrelate."""
# =============================================================================
#                                   TEST PLAN
# -----------------------------------------------------------------------------
# Here we detail the things we are concerned to test and specify which tests
# cover those concerns.
#
# Concerns:
# [ B] yield_reset_combinations() identity returns input
# [ B] yield_reset_combinations() iterates over replacements with resets=1
# [ B] yield_reset_combinations() replaces all with resets=2 (total moles is 2)
# [ B] yield_reset_combinations() replaces all with resets=3 (total moles is 2)
# -----------------------------------------------------------------------------
# Tests:
# [ A] test_A_Breathing
# [ B] test_B_yield_reset_combinations
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import mel.cmddebug.benchrelate


class Test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_A_Breathing(self):
        pass

    def test_B_yield_reset_combinations(self):
        moles = [
            {'uuid': '0'},
            {'uuid': '1'},
        ]
        theory = [('0', '0'), ('1', '1')]
        yield_combinations = mel.cmddebug.benchrelate.yield_reset_combinations

        # [ B] yield_reset_combinations() identity returns input
        result = list(yield_combinations(moles, moles, theory, 0))
        self.assertEqual(
            result,
            [(moles, moles, theory)])

        # [ B] yield_reset_combinations() iterates over replacements with
        # resets=1.
        result = list(yield_combinations(moles, moles, theory, 1))
        self.assertEqual(len(result), 2)
        assert_one_mapping_changed(self, *result[0])
        assert_one_mapping_changed(self, *result[1])

        # [ B] yield_reset_combinations() replaces all with resets=2 (total
        # moles is 2).
        result = list(yield_combinations(moles, moles, theory, 2))
        self.assertEqual(len(result), 1)
        assert_both_mappings_changed(self, *result[0])

        # [ B] yield_reset_combinations() replaces all with resets=3 (total
        # moles is 2).
        result = list(yield_combinations(moles, moles, theory, 3))
        self.assertEqual(len(result), 1)
        assert_both_mappings_changed(self, *result[0])


def assert_one_mapping_changed(test, from_moles, to_moles, theory):

    new_index = None
    new_uuid = None
    for i in range(2):
        this_uuid = to_moles[i]['uuid']
        if this_uuid != str(i):
            test.assertIsNone(new_index)
            new_index = i
            new_uuid = this_uuid

    for i in range(2):
        test.assertEqual(str(i), from_moles[i]['uuid'])
        if i != new_index:
            test.assertEqual(str(i), to_moles[i]['uuid'])
            test.assertEqual(str(i), theory[i][1])
        else:
            test.assertEqual(new_uuid, to_moles[i]['uuid'])
            test.assertEqual(new_uuid, theory[i][1])
        test.assertEqual(str(i), theory[i][0])


def assert_both_mappings_changed(test, from_moles, to_moles, theory):
    test.assertNotEqual(*to_moles)
    for i in range(2):
        test.assertEqual(str(i), from_moles[i]['uuid'])

        new_uuid = to_moles[i]['uuid']
        test.assertNotEqual(str(i), new_uuid)

        test.assertEqual(str(i), theory[i][0])
        test.assertEqual(new_uuid, theory[i][1])
