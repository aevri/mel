"""Test suite for mel.lib.svm."""
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
# [ B] test_B_Breathing
# =============================================================================

import unittest

import mel.lib.svm


class Test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_A_Breathing_Classifier(self):

        # Just ensure we can construct the things
        mel.lib.svm.Classifier()
        mel.lib.svm.Classifier(c=0.1, gamma=0.1)
        with self.assertRaises(ValueError):
            mel.lib.svm.Classifier(c=0.0, gamma=1.0)
        with self.assertRaises(ValueError):
            mel.lib.svm.Classifier(c=1.0, gamma=0.0)

        classifier = mel.lib.svm.Classifier()

        classifier.add_sample([1.0, 1.0], 0)
        classifier.add_sample([1.0, 1.0], 0)
        classifier.add_sample([0.0, 1.0], 2)
        classifier.add_sample([1.0, 1.0], 0)
        classifier.add_sample([0.0, 1.0], 2)
        classifier.add_sample([0.5, 0.5], 1)
        classifier.add_sample([0.5, 0.5], 1)
        classifier.add_sample([0.5, 0.5], 1)

        classifier.train()

        self.assertEqual(classifier.predict([1.0, 1.0]), 0)
        self.assertEqual(classifier.predict([0.0, 1.0]), 2)
        self.assertEqual(classifier.predict([0.5, 0.5]), 1)

    def test_B_Breathing_NamedClassifier(self):

        # Just ensure we can construct the things
        mel.lib.svm.NamedClassifier()
        mel.lib.svm.NamedClassifier(
            mel.lib.svm.Classifier())

        classifier = mel.lib.svm.NamedClassifier()

        classifier.add_sample([1.0, 1.0], "orange")
        classifier.add_sample([0.0, 1.0], "badger")
        classifier.add_sample([0.5, 0.5], "tea")

        classifier.train()

        self.assertEqual(classifier.predict([1.0, 1.0]), "orange")
        self.assertEqual(classifier.predict([0.0, 1.0]), "badger")
        self.assertEqual(classifier.predict([0.5, 0.5]), "tea")
# -----------------------------------------------------------------------------
# Copyright (C) 2017 Angelos Evripiotis.
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
