# Unit tests for mel.rotomap.identify

import unittest

import mel.rotomap.identify


class BounderTestCase(unittest.TestCase):

    def test_breathing(self):

        class BreathingMockHelper():

            def pos_guess(self, uuid_for_history, uuid_for_position, a):
                return tuple()

            def pos_guess_dict(self, uuid_for_history, uuid_for_position, a):
                return dict()

            def closest_uuids(self, a):
                raise NotImplementedError()

        possible_uuid_set = set()
        canonical_uuid_set = set()

        bounder = mel.rotomap.identify.Bounder(
            BreathingMockHelper(),
            possible_uuid_set,
            canonical_uuid_set)

    def test_known_known(self):

        test_self = self

        class KnownKnownMockHelper():

            def pos_guess(self, uuid_for_history, uuid_for_position, a):
                test_self.assertEqual(uuid_for_history, 'canonical')
                test_self.assertEqual(uuid_for_position, 'canonical')
                test_self.assertEqual(a, 'a')
                return [('correct', 1)]

            def pos_guess_dict(self, uuid_for_history, uuid_for_position, a):
                return dict(
                    self.pos_guess(
                        uuid_for_history, uuid_for_position, a))

            def closest_uuids(self, a):
                return ['canonical']

        canonical_uuid_set = {'canonical'}
        possible_uuid_set = {'correct', 'canonical'}

        bounder = mel.rotomap.identify.Bounder(
            KnownKnownMockHelper(),
            possible_uuid_set,
            canonical_uuid_set)

        self.assertEqual(
            bounder.lower_bound({'canonical': 'canonical', 'a': 'correct'}),
            1)

    def test_known_history_unknown_mole(self):

        test_self = self

        class KnownHistoryUnknownMoleMockHelper():

            def pos_guess(self, uuid_for_history, uuid_for_position, a):
                test_self.assertEqual(uuid_for_history, 'canonical')
                test_self.assertEqual(uuid_for_position, 'canonical')
                test_self.assertEqual(a, 'a')
                return [('correct', 1)]

            def pos_guess_dict(self, uuid_for_history, uuid_for_position, a):
                return dict(pos_guess(uuid_for_history, uuid_for_position, a))

            def closest_uuids(self, a):
                return ['canonical']

        canonical_uuid_set = {'canonical'}
        possible_uuid_set = {'correct', 'canonical'}

        bounder = mel.rotomap.identify.Bounder(
            KnownHistoryUnknownMoleMockHelper(),
            possible_uuid_set,
            canonical_uuid_set)

        self.assertEqual(
            bounder.lower_bound({'canonical': 'canonical', 'a': None}),
            1)

    def test_known_history_unknown_mole2(self):

        closest = {
            'pos1': ('sure', 'pos2'),
            'pos2': ('sure', 'pos1')
        }

        guesses = {
            ('sure', 'sure', 'pos1'): [('id1', 2), ('id2', 10)],
            ('sure', 'sure', 'pos2'): [('id2', 3), ('id1', 20)],
        }

        canonical_uuid_set = {'sure'}
        non_canonical_uuid_set = {'id1', 'id2'}

        bounder = make_bounder(
            closest,
            guesses,
            non_canonical_uuid_set,
            canonical_uuid_set)

        self.assertEqual(
            bounder.lower_bound({
                'sure': 'sure',
                'pos1': None,
                'pos2': None
            }),
            6)

        self.assertEqual(
            bounder.lower_bound({
                'sure': 'sure',
                'pos1': 'id1',
                'pos2': None
            }),
            6)

        self.assertEqual(
            bounder.lower_bound({
                'sure': 'sure',
                'pos1': None,
                'pos2': 'id2'
            }),
            6)

        self.assertEqual(
            bounder.lower_bound({
                'sure': 'sure',
                'pos1': 'id1',
                'pos2': 'id2'
            }),
            6)

        self.assertEqual(
            bounder.lower_bound({
                'sure': 'sure',
                'pos1': 'id2',
                'pos2': None
            }),
            200)

        self.assertEqual(
            bounder.lower_bound({
                'sure': 'sure',
                'pos1': None,
                'pos2': 'id1'
            }),
            200)

        self.assertEqual(
            bounder.lower_bound({
                'sure': 'sure',
                'pos1': 'id2',
                'pos2': 'id1'
            }),
            200)

    def test_unknown_history_unknown_mole2(self):

        closest = {
            'pos1': ['pos2'],
            'pos2': ['pos1']
        }

        guesses = {
            ('id1', 'pos1', 'pos2'): [('id2', 3)],
            ('id1', 'pos2', 'pos1'): [('id2', 20)],
            ('id2', 'pos2', 'pos1'): [('id1', 2)],
            ('id2', 'pos1', 'pos2'): [('id1', 10)],
        }

        canonical_uuid_set = set()
        non_canonical_uuid_set = {'id1', 'id2'}

        bounder = make_bounder(
            closest,
            guesses,
            non_canonical_uuid_set,
            canonical_uuid_set)

        self.assertEqual(
            bounder.lower_bound({
                'pos1': None,
                'pos2': None
            }),
            6)

        self.assertEqual(
            bounder.lower_bound({
                'pos1': 'id1',
                'pos2': None
            }),
            6)

        self.assertEqual(
            bounder.lower_bound({
                'pos1': None,
                'pos2': 'id2'
            }),
            6)

        self.assertEqual(
            bounder.lower_bound({
                'pos1': 'id1',
                'pos2': 'id2'
            }),
            6)

        self.assertEqual(
            bounder.lower_bound({
                'pos1': 'id2',
                'pos2': None
            }),
            200)

        self.assertEqual(
            bounder.lower_bound({
                'pos1': None,
                'pos2': 'id1'
            }),
            200)

        self.assertEqual(
            bounder.lower_bound({
                'pos1': 'id2',
                'pos2': 'id1'
            }),
            200)

    def test_unknown_history_unknown_mole3(self):

        closest = {
            'pos1': ['pos3', 'pos2'],
            'pos2': ['pos3', 'pos1'],
            'pos3': ['pos1', 'pos2'],
        }

        guesses = {
            ('id1', 'pos1', 'pos2'): [('id2', 20), ('id3', 300)],
            ('id1', 'pos1', 'pos3'): [('id2', 200), ('id3', 30)],
            ('id2', 'pos2', 'pos1'): [('id1', 10), ('id3', 300)],
            ('id2', 'pos2', 'pos3'): [('id1', 100), ('id3', 30)],
            ('id3', 'pos3', 'pos1'): [('id1', 10), ('id2', 200)],
            ('id3', 'pos3', 'pos2'): [('id1', 100), ('id2', 20)],

            ('id1', 'pos2', 'pos1'): [('id2', 200), ('id3', 300)],
            ('id1', 'pos2', 'pos3'): [('id2', 200), ('id3', 300)],
            ('id1', 'pos3', 'pos1'): [('id2', 200), ('id3', 300)],
            ('id1', 'pos3', 'pos2'): [('id2', 200), ('id3', 300)],

            ('id2', 'pos1', 'pos2'): [('id1', 100), ('id3', 300)],
            ('id2', 'pos1', 'pos3'): [('id1', 100), ('id3', 300)],
            ('id2', 'pos3', 'pos1'): [('id1', 100), ('id3', 300)],
            ('id2', 'pos3', 'pos2'): [('id1', 100), ('id3', 300)],

            ('id3', 'pos1', 'pos2'): [('id1', 100), ('id2', 200)],
            ('id3', 'pos1', 'pos3'): [('id1', 100), ('id2', 200)],
            ('id3', 'pos2', 'pos1'): [('id1', 100), ('id2', 200)],
            ('id3', 'pos2', 'pos3'): [('id1', 100), ('id2', 200)],
        }

        canonical_uuid_set = set()
        non_canonical_uuid_set = {'id1', 'id2', 'id3'}

        bounder = make_bounder(
            closest,
            guesses,
            non_canonical_uuid_set,
            canonical_uuid_set)

        def make_input(i1, i2, i3):
            names = [None, 'id1', 'id2', 'id3']
            return {
                'pos1': names[i1],
                'pos2': names[i2],
                'pos3': names[i3],
            }

        input_output = [
            ((0, 0, 0), 10 * 20 * 30),
            ((1, 0, 0), 10 * 20 * 30),
            ((0, 2, 0), 10 * 20 * 30),
            ((0, 0, 3), 10 * 20 * 30),
            ((1, 2, 0), 10 * 20 * 30),
            ((1, 0, 3), 10 * 20 * 30),
            ((0, 2, 3), 10 * 20 * 30),
            ((1, 2, 3), 10 * 20 * 30),

            ((2, 0, 0), 200 * 100 * 100),
            ((0, 1, 0), 200 * 100 * 200),
            ((2, 1, 0), 200 * 100 * 300),

            ((2, 1, 3), 200 * 100 * 300),
            ((3, 2, 1), 300 * 200 * 100),
        ]

        for input_, output in input_output:
            self.assertEqual(
                bounder.lower_bound(
                    make_input(*input_)),
                output)


class PosGuesserTestCase(unittest.TestCase):

    def test_breathing(self):

        pos_uuids = tuple()

        possible_uuid_set = set()
        canonical_uuid_set = set()

        guesser = mel.rotomap.identify.PosGuesser(
            pos_uuids,
            MockHelper({}, {}),
            canonical_uuid_set,
            possible_uuid_set)


class MockHelper():

    def __init__(self, closest, guesses):
        self.closest = closest
        self.guesses = guesses

    def pos_guess(self, uuid_for_history, uuid_for_position, a):
        g = self.guesses[(uuid_for_history, uuid_for_position, a)]
        return tuple(sorted(g, key=lambda x: x[1]))

    def pos_guess_dict(self, uuid_for_history, uuid_for_position, a):
        return dict(self.pos_guess(uuid_for_history, uuid_for_position, a))

    def closest_uuids(self, a):
        return self.closest[a]


def make_bounder(closest, guesses, non_canonical_uuid_set, canonical_uuid_set):

    possible_uuid_set = non_canonical_uuid_set | canonical_uuid_set
    helper = MockHelper(closest, guesses)
    return mel.rotomap.identify.Bounder(
        helper,
        possible_uuid_set,
        canonical_uuid_set)
