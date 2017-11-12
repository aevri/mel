# Unit tests for mel.rotomap.identify

import unittest

import mel.rotomap.identify


class BounderTestCase(unittest.TestCase):

    def test_breathing(self):

        def pos_guess(uuid_for_history, uuid_for_position, a):
            return tuple()

        def pos_guess_dict(uuid_for_history, uuid_for_position, a):
            return dict()

        def closest_uuids(a):
            raise NotImplementedError()

        possible_uuid_set = set()
        canonical_uuid_set = set()

        bounder = mel.rotomap.identify.Bounder(
            pos_guess,
            pos_guess_dict,
            closest_uuids,
            possible_uuid_set,
            canonical_uuid_set)

    def test_known_known(self):

        def pos_guess(uuid_for_history, uuid_for_position, a):
            self.assertEqual(uuid_for_history, 'canonical')
            self.assertEqual(uuid_for_position, 'canonical')
            self.assertEqual(a, 'a')
            return [('correct', 1)]

        def pos_guess_dict(uuid_for_history, uuid_for_position, a):
            return dict(pos_guess(uuid_for_history, uuid_for_position, a))

        def closest_uuids(a):
            return ['canonical']

        canonical_uuid_set = {'canonical'}
        possible_uuid_set = {'correct', 'canonical'}

        bounder = mel.rotomap.identify.Bounder(
            pos_guess,
            pos_guess_dict,
            closest_uuids,
            possible_uuid_set,
            canonical_uuid_set)

        self.assertEqual(
            bounder.lower_bound({'canonical': 'canonical', 'a': 'correct'}),
            1)

    def test_known_history_unknown_mole(self):

        def pos_guess(uuid_for_history, uuid_for_position, a):
            self.assertEqual(uuid_for_history, 'canonical')
            self.assertEqual(uuid_for_position, 'canonical')
            self.assertEqual(a, 'a')
            return [('correct', 1)]

        def pos_guess_dict(uuid_for_history, uuid_for_position, a):
            return dict(pos_guess(uuid_for_history, uuid_for_position, a))

        def closest_uuids(a):
            return ['canonical']

        canonical_uuid_set = {'canonical'}
        possible_uuid_set = {'correct', 'canonical'}

        bounder = mel.rotomap.identify.Bounder(
            pos_guess,
            pos_guess_dict,
            closest_uuids,
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


def make_bounder(
        closest, guesses, non_canonical_uuid_set, canonical_uuid_set):

    possible_uuid_set = non_canonical_uuid_set | canonical_uuid_set

    def pos_guess(uuid_for_history, uuid_for_position, a):
        return guesses[(uuid_for_history, uuid_for_position, a)]

    def pos_guess_dict(uuid_for_history, uuid_for_position, a):
        return dict(pos_guess(uuid_for_history, uuid_for_position, a))

    def closest_uuids(a):
        return closest[a]

    return mel.rotomap.identify.Bounder(
        pos_guess,
        pos_guess_dict,
        closest_uuids,
        possible_uuid_set,
        canonical_uuid_set)
