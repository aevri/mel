"""Test replacing parts of mel.rotomap.identify with pulp.

It turns out that it can be much slower than the branch-and-bound approach that
we've already written.

This seems to depend greatly on the number of candidate combinations that are
included, as controlled by the 'max_dist_sq' variable of the predictor
function.

Keep this file around to make it easier to re-evaluate that in the future.

"""

import copy
import sys

import numpy

import pulp

import mel.rotomap.identify
import mel.cmd.rotomapidentify
import mel.lib.kde


# This tutorial is pretty good for getting up to speed with this:
#
#   http://benalexkeen.com/linear-programming-with-python-and-pulp-part-6/
#


def main():
    # test_kde()
    # solve_mel(problem_9pos)
    solve(problem_9pos, program_pairs)


def test_kde():
    prior, current, predictor_positions = problem_2pos()
    lower = 1
    upper = -1

    predictor_uuid = 'A'
    target_uuid = 'B'
    prior_prow, prior_pcol = prior[predictor_uuid]
    prior_trow, prior_tcol = prior[target_uuid]
    prior_offset = (prior_trow - prior_prow, prior_tcol - prior_pcol)

    training_data = numpy.array(
        [numpy.array([prior_offset[1]]), numpy.array([prior_offset[0]])]
    )
    kde = mel.lib.kde.Kde(training_data, lower, upper)

    # predictor_pos = 'a'
    # target_pos = 'b'
    # current_prow, current_pcol = current[predictor_pos]
    # current_trow, current_tcol = current[target_pos]
    # current_offset = (
    #     current_trow - current_prow, current_tcol - current_pcol)

    print(
        kde(
            numpy.array(
                [prior_offset[1], prior_offset[0]]
            )
        )
    )


def solve_mel(scenario_fn):
    prior, current, predictor_positions = scenario_fn()

    uuid_to_frameposlist = {
        uuid_: [('prior', numpy.array(pos) * 0.1)]
        for uuid_, pos in prior.items()
    }

    box_radius = 0.4

    warm_classifier = mel.rotomap.identify.MoleRelativeClassifier(
        uuid_to_frameposlist, box_radius
    )

    possible_uuid_set = frozenset(uuid_to_frameposlist.keys())

    # uuid_to_pos = mel.rotomap.identify.frame_to_uuid_to_pos(frame)
    uuid_to_pos = {
        uuid_: numpy.array(pos) * 0.1
        for uuid_, pos in current.items()
    }

    canonical_uuid_set = frozenset()

    # Note that the order of adding UUIDs is important, in other places
    # we'll be relying on using '<' on ids to determine if an id refers to
    # a canonical mole.
    trans = mel.rotomap.identify.UuidToIndexTranslator()
    trans.add_uuids(canonical_uuid_set)
    num_canonicals = trans.num_uuids()
    trans.add_uuids_with_imposters(uuid_to_pos.keys())
    num_locations = trans.num_uuids() - trans.num_imposters()
    trans.add_uuids(possible_uuid_set)
    num_identities = trans.num_uuids()

    positions = trans.uuid_dict_to_index_tuple(uuid_to_pos, num_locations)

    calc_guesses = mel.rotomap.identify.make_calc_guesses(
        positions, trans, warm_classifier
    )
    predictors = mel.rotomap.identify.predictors(positions, num_canonicals)

    bounder = mel.cmd.rotomapidentify.BounderWrapper(
        tuple(predictor_loc for (_, predictor_loc) in predictors),
        calc_guesses,
        num_identities,
        num_canonicals,
    )

    print(f"Got {num_identities} identities.")
    print(f"Got {num_canonicals} canonicals.")

    guesser = mel.rotomap.identify.PosGuesser(
        num_locations, predictors, bounder, num_canonicals, num_identities
    )

    cost, old_to_new = mel.rotomap.identify.best_match_combination(
        guesser, max_iterations=1 * 10 ** 5
    )

    old_to_new = {
        trans.uuid_(old): trans.uuid_(new)
        for old, new in enumerate(old_to_new)
    }

    new_id_set = set(old_to_new.values())
    for uuid_ in current.keys():
        old_id = uuid_
        new_id = old_to_new[old_id]
        if new_id is not None:
            print(f'{old_id} -> {new_id}')


def solve(scenario_fn, problem_fn):
    positions, identities, predictor_positions, predict = make_problem(
        *scenario_fn())

    prob = pulp.LpProblem("Mel", pulp.LpMinimize)
    choices = pulp.LpVariable.dicts(
        "PosIdent",
        (positions, identities),
        0,
        1,
        pulp.LpInteger)

    problem_fn(
        prob, choices, positions, identities, predictor_positions, predict)

    # print()
    # print(prob)
    # print()

    print("Solving ...")
    # prob.solve()
    # prob.solve(pulp.solvers.PULP_CBC_CMD(fracGap=0.01))
    prob.solve(pulp.solvers.GLPK_CMD())

    print()
    print("Status:", pulp.LpStatus[prob.status])
    print()

    print()
    print("Cost:", pulp.value(prob.objective))
    print()

    # print()
    # print("Variables:")
    # for x in prob.variables():
    #     print(" ", pulp.value(x), x)
    # print()

    # print()
    # print("Contraints:")
    # for name, constraint in prob.constraints.items():
    #     print(" ", pulp.value(constraint), name, constraint)
    # print()

    for p in positions:
        for i in identities:
            val = pulp.value(choices[p][i])
            if val == 1:
                print(p, '->', i)


def problem_9pos():

    _ = None

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"
    I = "I"

    prior = grid_to_uuid_pos_dict((
        F, _, _, _, _, _, _, _, _, C,
        _, H, _, _, _, _, _, _, _, _,
        _, _, _, I, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, B, _, _, _,
        _, A, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, G, _, _, _, _, _, _, _, D,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, E,
    ))

    a = "a"
    b = "b"
    c = "c"
    d = "d"
    e = "e"
    f = "f"
    g = "g"
    h = "h"
    i = "i"

    current = grid_to_uuid_pos_dict((
        f, h, _, _, _, _, _, _, _, c,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, i, _, _, _, _, _,
        a, _, _, _, _, _, _, _, _, _,
        _, g, _, _, _, b, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, d,
        _, _, _, _, _, _, _, _, _, e,
    ))

    predictor_positions = {
        a: g,
        b: i,
        c: b,
        d: b,
        e: d,
        f: a,
        g: a,
        h: f,
        i: g,
    }

    return prior, current, predictor_positions


def problem_7pos():

    _ = None

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"

    prior = grid_to_uuid_pos_dict((
        F, _, _, _, _, _, _, _, _, C,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, B, _, _, _,
        _, A, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, G, _, _, _, _, _, _, _, D,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, E,
    ))

    a = "a"
    b = "b"
    c = "c"
    d = "d"
    e = "e"
    f = "f"
    g = "g"

    current = grid_to_uuid_pos_dict((
        f, _, _, _, _, _, _, _, _, c,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        a, _, _, _, _, _, _, _, _, _,
        _, g, _, _, _, b, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, d,
        _, _, _, _, _, _, _, _, _, e,
    ))

    predictor_positions = {
        a: g,
        b: a,
        c: b,
        d: b,
        e: d,
        f: a,
        g: a,
    }

    return prior, current, predictor_positions


def problem_5pos():

    _ = None

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"

    prior = grid_to_uuid_pos_dict((
        _, _, _, _, _, _, _, _, _, C,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, B, _, _, _,
        _, A, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, D,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, E,
    ))

    a = "a"
    b = "b"
    c = "c"
    d = "d"
    e = "e"

    current = grid_to_uuid_pos_dict((
        _, _, _, _, _, _, _, _, _, c,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        a, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, b, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, d,
        _, _, _, _, _, _, _, _, _, e,
    ))

    predictor_positions = {
        a: b,
        b: a,
        c: b,
        d: b,
        e: d,
    }

    return prior, current, predictor_positions


def problem_5pos_identical():

    _ = None

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"

    prior = grid_to_uuid_pos_dict((
        _, _, _, _, _, _, _, _, _, C,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, B, _, _, _,
        _, A, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, D,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, E,
    ))

    a = "a"
    b = "b"
    c = "c"
    d = "d"
    e = "e"

    current = grid_to_uuid_pos_dict((
        _, _, _, _, _, _, _, _, _, c,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, b, _, _, _,
        _, a, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, d,
        _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, e,
    ))

    predictor_positions = {
        a: b,
        b: a,
        c: b,
        d: b,
        e: d,
    }

    return prior, current, predictor_positions


def make_problem(prior, current, predictor_positions):

    positions = list(current.keys())

    known_identities = list(prior.keys())
    new_identities = [str(i) for i in range(len(current))]
    identities = known_identities + new_identities

    predict_fn = make_pair_predictor(new_identities, prior, current)

    for targ, pred in predictor_positions.items():
        pred_val = pred.upper()
        targ_val = targ.upper()
        print(
            f"{pred}={pred_val}, {targ}={targ_val}",
            predict_fn(pred, pred_val, targ, targ_val))

    return positions, identities, predictor_positions, predict_fn


def problem_3pos():

    __ = None
    _A = "a"
    _B = "b"
    _C = "c"

    prior = grid_to_uuid_pos_dict((
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, _B, __, __, __,
        __, _A, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, _C,
    ))

    _X = "x"
    _Y = "y"
    _Z = "z"
    current = grid_to_uuid_pos_dict((
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, _Y, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, _X, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, _Z, __,
    ))

    predictor_positions = {
        _X: _Y,
        _Y: _X,
        _Z: _Y,
    }

    return prior, current, predictor_positions


def problem_2pos():

    __ = None
    _A = "A"
    _B = "B"

    prior = grid_to_uuid_pos_dict((
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, _B, __, __, __,
        __, _A, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
    ))

    _a = "a"
    _b = "b"
    current = grid_to_uuid_pos_dict((
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, _b, __, __, __,
        __, __, _a, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
        __, __, __, __, __, __, __, __, __, __,
    ))


    predictor_positions = {
        _a: _b,
        _b: _a,
    }

    return prior, current, predictor_positions


def make_pair_predictor(new_identities, prior, current):

    def predict_fn(predictor_pos, predictor_uuid, target_pos, target_uuid):
        assert predictor_pos != target_pos

        # TODO
        if predictor_uuid in new_identities:
            result = 0
        elif target_uuid in new_identities:
            result = 0
        else:

            # Find the prior co-occurence and compute prob from distance.
            prior_prow, prior_pcol = prior[predictor_uuid]
            prior_trow, prior_tcol = prior[target_uuid]
            prior_offset = (prior_trow - prior_prow, prior_tcol - prior_pcol)
            current_prow, current_pcol = current[predictor_pos]
            current_trow, current_tcol = current[target_pos]
            current_offset = (
                current_trow - current_prow, current_tcol - current_pcol)

            offset_offset = (
                prior_offset[0] - current_offset[0],
                prior_offset[1] - current_offset[1],
            )

            x2 = offset_offset[0] ** 2
            y2 = offset_offset[1] ** 2

            # This number has a great effect on the running time, as many more
            # candidate mole combinations might be included.
            max_dist_sq = 25

            result = max(0, 1 - ((x2 + y2) / max_dist_sq))

            # print(offset_offset, x2, y2)
        # print(
        #     f"{predictor_pos}={predictor_uuid}, {target_pos}={target_uuid}",
        #     result
        # )
        return result

    return predict_fn


def program_pairs(
        prob, choices, positions, identities, predictor_positions, predict_fn):

    # prob += (0, "Arbitrary Objective Function")

    pair_costs = []
    pair_constraints = []

    # Costs of (position -> identity) choices.
    for p in positions:
        for i in identities:
            p2 = predictor_positions[p]
            for i2 in identities:
                if i == i2:
                    continue

                probability = predict_fn(p2, i2, p, i)
                cost = -probability
                if not cost:
                    continue

                pair_var = pulp.LpVariable(
                    f"Pair_{p}_{i}_by_{p2}_{i2}", 0, 1, pulp.LpInteger)

                # pair_var is 1 iff both parts of pair are active, else 0.
                pair_constraints.append(pair_var <= choices[p][i])
                pair_constraints.append(pair_var <= choices[p2][i2])
                pair_constraints.append(
                    pair_var >= choices[p][i] + choices[p2][i2] - 1)

                # print(cost)
                pair_costs.append(pair_var * cost)

    # prob += choices[_C][_A] * choices[_D][_B] * -predict(_C, _A, _D, _B), ""
    # prob += choices[_C][_A] * 1 * choices[_D][_B], ""
    # prob += 1 * choices[_D][_B], ""

    prob += pulp.lpSum(pair_costs), "Objective"

    for constraint in pair_constraints:
        prob += constraint

    # One identity per position.
    for p in positions:
        prob += pulp.lpSum([choices[p][i] for i in identities]) == (1, "")

    # At most one position per identity.
    for i in identities:
        prob += pulp.lpSum([choices[p][i] for p in positions]) <= (1, "")


def program_random(prob, choices, positions, identities, _1, _2):

    prob += (0, "Arbitrary Objective Function")

    # One identity per position.
    for p in positions:
        prob += pulp.lpSum([choices[p][i] for i in identities]) == (1, "")

    # At most one position per identity.
    for i in identities:
        prob += pulp.lpSum([choices[p][i] for p in positions]) <= (1, "")

    prob.writeLP("mel.lp")


def grid_to_uuid_pos_dict(grid):
    assert(len(grid) == 10 * 10)
    uuid_pos_dict = {}
    for row in range(10):
        for col in range(10):
            uuid_ = grid[row * 10 + col]
            if uuid_ is not None:
                uuid_pos_dict[uuid_] = (row, col)
    return uuid_pos_dict


if __name__ == "__main__":
    sys.exit(main())
