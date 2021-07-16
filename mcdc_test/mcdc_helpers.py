from functools import reduce


class Path(dict):
    def __init__(self, d):
        dict.__init__(self, d)
        self.origs = set()
        # TODO: This is silly, we just did `uniformize`!
        for (k, v) in d.items():
            if v is not None:
                self.origs.add(k)

    def size(self):
        assert len({k for k in self.origs if self[k] is None}) == 0
        return len(self.origs)


def better_size(tcs, pairs):
    # type: (dict, tuple) -> int
    # |(conditions(p0) ∪ conditions(p1))\conditions(tcs)|
    # The idea is that if the pair has more conditions that we might
    # still use in the future, then it's better.
    # We take the underlying paths, though, since an instantiated "?"
    # cannot help us make a useful path.

    def conditions(pi):
        # type: (Path) -> set
        return pi.origs

    def conditions_tcs(tcs):
        # type: (dict) -> set
        # returns (conditions(p0) ∪ conditions(p1) for all c in tcs[c] = (p0, p1)

        res = set()
        for p in tcs.values():
             res.union(conditions(p[0]) | conditions(p[1]))
        return res

    p0, p1 = pairs
    res = (conditions(p0) | conditions(p1)) - conditions_tcs(tcs)
    return len(res)


def better_size2(tcs, pairs):
    # type: (dict, tuple) -> int
    # |(conditions(p0) \ conditions(tcs)| + |(conditions(p1) \ conditions(tcs)|

    def conditions(pi):
        # type: (Path) -> set
        # returns all the conditions c where pi[c] != '?'
        return pi.origs

    def conditions_tcs(tcs):
        # type: (dict) -> set
        # returns (conditions(p0) ∪ conditions(p1) for all c in tcs[c] = (p0, p1)

        res = set()
        for p in tcs.values():
             res.union(conditions(p[0]) | conditions(p[1]))
        return res

    p0, p1 = pairs
    return len(conditions(p0) - conditions_tcs(tcs)) + (len(conditions(p1) - conditions_tcs(tcs)))


# def equal(path_1, path_2):
#     # type: (dict, dict) -> bool
#     # path_1 = {a: 0, b: None, c: 0}
#     # path_2 = {a: 1, b: None, c: None}
#
#     # Both path should have the same keys
#     # have_same_keys_assert(path_1, path_2)
#     # conds = set(path_1.keys()).union(path_2.keys())
#     conds = path_1.keys()
#     return path_1.keys() == path_2.keys() and all(path_1[c] == path_2[c] for c in conds)


def uniformize(path, conditions):
    # type: (dict, tuple) -> dict
    # path = {a: 0, c: 0}
    # conditions = (a, b, c)
    # return {a: 1, c: 0, b: None}]

    # Missing values are filled with 'None'
    # WARNING!: Conditions in the dictionary are not sorted alphabetically

    for k in conditions - path.keys():
        path.update({k: None})

    return path


def is_uniformized(path, conditions):
    # type: (dict, tuple) -> bool
    return type(path) == dict and path.keys() == set(conditions)


def lrlr(ordered_conditions, path):
    # type: (dict, dict) -> str
    """Prints a complete path as a binary string.
    The name is a pun on Left-Right-Left-Right...
    """
    res = ''
    for c in ordered_conditions:
        if path[c] is None:
            res += '?'
        else:
            assert path[c] in {0, 1}
            res += str(path[c])
    return res


def count_none(path):
    # type: (dict) -> int
    # return reduce((lambda acc, k: acc + (1 if path[k] is None else 0)), path, 0)
    # return sum(1 if pi is None else 0 for pi in path.values())
    return sum(pi is None for pi in path.values())


def has_none(path):
    # type: (dict) -> bool
    # reduce((lambda acc k: acc || tc[k] is None), tc, False)
    # return count_none(path) > 0
    return None in path.values()


def have_same_keys_assert(path_1, path_2):
    # type: (dict, dict) -> None
    # path_1 = {a: 0, b: None, c: 0}
    # path_2 = {a: 1, b: None, c: None}

    # path_1 and path_2 must have the same conditions (i.e., {a, b, c})
    assert path_1.keys() == path_2.keys(), \
        "path_0 and path_1 have different keys:\npath_0: {0}\npath_1: {1}\ndifference: {2}". \
            format(path_1.keys(), path_2.keys(),
                   {c for c in set(path_1.keys()).symmetric_difference(path_2.keys())})


def xor(path_1, path_2, condition):
    # type: (dict, dict, BDDVariable) -> bool
    # path_1 = {a: 0, b: 1, c: None}
    # path_2 = {a: 1, b: 1, c: None}
    # return True iff path_1[condition] ! = path_2[condition]

    # xor accepts that:
    # (path_1[condition] is not None) and
    # (path_2[condition] is None)

    have_same_keys_assert(path_1, path_2)

    # {c for c in set(path_1.keys()).union(path_2.keys())
    # if (c not in path_1.keys()) or (c not in path_2.keys())}

    conds = path_1.keys() - {condition}

    result = (path_1[condition] != path_2[condition]) and \
             all((path_1[c] == path_2[c]) or
                 (path_1[c] is None) or
                 (path_2[c] is None) for c in conds)
    return result


def merge_required(path_1, path_2):
    # type: (dict, dict) -> bool
    # path_1 = {a: 0, b: None, c: 0}
    # path_2 = {a: 1, b: None, c: None}
    # path_2 = {a: 1, b: 1, c: None}

    # output:
    # True iff any(((path_1[c] is None) and (path_2[c] is not None)) or
    #              ((path_1[c] is not None) and (path_2[c] is None))
    #              for c in conditions)

    have_same_keys_assert(path_1, path_2)
    conditions = path_1.keys()
    return any(((path_1[c] is None) and (path_2[c] is not None)) or
               ((path_1[c] is not None) and (path_2[c] is None))
               for c in conditions)


def merge(path_1, path_2):
    # type: (dict, dict) -> bool
    # destructively updates paths in place
    # path_1 = {a: 0, b: None, c: 0}
    # path_2 = {a: 1, b: None, c: None}
    # path_2 = {a: 1, b: 1, c: None}

    # output:
    # path_1 = {a: 0, b: 1, c: 0}
    # path_2 = {a: 1, b: 1, c: 0}

    have_same_keys_assert(path_1, path_2)
    # conditions = (a, b, c)
    conditions = path_1.keys()

    # TODO: Given that we're doing a loop anyway, we might want to avoid
    # looping twice and simply collect the m_r in the loop here.
    m_r = merge_required(path_1, path_2)

    sanity = 0  # see below
    for c in conditions:
        if (path_1[c] is None) and (path_2[c] is not None):
            path_1[c] = path_2[c]
        elif (path_1[c] is not None) and (path_2[c] is None):
            path_2[c] = path_1[c]
        elif (path_1[c] is not None) and (path_2[c] is not None):
            # Should never trigger -- would mean incompatible!
            if path_1[c] != path_2[c]:
                sanity = sanity + 1
                assert sanity <= 1, (found_diff_c, c)
                found_diff_c = c

    return m_r


# Merges two paths if permitted (= unification), None otherwise.
def merge_Maybe_except_c(c_excl, path_1, path_2):
    # type: (BDDVariable, dict, dict) -> dict
    # have_same_keys_assert(path_1,path_2)
    # print("merge:\t{0}".format(path_2))
    path_out = Path(path_1)
    path_out.origs = path_1.origs
    conditions = path_1.keys()
    for c in conditions:
        if c == c_excl:
            # Let's not look at this one.
            path_out[c] = path_1[c]
            continue
        if (path_1[c] is None) and (path_2[c] is not None):
            path_out[c] = path_2[c]
        elif (path_1[c] is not None) and (path_2[c] is None):
            path_out[c] = path_1[c]
        elif (path_1[c] is None) and (path_2[c] is None):
            path_out[c] = None
        elif path_1[c] != path_2[c]:
            return None
        else:
            assert path_1[c] == path_2[c]
            path_out[c] = path_1[c]
    return path_out


def merge_Maybe_except_c_bool(c_excl, path_1, path_2):
    # type: (BDDVariable, dict, dict) -> bool
    # Same as above, but doesn't construct anything.
    conditions = path_1.keys()
    for c in conditions:
        if c == c_excl:
            # Let's not look at this one.
            continue
        if (path_1[c] is None) and (path_2[c] is not None):
            continue
        elif (path_1[c] is not None) and (path_2[c] is None):
            continue
        elif (path_1[c] is None) and (path_2[c] is None):
            continue
        elif path_1[c] != path_2[c]:
            return None
        else:
            assert path_1[c] == path_2[c]
            continue
    return True


def merge_Maybe(path_1, path_2):
    return merge_Maybe_except_c(None, path_1, path_2)


def unique_tests(test_case):
    # type: (dict) -> (list)

    # test_case[a] = ({a: 0, b: 0, c: 0}, {a: 1, b: 1, c: 0})
    # test_case[b] = ({a: 1, b: 0, c: 0}, {a: 1, b: 1, c: 0})
    # test_case[c] = ({a: 0, b: 0, c: 0}, {a: 0, b: 0, c: 1})

    # return unique tests [{a: 0, b: 0, c: 0}, {a: 1, b: 1, c: 0},...]

    keys = sorted(test_case.keys())
    result = set()
    for (p0, p1) in test_case.values():
        # p0_tuple = tuple(p0.values())
        # p1_tuple = tuple(p1.values())
        # Iterate over all conditions c_i in alphabetical order (i.e., abc).
        # Conditions in the dictionary are not sorted alphabetically.
        p0_tuple = tuple(p0[c] for c in sorted(p0))
        p1_tuple = tuple(p1[c] for c in sorted(p1))

        result.add(p0_tuple)
        result.add(p1_tuple)

    return [dict(zip(keys, r)) for r in result]


def replace_final_question_marks(test_case):
    # type: (dict) -> bool
    # Arbitrarily replaces remaining question marks with 1.
    # The result tells you if at least one ? had to be replaced.
    conditions = test_case.keys()
    have_warned = False
    for cond in test_case:
        path_zero, path_one = test_case[cond]
        for k in path_zero.keys():
            if path_zero[k] is None:
                if not have_warned:
                    have_warned = True
                path_zero[k] = 1
        for k in path_one.keys():
            if path_one[k] is None:
                if not have_warned:
                    have_warned = True
                path_one[k] = 1
        test_case[cond] = (path_zero, path_one)
    return have_warned


def stabilize(test_case):
    # type: (dict) -> None

    # test_case[a] = ({a: 0, b: None, c: 0}, {a: 1, b: 1, c: None})
    # test_case[b] = ({a: 1, b: 0, c: 0}, {a: 1, b: 1, c: None})
    # test_case[c] = ({a: 0, b: None, c: 0}, {a: 0, b: None, c: 1})

    while any(merge_required(path_zero, path_one) for path_zero, path_one in test_case.values()):
        #       p0  |   p1
        # ----------------------
        # [a0 b0 c0 | !a0 b0 c0]
        # [a1 b1 c1 | a1 !b1 c1]
        # [a2 b2 c2 | a2 b2 !c2]

        for cond in test_case:
            path_zero, path_one = test_case[cond]
            # path_zero = {a: 0, b: None, c: 0}
            # path_one = {a: 1, b: 1, c: None}

            # Replace '?' in path_zero by the value in path_one for the same condition c, and viceversa
            merge(path_zero, path_one)

        # Q: what does it happen if path_0[j] == path_1[j] == "?"?
        # path_zero = {a: 0, b: None, c: 0}
        # path_one = {a: 1, b: None, c: 1}

        # A: Ideally, all '?' should be instantiated when propagate(test_case)
        # TODO: Why is a comment about `propagate` here in `stabilize`?!


def propagate(test_case):
    # type: (dict) -> dict

    # test_case[a] = ({a: 0, b: None, c: 0}, {a: 1, b: None, c: 1})
    # test_case[b] = ({a: 1, b: 0, c: None}, {a: 1, b: 1, c: None})
    # test_case[c] = ({a: 0, b: None, c: 0}, {a: 0, b: None, c: 1})

    # By construction, all the variables in the diagonal are distinct of None
    #       p0  |   p1
    # ----------------------
    # [a0 b0 c0 | !a0 b0 c0]
    # [a1 b1 c1 | a1 !b1 c1]
    # [a2 b2 c2 | a2 b2 !c2]

    # TODO: How do we propagate a value in the diagonal to other rows of the table?
    #  For instance, selecting the rows with higher number of '?' so that these gaps are instantiated firstly.

    # We have already partially instantiated p0, so we cannot use reuse[path] for sorting the set PSI

    # psi_0 = [p0 for p0, _ in test_case.values()]
    # psi_0 = SortedList((p0 for p0, _ in test_case.values()), key=lambda path: reuse[path])
    # p0 = psi_0[0]
    # for pi in psi_0[1:]:
    #     merge(p0, pi)

    # Naive solution: Given a TC with ?, find the TC that it matches best with:
    #  -> Reuse will increase/ number of TCs will stay constant.
    # VS is not sure if you can do this iteratively (for 2x?, you could still have 1 left after merge).
    # ...??... vs ...1?... and ...?1... -- can one be the wrong choice?

    # local helper:
    def pick_best(acc, path):
        # type: (int, dict) -> (int, dict)
        c = count_none(path)
        if c < acc[0]:
            return (c, path)
        else:
            return acc

    # TODO: misleading name, it's Test Case*S*!
    tc_out = dict()
    conditions = test_case.keys()
    for cond in test_case:
        path_zero, path_one = test_case[cond]
        # path_zero = {a: 0, b: None, c: 0}
        # path_one = {a: 1, b: 1, c: None}
        tc_out[cond] = (path_zero, path_one)
        if has_none(path_zero):
            # calculate merge with every other (fitting) path, count remaining "?". Pick lowest.
            # TODO(?): We don't care for now that we're merging with ourselves.
            newTCs = {key: merge_Maybe(path_zero, value[0]) for
                      (key, value) in test_case.items()}
            _, bestMatch = reduce(lambda acc, x: acc if (newTCs[x] is None) else pick_best(acc, newTCs[x]),
                                  newTCs, (len(conditions), None))
            if bestMatch is not None:
                assert count_none(bestMatch) <= count_none(path_zero)
                # TODO: This is just sloppy, we want to make a copy but are updating in place.
                merge(bestMatch, path_one)
                tc_out[cond] = (bestMatch, path_one)

        # Now do the same for paths to True:
        if has_none(path_one):
            # calculate merge with every other (fitting) path, count remaining "?". Pick lowest.
            newTCs = {key: merge_Maybe(path_one, value[1]) for
                      (key, value) in test_case.items()}
            _, bestMatch = reduce(lambda acc, x: acc if (newTCs[x] is None) else pick_best(acc, newTCs[x]),
                                  newTCs, (len(conditions), None))
            if bestMatch is not None:
                assert count_none(bestMatch) <= count_none(path_one)
                # TODO: This is just sloppy, we want to make a copy but are updating in place.
                merge(bestMatch, tc_out[cond][0])
                tc_out[cond] = (tc_out[cond][0], bestMatch)  # Ugh

        assert count_none(tc_out[cond][0]) <= count_none(path_zero)
        assert count_none(tc_out[cond][1]) <= count_none(path_one)
    return tc_out


def size(path):
    # type: (dict) -> int
    # Counts positions that are not None. Hence doesn't need uniform paths.
    return sum(0 if i is None else 1 for i in path.values())


def negate(bit):
    # type: (int) -> int
    # Negate the current bit using (+ 1 % 2)
    return (bit + 1) % 2


def instantiate(test_case):
    # type: (dict) -> dict

    # test_case[a] = ({a: 0, b: None, c: 0}, {a: 1, b: 1, c: None})
    # test_case[b] = ({a: 1, b: 0, c: 0}, {a: 1, b: 1, c: None})
    # test_case[c] = ({a: 0, b: None, c: 0}, {a: 0, b: None, c: 1})

    # Abstractly,
    #       p0  |   p1
    # ----------------------
    # [a0 b0 c0 | !a0 b0 c0]
    # [a1 b1 c1 | a1 !b1 c1]
    # [a2 b2 c2 | a2 b2 !c2]

    # Synchronize p0 and p1 for each row, so that they only differ in one position (e.g., a0 vs !a0)
    stabilize(test_case)
    # Propagate values in the diagonal to instantiate "?"s across test cases
    # num_conditions = len(test_case.keys())
    # for i in range(num_conditions):
    #     test_case = propagate(test_case)
    #     # TODO: is only once at the end enough?
    #     stabilize(test_case)

    new_test_case = propagate(test_case)
    stabilize(new_test_case)
    while new_test_case != test_case:
        test_case = new_test_case
        new_test_case = propagate(test_case)
        # TODO: is only once at the end enough?
        stabilize(new_test_case)

    return test_case
