import collections
import csv
import itertools
import logging
import sys
import functools
from functools import reduce
from itertools import chain, product, repeat, accumulate
from random import randint, seed

from graphviz import Source
from matplotlib import pyplot as plt
from sortedcontainers import SortedList

import tcasii
from vsplot import plot
from mcdctestgen import run_experiment, calc_reuse, calc_may_reuse
from pyeda.boolalg.bdd import _path2point, BDDNODEZERO, BDDNODEONE, BDDZERO, BDDONE
from mcdc_helpers import uniformize, merge, instantiate, unique_tests, size, merge_Maybe_except_c, negate, \
    is_uniformized, lrlr, xor

logger = None  # lazy


def bfs_upto_c(f, c):
    """Iterate through nodes in BFS order."""
    queue = collections.deque()
    queue.append(([f.node], f.node))
    while queue:
        (path, node) = queue.pop()
        # Safety-belt:
        assert path[-1].root <= node.root, (path[-1].root, node.root)
        # Assume we shot past it:
        if node.root > c.uniqid:
            continue
        if node.root == c.uniqid:
            # `path` ends with `node`
            yield path, node
        else:
            if node.lo is not None:
                queue.appendleft((path+[node.lo], node.lo))
            if node.hi is not None:
                queue.appendleft((path+[node.hi], node.hi))


def memoized_generator(f):
    # https://stackoverflow.com/a/53437323/60462
    cache = {}

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        k = args, frozenset(kwargs.items())
        it = cache[k] if k in cache else f(*args, **kwargs)
        cache[k], result = itertools.tee(it)
        return result
    return wrapper


# @memoized_generator
def terminals_via_bfs_from(node):
    # Pushed down `terms`, since it was constant anyway and couldn't be @cached.
    terms = set([BDDNODEZERO, BDDNODEONE])
    assert len(terms) >= 1 or len(terms) <= 2
    queue = collections.deque()
    queue.append((([], []), node))
    # Note that we're a bit inconsistent here on purpose:
    # We start with [], and we know we'll never yield on the first seed.
    # All other paths will actually contain `node` at the end,
    # with the trailing terminal.
    while queue:
        (path, trace), node = queue.pop()
        if node in terms:
            yield (path + [node], trace), node
        else:
            # Methinks the path is enough, don't need `node.root`?
            if node.lo is not None:
                queue.appendleft(((path+[node], trace+[(node.root, False)]), node.lo))
            if node.hi is not None:
                queue.appendleft(((path+[node], trace+[(node.root, True)]), node.hi))


# @memoized_generator
def pairs_from_node(f, v_c):
    def _check_monotone(acc, t):
        lt = len(t[0][0])
        res = lt >= acc[0]
        assert res
        return lt, t

    ts = terminals_via_bfs_from(v_c)
    if sys.version_info >= (3, 8):
        ts_checked = accumulate(ts, _check_monotone, initial=(-1, None))
        ts_checked.__next__()
    else:
        ts_checked = zip(repeat(None), ts)
    # let's take one path and try find its independence partner:
    # TODO: will EACH have an i-partner, one, or some?
    for _, t in ts_checked:
        for i_p, i_n in independence_day_for_condition(f, v_c, t):
            yield t[0][0], (i_p, i_n)


def find_partner_from_following(f, node, terminal, path, suffix, seen_nodes_on_other_path):
    # TODO: is this a function or a relation?
    # `path` may or may not start at the root.
    # No point in memoizing, though, since `seen_nodes` will be unique
    # (although possibly overlapping) between runs starting from the same `node`.
    if node in seen_nodes_on_other_path:
        return
    suffix_rest = suffix
    if node in set([BDDNODEZERO, BDDNODEONE]):
        if node is terminal:
            yield path, node
            return
        else:
            # print('Looked for {}, got {}.'.format(ttff(terminal), ttff(node)))
            return  # Or what?
    else:
        if len(suffix) == 0:
            # TODO: Can eliminate, should be equivalent to running through an empty for-loop!
            yield from find_partner_from_following(f, node.lo, terminal, path + [node.lo], [], seen_nodes_on_other_path)
            yield from find_partner_from_following(f, node.hi, terminal, path + [node.hi], [], seen_nodes_on_other_path)
            return
        else:
            for (c, s) in suffix:
                if node in seen_nodes_on_other_path:
                    return
                if node in set([BDDNODEZERO, BDDNODEONE]):
                    if node is terminal:
                        yield path, node
                        return
                    else:
                        # print('Looked for {}, got {}.'.format(ttff(terminal), ttff(node)))
                        return  # Or what?
                else:
                    if node.root == c:
                        # Consume
                        _, *suffix_rest = suffix_rest  # tail
                        if s:  # T = right
                            node = node.hi
                        else:
                            node = node.lo
                        path = path + [node]
                    else:
                        # We don't know if the ? is in suffix or our path, so...
                        # 1) If the paths had ?, we'll catch up during the search since
                        #       `suffix_rest` is still the "longer" version
                        # 2) If the we've a ? here in the partner-path, we may have to discard
                        #       some elements from `suffix_rest` to catch up
                        # Q: can we distinguish both cases? If we'd know how to check the order...
                        # TODO: BFS or what?!
                        if node.root > c:
                            continue  # we'll just keep stripping
                        else:
                            assert node.root < c
                            # Restart search from children with currently remaining suffix
                            yield from find_partner_from_following(f, node.lo, terminal, path + [node.lo], suffix_rest, seen_nodes_on_other_path)
                            yield from find_partner_from_following(f, node.hi, terminal, path + [node.hi], suffix_rest, seen_nodes_on_other_path)
                        return
            # assert node in set([BDDNODEZERO, BDDNODEONE]), uniformize(_path2point(path), f.inputs)
            if node is terminal:
                yield path, node
            else:
                if node not in set([BDDNODEZERO, BDDNODEONE]):
                    # suffix was shorter, so bfs for the right terminal
                    # TODO: that's not BFS yet ;-)
                    yield from find_partner_from_following(f, node.lo, terminal, path + [node.lo], [], seen_nodes_on_other_path)
                    yield from find_partner_from_following(f, node.hi, terminal, path + [node.hi], [], seen_nodes_on_other_path)


def ttff(node):
    if node is BDDNODEONE:
        return 1
    elif node is BDDNODEZERO:
        return 0
    else:
        assert False


def independence_day_for_condition(f, v_c, t):
    # Can't memoize, since `t#atoe` is always used exactly once only.
    ((atoe, suffix), terminal) = t
    if terminal is BDDNODEONE:
        opposite = BDDNODEZERO
    else:
        assert terminal is BDDNODEZERO
        opposite = BDDNODEONE
    # Not general enough:
    # assert c_s is not 'c' or (terminal is not BDDNODEZERO or len(suffix_l) == 2)
    # Flip start of suffix:
    (cur_c, cur_s), *suffix_rest = suffix
    if cur_s:
        v_c = v_c.lo
    else:
        v_c = v_c.hi
    partners = find_partner_from_following(f, v_c, opposite, [v_c], suffix_rest, atoe)
    yield from partners


class UseFirst:
    def __init__(self, c):
        pass

    @staticmethod
    def pick_best(test_case_pairs, c, pair):
        return pair

    @staticmethod
    def reconsider_best_of_the_worst(_test_case_pairs):
        return None


class Reuser:
    """This class takes the first pair that has any reuse. Worst case is that we don't
    have any, in which case you get some pair that we have looked at.
    Make special provision for the root node and take the first pair we find."""
    def __init__(self, c):
        # The pool transfers state across all (visited) i-pairs
        #   until `pick_best` is happy. We may then use this info
        #   to reconsider our choice.
        self.pool = []
        self.c = c

    def pick_best(self, test_case_pairs, _c, pair):
        """None: keep on looking; otherwise return with `pair`.
        We give you the chance to path in *some* *other* condition `c`
        if that's useful to you."""
        path_ff, path_tt = pair
        if len(test_case_pairs) == 0:
            # Don't bother if we're in the first round.
            assert self.pool == []
            return pair  # if you just want the first
            # A different approach might want to consider throwing ALL
            #    initial pairs up to a given depths into the pool.
        else:
            cr_tt = calc_reuse(path_tt, test_case_pairs)
            cr_ff = calc_reuse(path_ff, test_case_pairs)
            if cr_tt + cr_ff > 0:
                self.pool = []
                return pair
            else:
                # Let's look for a better match.
                # Since we don't know if we find one,
                # we'll just keep the current result for that case,
                # but maybe overwrite it later.
                self.pool = self.pool + [pair]
                return None

    def reconsider_best_of_the_worst(self, _test_case_pairs):
        if len(self.pool) > 1:
            # We didn't find anything suitable.
            # Overwrite last result with a random choice,
            #  or whatever.
            # i = randint(0, len(self.pool) - 1)
            i = 0  # TODO: switch for determinism
            return self.pool[i]
        return None


class LongestPath:
    def __init__(self, c):
        # The pool transfers state across all (visited) i-pairs
        #   until `pick_best` is happy. We may then use this info
        #   to reconsider our choice.
        # Could be a set if we'd bother.
        self.pool = []
        self.c = c

    def pick_best(self, test_case_pairs, c, pair):
        # True: don't look for another, False: keep on looking.
        # Modifies `test_case_pairs` as side-effect!
        self.pool = self.pool + [pair]
        # TODO: could short-cut if we spot _a_ best result (reuse = 1, max len).
        return None

    def reconsider_best_of_the_worst(self, test_case_pairs):
        assert len(self.pool) > 0
        el = SortedList(self.pool, key=lambda path: (-calc_reuse(path[0], test_case_pairs) - calc_reuse(path[1], test_case_pairs),
                                                       # highest reuse/longest path
                                                       -size(path[0]) - size(path[1])
                                                       ))[0]
        return el


class LongestMerge(LongestPath):
    def reconsider_best_of_the_worst(self, test_case_pairs):
        assert len(self.pool) > 0
        el = SortedList(self.pool, key=lambda path: (-calc_reuse(path[0], test_case_pairs) - calc_reuse(path[1], test_case_pairs),
                                                       # highest reuse/longest path
                                                       -size(path[0]) - size(path[1])
                                                       ))[0]
        merged = (merge_Maybe_except_c(self.c, el[0], el[1]), merge_Maybe_except_c(self.c, el[1], el[0]))
        assert merged[0] is not None
        assert merged[1] is not None
        return merged


class LongestMayMerge(LongestPath):
    def reconsider_best_of_the_worst(self, test_case_pairs):
        assert len(self.pool) > 0
        el = SortedList(self.pool, key=lambda path: (-calc_may_reuse(path[0], test_case_pairs) - calc_may_reuse(path[1], test_case_pairs),
                                                       # highest reuse/longest path
                                                       -size(path[0]) - size(path[1])
                                                       ))[0]
        merged = (merge_Maybe_except_c(self.c, el[0], el[1]), merge_Maybe_except_c(self.c, el[1], el[0]))
        assert merged[0] is not None
        assert merged[1] is not None
        return merged


class ShortestPathMerge(LongestPath):

    def reconsider_best_of_the_worst(self, test_case_pairs):
        assert len(self.pool) > 0
        el = SortedList(self.pool, key=lambda path: (size(path[0]) + size(path[1]),
            -calc_may_reuse(path[0], test_case_pairs) - calc_may_reuse(path[1], test_case_pairs)
                                                       ))[0]
        merged = (merge_Maybe_except_c(self.c, el[0], el[1]), merge_Maybe_except_c(self.c, el[1], el[0]))
        return merged


class ShortestPath(LongestPath):

    def reconsider_best_of_the_worst(self, test_case_pairs):
        assert len(self.pool) > 0
        return SortedList(self.pool, key=lambda path: (-calc_reuse(path[0], test_case_pairs) - calc_reuse(path[1], test_case_pairs),
                                                       # highest reuse/longest path
                                                       size(path[0]) + size(path[1])))[0]


class BestReuseOnly(LongestPath):

    def reconsider_best_of_the_worst(self, test_case_pairs):
        assert len(self.pool) > 0
        return SortedList(self.pool, key=lambda path: (-calc_reuse(path[0], test_case_pairs) - calc_reuse(path[1], test_case_pairs)))[0]


class ShortestNoreusePath(LongestPath):

    def reconsider_best_of_the_worst(self, test_case_pairs):
        assert len(self.pool) > 0
        return SortedList(self.pool, key=lambda path: (-calc_reuse(path[0], test_case_pairs) - calc_reuse(path[1], test_case_pairs),
                                                       # highest reuse/longest path
                                                       size(path[0]) + size(path[1])))[0]


def order_path_pair(path_a, path_b, pb):
    # Just a sanity check that we didn't add duplicates.
    #  Moved out of the way for readability.
    #  Never have a node twice in a path:
    assert len(path_a) == len(set(path_a))
    assert len(path_b) == len(set(path_b))
    # Construct pair in the right order:
    if ttff(pb[1]):
        path_ff = path_a
        path_tt = path_b
    else:
        path_ff = path_b
        path_tt = path_a
    return path_ff, path_tt


def find_existing_candidates(f, c, test_case_pairs):
    # type: (BinaryDecisionDiagram, BDDVariable, dict) -> list
    # Get a unique instance for each test case that we have generated until now
    test_cases_to_false = [(p0, False) for (p0, _) in test_case_pairs.values()]
    test_cases_to_true = [(p1, True) for (_, p1) in test_case_pairs.values()]
    test_cases = test_cases_to_true + test_cases_to_false
    candidates = []
    for tc, b in test_cases:
        # tc[c] = 0/1 or None
        if tc[c] is not None:
            # Clone current test case
            tc_x = tc.copy()
            tc_x[c] = negate(tc_x[c])
            # f.restrict(tc) may return 0/1 or a new restricted BDD
            # So f.restrict(tc) != f.restrict(tc_x) is not enough for adding (tc, tc_x) as candidate
            # to the 'candidates' list. We have to ensure that the result of the evaluation is 0/1.
            val_1 = f.restrict(tc)
            val_2 = f.restrict(tc_x)
            if val_1 != val_2 and {val_1, val_2} == {BDDNODEZERO, BDDNODEONE}:
                # Add the test cases for condition 'c'
                if b:
                    cand = (tc_x, tc.copy())
                else:
                    cand = (tc.copy(), tc_x)
                candidates.append(cand)
    return candidates


def run_one_pathsearch(f, reuse_h):
    def difference(c, list_paths_1, list_paths_2, test_case_pairs):
        # type: (_, iter, iter, dict) -> list
        # list_paths_1 = [p0, p1]
        # list_paths_2 = [p1, p2]
        # where:
        # p0 = {a: 0, b: None, c: 0}
        # p1 = {a: 1, b: None, c: None}

        # list_paths_1 - list_paths_2 == [p0]

        # return [(p0, p1) for (p0, p1) in list_paths_1 if (p0, p1) not in list_paths_2]
        # return [pairs for pairs in list_paths_1 if pairs not in list_paths_2]
        result = []
        temp = []
        for pair in list_paths_1:
            p0, p1 = pair
            # Current pair is ok if there exist some more general pair in list_paths_2
            found = False
            for pair_2 in list_paths_2:
                p00, p11 = pair_2
                if merge_Maybe_except_c(c, p0, p00) is not None and merge_Maybe_except_c(c, p1, p11) is not None:
                    found = True
            if not found:
                result += [pair]

        for pair_2 in list_paths_2:
            found = False
            p0, p1 = pair_2
            for pair_1 in list_paths_1:
                p00, p11 = pair_1
                if merge_Maybe_except_c(c, p0, p00) is not None and merge_Maybe_except_c(c, p1, p11) is not None:
                    found = True
            if not found:
                temp += [pair_2]

        for p0, p1 in temp:
            cr_ff = calc_reuse(p0, test_case_pairs)
            cr_tt = calc_reuse(p1, test_case_pairs)
            # assert cr_ff + cr_tt == 0

        return result

    def is_subset(c, list_paths_1, list_paths_2, test_case_pairs):
        # type: (iter, iter) -> bool
        # list_paths_1 = [p0, p1]
        # list_paths_2 = [p1, p2]
        # where:
        # p0 = {a: 0, b: None, c: 0}
        # p1 = {a: 1, b: None, c: None}

        # return list_paths_1 - list_paths_2 == []
        return len(difference(c, list_paths_1, list_paths_2, test_case_pairs)) == 0

    def uniformize_list(pa, pb):
        path_a = uniformize(_path2point(pa), f.inputs)
        path_b = uniformize(_path2point(pb[0]), f.inputs)
        # TODO: unclear why this doesn't work on pa/pb[0]
        pair = order_path_pair(path_a, path_b, pb)
        return pair

    def _check_monotone(acc, t):
        lt0 = len(t[0])  # t[0]!
        res = lt0 >= acc[0]
        assert res
        return lt0, t

    fs = sorted(f.support, key=lambda c: c.uniqid)
    test_case_pairs = dict()
    for c in fs:
        result_ex_cand = find_existing_candidates(f, c, test_case_pairs)
        # Go through the BDD for creating a set of independent testcase candidates for condition 'c'
        # print('*** Condition:', c)
        ns = bfs_upto_c(f, c)
        # Note that this list can contain multiple paths to the same node.
        if sys.version_info >= (3, 8):
            checked_ns = accumulate(ns, _check_monotone, initial=(-1, None))
            checked_ns.__next__()  # ditch accumulate-initializer
        else:
            checked_ns = zip(repeat(None), ns)
        result = chain.from_iterable(map(lambda xpq: uniformize_list(prefix + xpq[0], (prefix + xpq[1][0], xpq[1][1])),
                                         pairs_from_node(f, v_c)) for _, (prefix, v_c) in checked_ns)
        # We're formatting all pairs in 'result' so that they match the same format than pairs in the 'result' list returned
        # by find_existing_candidates()

        # TODO: assert that the intersection of José hack and the old result is not empty.
        # Check if the existing candidates is a proper subset of the testcases proposed by the heuristics
        # assert set(result_ex_cand) <= set(result), "Not a proper subset" ---> unavaiable because 'dict's are not hashable

        # Auxiliar iterator
        result_ex_cand, list_1 = itertools.tee(result_ex_cand)
        result, list_2 = itertools.tee(result)
        llist_1 = list(list_1)
        llist_2 = list(list_2)
        # l1 = [(p0, p1), (p2, p3)]
        # l2 = [(p4, p1), (p5, p3)]

        dif = difference(c, llist_1, llist_2, test_case_pairs)
        for pair in dif:
            assert f.restrict(pair[0]) == BDDZERO, lrlr(fs, pair[0])
            assert f.restrict(pair[1]) == BDDONE, lrlr(fs, pair[1])
            assert xor(pair[0], pair[1], c)

        assert is_subset(c, llist_1, llist_2, test_case_pairs), \
            "Not a proper subset: {0}".format([(lrlr(fs, pair[0]), lrlr(fs, pair[1])) for pair in dif])

        result = result_ex_cand if len(llist_1) > 0 else result
        # Actually: Or is it even stronger? All of those in the original approach that have reuse > 0 are EXACTLY José's!
        # TODO: assert that all OTHER old results have reuse = 0.
        # (We know that there exist n+1 solutions that we would only find if we would pick the right reuse=0 now.)
        # Use a fresh instance for every condition:
        reuse_strategy = reuse_h(c)
        for pair in result:
            assert pair[0] != pair[1]
            assert f.restrict(pair[0]) == BDDZERO, lrlr(fs, pair[0])
            assert f.restrict(pair[1]) == BDDONE, lrlr(fs, pair[1])
            pick = reuse_strategy.pick_best(test_case_pairs, c, pair)
            if pick is not None:
                test_case_pairs[c] = pair
                break
        # If we didn't find any single "best" i-pair,
        #   you may e.g. pick a random one here.
        want_reconsider = reuse_strategy.reconsider_best_of_the_worst(test_case_pairs)
        if want_reconsider is not None:
            test_case_pairs[c] = want_reconsider
    assert len(test_case_pairs.keys()) == len(f.inputs), "obvious ({})".format(len(test_case_pairs.keys()))
    # Lifted from bdd.py:
    test_case = instantiate(test_case_pairs)
    uniq_test = unique_tests(test_case)
    num_test_cases = len(uniq_test)
    return test_case, num_test_cases, uniq_test


if __name__ == "__main__":
    try:
        maxRounds = int(sys.argv[1])
    except IndexError:
        maxRounds = 42

    RNGseed = 42
    # XXX Oh wow, it's even worse; MP uses a global random state?!
    #     https://github.com/numpy/numpy/issues/9650#issuecomment-327144993
    seed(RNGseed)

    hs = [LongestPath, LongestMerge, LongestMayMerge]
    # f = tcasii.makeLarge(tcasii.D15)
    # allKeys, plot_data, t_list = run_experiment(maxRounds, hs, [f], [len(f.inputs)], run_one_pathsearch)
    allKeys, plot_data, t_list = run_experiment(maxRounds, hs, tcasii.tcas, tcasii.tcas_num_cond, run_one_pathsearch)

    # plot_data and wall_clock_list must have the same length
    assert len(t_list) == len(plot_data)

    # TODO: Pity, for now you'll have to wait again for plotting.
    # Probably we could sneak in a callback again if we really need it.
    for (hi, resultMap), t in zip(plot_data, t_list):
        # Gnuplot:
        chart_name = 'VS-{}.{}-{}'.format(hs[hi](None).__class__.__name__, RNGseed, maxRounds)

        with open('{}_resultMap.csv'.format(chart_name), 'w') as csvfile:
            result_map_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for col, rm in enumerate(resultMap.values()):
                result_map_writer.writerow([col, rm])
        plot(allKeys, chart_name, resultMap, t)
    plt.show()  # Wait for windows
