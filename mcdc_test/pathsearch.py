import collections
import copy
import csv
import itertools
import logging
import sys
import functools
from itertools import chain, repeat, accumulate, takewhile
from random import seed
from typing import List

from matplotlib import pyplot as plt
from sortedcontainers import SortedList

import tcasii
from comparePlotResults import results_better, results_better_n_plus_1
from vsplot import plot
from mcdctestgen import run_experiment, calc_reuse, calc_may_reuse
from pyeda.boolalg.bdd import _path2point, BDDNODEZERO, BDDNODEONE, BDDZERO, BDDONE, _iter_all_paths, bdd2expr
from mcdc_helpers import uniformize, instantiate, unique_tests, size, merge_Maybe_except_c, negate, \
    lrlr, xor, replace_final_question_marks, better_size2, Path

logger = None  # lazy


def bfs_upto_c(f, c):
    # type: (BinaryDecisionDiagram, BDDNode) -> (List[BDDNode], BDDNode)
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
    # type: (BinaryDecisionDiagram) -> callable
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
    # type: (BDDNode) -> ((List[BDDNode], List[BDDNode, bool]), BDDNode)
    # Pushed down `terms`, since it was constant anyway and couldn't be @cached.
    terms = {BDDNODEZERO, BDDNODEONE}
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
    # type: (BinaryDecisionDiagram, BDDNode) -> _
    def _check_monotone(acc, t):
        # type: ((int, _), (List[BDDNode], BDDNode)) -> (int, BDDNode)
        # ((List[BDDNode], List[BDDNode, bool]), BDDNode)
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
    # type: (BinaryDecisionDiagram, BDDNode, BDDNode, List[BDDNode], List[BDDNode], List[BDDNode]) -> (List[BDDNode], BDDNode)
    # TODO: is this a function or a relation?
    # `path` may or may not start at the root.
    # No point in memoizing, though, since `seen_nodes` will be unique
    # (although possibly overlapping) between runs starting from the same `node`.
    if node in seen_nodes_on_other_path:
        return
    suffix_rest = suffix
    if node in {BDDNODEZERO, BDDNODEONE}:
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
                if node in {BDDNODEZERO, BDDNODEONE}:
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
                if node not in {BDDNODEZERO, BDDNODEONE}:
                    # suffix was shorter, so bfs for the right terminal
                    # TODO: that's not BFS yet ;-)
                    yield from find_partner_from_following(f, node.lo, terminal, path + [node.lo], [], seen_nodes_on_other_path)
                    yield from find_partner_from_following(f, node.hi, terminal, path + [node.hi], [], seen_nodes_on_other_path)


def ttff(node):
    # type: (BDDNode) -> int
    if node is BDDNODEONE:
        return 1
    elif node is BDDNODEZERO:
        return 0
    else:
        assert False


def independence_day_for_condition(f, v_c, t):
    # type: (BinaryDecisionDiagram, BDDNode, tuple) -> (List[BDDNode], BDDNode)
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
    def __init__(self, f, c, _rng):
        pass

    @staticmethod
    def pick_best(test_case_pairs, c, pair):
        return pair

    @staticmethod
    def reconsider_best_of_the_worst(_test_case_pairs):
        return None


def random_ranked(cls, rng, choices, rank):
    # type: (object, list, list, callable) -> int
    """Pick some (random?) element from the best-ranked bucket."""
    assert len(choices) > 0
    the_list = SortedList(choices, key=rank)
    # Next, we partition off all "best" elements:
    pred_0 = rank(the_list[0])
    els_it = takewhile(lambda p: rank(p) == pred_0, the_list)
    els = list(els_it)
    i = rng.randint(0, len(els)-1)
    logger = logging.getLogger("MCDC")
    logger.debug("{}: Picking index {}/{}".format(cls.__class__.__name__, i, len(els)))
    # print("{}: Picking index {}/{}".format(cls.__class__.__name__, i, len(els)))
    return els[i]


class Reuser:
    """This class takes the first pair that has any reuse. Worst case is that we don't
    have any, in which case you get some pair that we have looked at.
    Make special provision for the root node and take the first pair we find."""
    def __init__(self, f, c, rng):
        # type: (BinaryDecisionDiagram, BDDVariable, callable) -> None
        # The pool transfers state across all (visited) i-pairs
        #   until `pick_best` is happy. We may then use this info
        #   to reconsider our choice.
        self.f = f
        self.pool = []
        self.c = c
        self.rng = rng

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
            return random_ranked(self, self.rng, self.pool, lambda _: True)
        return None


class RandomReuser(Reuser):
    """Either we only pick randomly from those that have any reuse, or a completely random one."""
    def __init__(self, f, c, rng):
        self.f = f
        self.pool_all = []
        self.pool = []
        self.c = c
        self.rng = rng

    def pick_best(self, test_case_pairs, _c, pair):
        if calc_reuse(pair[0], test_case_pairs) + calc_reuse(pair[1], test_case_pairs) > 0:
            self.pool.append(pair)
            self.pool_all = None
        else:
            if self.pool_all is not None:
                self.pool_all.append(pair)
        return None

    def reconsider_best_of_the_worst(self, test_case_pairs):
        # type: (dict) -> (dict, dict)
        def rank(path):
            # type: ((dict, dict)) -> bool
            r0 = calc_reuse(path[0], test_case_pairs)
            r1 = calc_reuse(path[1], test_case_pairs)
            return not (r0 > 0 and r1 > 0)
        if len(self.pool) > 0:
            return random_ranked(self, self.rng, self.pool, rank)
        else:
            return random_ranked(self, self.rng, self.pool_all, rank)


class LongestPath:
    def __init__(self, f, c, rng):
        # The pool transfers state across all (visited) i-pairs
        #   until `pick_best` is happy. We may then use this info
        #   to reconsider our choice.
        # Could be a set if we'd bother.
        self.f = f
        self.pool = []
        self.c = c
        self.rng = rng

    def pick_best(self, test_case_pairs, c, pair):
        # True: don't look for another, False: keep on looking.
        # Modifies `test_case_pairs` as side-effect!
        self.pool = self.pool + [pair]
        # TODO: could short-cut if we spot _a_ best result (reuse = 1, max len).
        return None

    def mkNegated(self, tc):
        m10 = copy.copy(tc)  # Quickly(?) construct partner
        tc_x_orig_keys = {x for x in m10.origs if x.uniqid <= self.c.uniqid}
        m10[self.c] = negate(m10[self.c])
        f_m10 = self.f.restrict(m10)
        m10.origs = tc_x_orig_keys.union(f_m10.inputs)
        return m10

    def reconsider_best_of_the_worst(self, test_case_pairs):
        # type: (dict) -> (dict, dict)
        def rank(path):
            # type: ((dict, dict)) -> (bool, int, int)
            r0 = calc_reuse(path[0], test_case_pairs)
            r1 = calc_reuse(path[1], test_case_pairs)
            # XXX: not always true
            # assert r0 + r1 == max(r0, r1), self.__class__.__name__ + str((r0, r1))
            return (not(r0 > 0 and r1 > 0), -r0 - r1,
                    # highest reuse/longest path
                    -path[0].size() - path[1].size()
                    )
        el = random_ranked(self, self.rng, self.pool, rank)
        m01 = merge_Maybe_except_c(self.c, el[0], el[1])
        assert m01 is not None
        m10 = self.mkNegated(m01)
        return m01, m10


class BetterSize(LongestPath):
    def reconsider_best_of_the_worst(self, test_case_pairs):
        # type: (dict) -> (dict, dict)
        def rank(path):
            # type: ((dict, dict)) -> (bool, int)
            r0 = calc_reuse(path[0], test_case_pairs)
            r1 = calc_reuse(path[1], test_case_pairs)
            return (not(r0 > 0 and r1 > 0),
                    -better_size2(test_case_pairs, path)
                    )
        el = random_ranked(self, self.rng, self.pool, rank)
        m01 = merge_Maybe_except_c(self.c, el[0], el[1])
        assert m01 is not None
        m10 = self.mkNegated(m01)
        return m01, m10


class LongestBool(LongestPath):
    def reconsider_best_of_the_worst(self, test_case_pairs):
        # type: (dict) -> (dict, dict)
        def rank(path):
            # type: ((dict, dict)) -> (bool, bool, int)
            r0 = calc_reuse(path[0], test_case_pairs)
            r1 = calc_reuse(path[1], test_case_pairs)
            return (not(r0 > 0 and r1 > 0), not r0 + r1 > 0,
                    # Since False < True, if there is reuse, we need False, so that it goes to the front.
                    # longest path
                    -path[0].size() - path[1].size()
                    )
        el = random_ranked(self, self.rng, self.pool, rank)
        m01 = merge_Maybe_except_c(self.c, el[0], el[1])
        assert m01 is not None
        m10 = self.mkNegated(m01)
        return m01, m10


class LongestMayMerge(LongestPath):
    def reconsider_best_of_the_worst(self, test_case_pairs):
        # type: (dict) -> (dict, dict)
        def rank(path):
            # type: ((dict, dict)) -> (bool, int, int)
            # `calc_may_reuse()` is much slower then just `calc_reuse()`.
            r0 = calc_may_reuse(path[0], test_case_pairs)
            r1 = calc_may_reuse(path[1], test_case_pairs)
            return (not(r0 > 0 and r1 > 0), -r0 - r1,
                    # highest reuse/longest path
                    -path[0].size() - path[1].size())
        el = random_ranked(self, self.rng, self.pool, rank)
        m01 = merge_Maybe_except_c(self.c, el[0], el[1])
        assert m01 is not None
        m10 = self.mkNegated(m01)
        return m01, m10


class LongestBoolMay(LongestMayMerge):
    def reconsider_best_of_the_worst(self, test_case_pairs):
        # type: (dict) -> (dict, dict)
        def rank(path):
            # type: ((dict, dict)) -> (bool, bool, int)
            # `calc_may_reuse()` is much slower then just `calc_reuse()`.
            r0 = calc_may_reuse(path[0], test_case_pairs)
            r1 = calc_may_reuse(path[1], test_case_pairs)
            return (not (r0 > 0 and r1 > 0), not (r0 + r1 > 0),
                    # highest reuse/longest path
                    -path[0].size() - path[1].size())
        el = random_ranked(self, self.rng, self.pool, rank)
        m01 = merge_Maybe_except_c(self.c, el[0], el[1])
        assert m01 is not None
        m10 = self.mkNegated(m01)
        return m01, m10


class ShortestPathMerge(LongestPath):

    def reconsider_best_of_the_worst(self, test_case_pairs):
        # type: (dict) -> (dict, dict)
        def rank(path):
            # type: ((dict, dict)) -> (int, int)
            return (size(path[0]) + size(path[1]),
                    -calc_may_reuse(path[0], test_case_pairs) - calc_may_reuse(path[1], test_case_pairs))
        el = random_ranked(self, self.rng, self.pool, rank)
        merged = (merge_Maybe_except_c(self.c, el[0], el[1]), merge_Maybe_except_c(self.c, el[1], el[0]))
        return merged


class ShortestPath(LongestPath):

    def reconsider_best_of_the_worst(self, test_case_pairs):
        # type: (dict) -> (dict, dict)
        def rank(path):
            # type: ((dict, dict)) -> (int, int)
            return (-calc_reuse(path[0], test_case_pairs) - calc_reuse(path[1], test_case_pairs),
                    # highest reuse/longest path
                    size(path[0]) + size(path[1]))
        return random_ranked(self, self.rng, self.pool, rank)


class BestReuseOnly(LongestPath):

    def reconsider_best_of_the_worst(self, test_case_pairs):
        # type: (dict) -> (dict, dict)
        def rank(path):
            # type: ((dict, dict)) -> int
            return -calc_reuse(path[0], test_case_pairs) - calc_reuse(path[1], test_case_pairs)
        return random_ranked(self, self.rng, self.pool, rank)


class ShortestNoreusePath(LongestPath):

    def reconsider_best_of_the_worst(self, test_case_pairs):
        # type: (dict) -> (dict, dict)
        def rank(path):
            # type: ((dict, dict)) -> (int, int)
            (-calc_reuse(path[0], test_case_pairs) - calc_reuse(path[1], test_case_pairs),
             # highest reuse/longest path
             size(path[0]) + size(path[1]))
        return random_ranked(self, self.rng, self.pool, rank)


def order_path_pair(path_a, path_b, pb):
    # type: (dict, dict, tuple) -> (dict, dict)
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

    def unsatisfy_all(f):
        # type: (BinaryDecisionDiagram) -> iter
        for path in _iter_all_paths(f.node, BDDNODEZERO):
            yield _path2point(path)

    def filtered_restrict(f, tc):
        # type: (BinaryDecisionDiagram, dict) -> BinaryDecisionDiagram
        filtered_tc = {key: val for key, val in tc.items() if val is not None}
        return f.restrict(filtered_tc)

    # Get a unique instance for each test case that we have generated until now
    test_cases_to_false = [(p0, False) for (p0, _) in test_case_pairs.values()]
    test_cases_to_true = [(p1, True) for (_, p1) in test_case_pairs.values()]
    test_cases = test_cases_to_true + test_cases_to_false
    for tc, b in test_cases:
        assert type(tc) == Path
        val_1 = filtered_restrict(f, tc)  # assert only
        assert (val_1.is_zero() and not b) or (val_1.is_one() and b), (val_1, b)
        # tc[c] = 0/1 or None
        if tc[c] is not None:
            # Clone current test case
            tc_x = copy.copy(tc)
            assert type(tc_x) == Path
            tc_x_orig_keys = {x for x in tc_x.origs if x.uniqid <= c.uniqid}
            tc_x[c] = negate(tc_x[c])
            # f.restrict(tc) may return 0/1 or a new restricted BDD
            # So f.restrict(tc) != f.restrict(tc_x) is not enough for adding (tc, tc_x) as candidate
            # to the 'candidates' list. We have to ensure that the result of the evaluation is 0/1.
            val_2 = filtered_restrict(f, tc_x)
            if not b:
                if val_2.is_one():
                    tc_x.origs = tc_x_orig_keys
                    yield (tc, tc_x)
                else:
                    for res in val_2.satisfy_all():
                        val_2.restrict(res)
                        if val_2.is_one():
                            print("res: {0}".format(res))
                            tc_0 = Path(dict(tc).update(res))
                            tc_0.origs = tc.origs  # we had all that we needed
                            tc_1 = Path(dict(tc_x).update(res))
                            tc_1.origs = tc_x_orig_keys.union(res.keys())
                            yield (tc_0, tc_1)
            else:
                if val_2.is_zero():
                    tc_x.origs = tc_x_orig_keys
                    yield (tc_x, tc)
                else:
                    for res in unsatisfy_all(val_2):
                        val_2.restrict(res)
                        if val_2.is_zero():
                            print("res: {0}".format(res))
                            tc_0 = Path(dict(tc_x).update(res))
                            tc_0.origs = tc_x_orig_keys.union(res.keys())
                            tc_1 = Path(dict(tc).update(res))
                            tc_1.origs = tc.origs
                            yield (tc_0, tc_1)
    return


def run_one_pathsearch(f, reuse_h, rng):
    def difference(c, list_paths_1, list_paths_2, test_case_pairs):
        # type: (BDDVariable, list, list, dict) -> list
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
        # type: (BDDVariable, list, list, dict) -> bool
        # list_paths_1 = [p0, p1]
        # list_paths_2 = [p1, p2]
        # where:
        # p0 = {a: 0, b: None, c: 0}
        # p1 = {a: 1, b: None, c: None}

        # return list_paths_1 - list_paths_2 == []
        return len(difference(c, list_paths_1, list_paths_2, test_case_pairs)) == 0

    def uniformize_list(pa, pb):
        # type: (tuple, tuple) -> (dict, dict)
        path_a = uniformize(_path2point(pa), f.inputs)
        path_b = uniformize(_path2point(pb[0]), f.inputs)
        # TODO: unclear why this doesn't work on pa/pb[0]
        pair = order_path_pair(path_a, path_b, pb)
        return pair

    def _check_monotone(acc, t):
        # type: (iter, iter) -> (int, iter)
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
        result = map(lambda p0p1: (Path(p0p1[0]), Path(p0p1[1])), result)
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
        reuse_strategy = reuse_h(f, c, rng)
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
        # The current test case MAY have been derived from an existing one, so we should specialise.
        test_case_pairs = instantiate(test_case_pairs)
        # Note: there was no guarantee yet if the pair is fully merged, the heuristics should make sure.
    assert len(test_case_pairs.keys()) == len(f.inputs), "obvious ({})".format(len(test_case_pairs.keys()))
    # Lifted from bdd.py:
    # TODO -- eliminate: test_case = instantiate(test_case_pairs)
    replace_final_question_marks(test_case_pairs)
    uniq_test = unique_tests(test_case_pairs)
    num_test_cases = len(uniq_test)
    return test_case_pairs, num_test_cases, uniq_test


if __name__ == "__main__":
    try:
        maxRounds = int(sys.argv[1])
    except IndexError:
        maxRounds = 42

    try:
        rngRounds = int(sys.argv[2])
    except IndexError:
        rngRounds = 3

    RNGseed = 11
    # XXX Oh wow, it's even worse; MP uses a global random state?!
    #     https://github.com/numpy/numpy/issues/9650#issuecomment-327144993
    seed(RNGseed)

    # LongestPath and LongestMayMerge seem identical.
    hs = [LongestMayMerge, LongestPath, LongestBool, LongestBoolMay, BetterSize, RandomReuser]
    # f = tcasii.makeLarge(tcasii.D1)
    # allKeys, plot_data, t_list = run_experiment((maxRounds, rngRounds), hs, [f], [len(f.inputs)], run_one_pathsearch)
    # t_list = execution time
    allKeys, plot_data, t_list = run_experiment((maxRounds, rngRounds), hs, tcasii.tcas, tcasii.tcas_num_cond, run_one_pathsearch)

    # plot_data and wall_clock_list must have the same length
    assert len(t_list) == len(plot_data)

    # TODO: Pity, for now you'll have to wait again for plotting.
    # Probably we could sneak in a callback again if we really need it.
    def only_nplus1(args):
        hi, resultMap = args
        result_vec = []
        for vs in resultMap.values():
            (ns, count) = vs[0]
            if (ns == tcasii.tcas_num_cond[hi]+1):
                result_vec.append(count)
            else:
                # Obvs wouldn't work so well with masking:
                result_vec.append(-1)
        return result_vec

    ls = list(map(only_nplus1, plot_data))
    print(results_better(ls))  # TODO: Probably incorrect. `results_better` uses <=, but n+1 should use >=.
    print(results_better_n_plus_1(ls, tcasii.tcas_num_cond))

    for (hi, resultMap), t in zip(plot_data, t_list):
        # Gnuplot:
        chart_name = 'VS-{}.{}-{}-{}'.format(hs[hi](None, None, None).__class__.__name__, RNGseed, maxRounds, rngRounds)

        with open('{}_resultMap.csv'.format(chart_name), 'w') as csvfile:
            result_map_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for col, rm in enumerate(resultMap.values()):
                result_map_writer.writerow([col, rm])
        plot(allKeys, chart_name, resultMap, t)
    plt.show()  # Wait for windows
