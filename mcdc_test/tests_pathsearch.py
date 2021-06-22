from functools import reduce
from itertools import chain, product, tee
from random import Random

import pytest
from graphviz import Source

from pyeda.boolalg.bdd import _path2point
from pyeda.boolalg.bfarray import bddvars

import tcasii
from pathsearch import *
from mcdc_helpers import better_size

def test_something():
    f = tcasii.D3
    fs = list(f.support)
    cs = map(str, fs)
    di = list(cs).index('d')
    c = fs[di]
    assert str(c) == 'd', str(c)
    ns = list(bfs_upto_c(f, c))
    assert len(ns) == 2, ns
    (prefix, fst) = ns[0]
    # print(uniformize(_path2point(p), f.inputs))
    ts = list(terminals_via_bfs_from(fst))
    tsm = list(map(lambda t: uniformize(_path2point(prefix + t[0][0]), f.inputs), ts))
    assert len(tsm) == 5
    # length monotone?
    check, check_l = reduce(lambda acc, t: (acc[0] and len(t) >= acc[1], len(t)), tsm, (True, -1))
    assert check, check_l
    # let's take one path and find its independence partner:
    for ((atoe, suffix), terminal) in ts:
        atoe_s = uniformize(_path2point(atoe), f.inputs)
        prefix_s = uniformize(_path2point(prefix), f.inputs)
        print('A', str(ttff(terminal)), atoe_s)
        print('P:', prefix_s, 'S:', suffix)
        if terminal is BDDNODEONE:
            opposite = BDDNODEZERO
        else:
            assert terminal is BDDNODEZERO
            opposite = BDDNODEONE
        # Flip start of suffix:
        cur, *rest = suffix
        cur_c, cur_s = cur
        suffix_flipped = [(cur_c, not cur_s)] + rest
        print('SF:', suffix_flipped)
        partners = find_partner_from_following(f, fst, opposite, prefix, suffix_flipped, prefix + atoe)
        print(list(partners))
        # Produces paths to either leaf for now:
        pss = list(map(lambda p: '{}:{}'.format(ttff(p[1]), uniformize(_path2point(p[0]), f.inputs)), partners))
        print(pss)


def test_D1():
    f = tcasii.D1
    # src = Source(f.to_dot())
    # src.render('/tmp/1', view=True)
    fs = list(f.support)
    cs = list(map(str, fs))
    for c_s in 'abcde':
        print('*** Condition:', c_s)
        di = list(cs).index(c_s)
        c = fs[di]
        assert str(c) == c_s, str(c)
        ns = list(bfs_upto_c(f, c))
        assert c_s != 'a' or len(ns) == 1
        assert c_s != 'b' or len(ns) == 1
        assert c_s != 'c' or len(ns) == 1
        assert c_s != 'd' or len(ns) == 2
        assert c_s != 'e' or len(ns) == 4
        check, check_l = reduce(lambda acc, t: (acc[0] and len(t[0]) >= acc[1], len(t[0])), ns, (True, -1))
        assert check, check_l
        found_partner = False
        for (prefix, fst) in ns:
            print('**** Next:')
            assert c_s != 'a' or len(prefix) == 1
            assert c_s != 'b' or len(prefix) == 2
            assert c_s != 'c' or len(prefix) == 3
            # Shows all None for the root, since there's "nowhere to go":
            print(uniformize(_path2point(prefix), f.inputs))
            ts = list(terminals_via_bfs_from(fst))
            assert c_s != 'c' or len(ts) == 5
            tsm = list(map(lambda t: uniformize(_path2point(prefix + t[0][0]), f.inputs), ts))
            # length monotone?
            check, _ = reduce(lambda acc, t: (acc[0] and len(t) >= acc[1], len(t)), tsm, (True, -1))
            assert check
            # let's take one path and try find its independence partner:
            # TODO: will EACH have an i-partner, one, or some?
            partners = map(lambda i: (prefix + i[0], i[1]), chain.from_iterable([independence_day_for_condition(f, fst, t) for t in ts]))
            partners_l = list(partners)
            pss = list(map(lambda p: '{}:{}'.format(ttff(p[1]), uniformize(_path2point(p[0]), f.inputs)), partners_l))
            # assert not found_partner
            found_partner = len(pss) > 0
            if found_partner:
                break
            else:
                print("No such luck yet")
        assert found_partner, "There must be one?"


def test_D15():
    f = tcasii.D15
    # src = Source(f.to_dot())
    # src.render('/tmp/1', view=True)
    fs = list(f.support)
    cs = list(map(str, fs))
    counter = dict()
    for c_s in cs:
        print('*** Condition:', c_s)
        counter[c_s] = []
        di = list(cs).index(c_s)
        c = fs[di]
        assert str(c) == c_s, str(c)
        ns = list(bfs_upto_c(f, c))
        assert c_s != 'a' or len(ns) == 1
        assert c_s != 'b' or len(ns) == 2
        assert c_s != 'c' or len(ns) == 3
        assert c_s != 'd' or len(ns) == 1
        assert c_s != 'e' or len(ns) == 1
        assert c_s != 'g' or len(ns) == 6
        assert c_s != 'l' or len(ns) == 6
        check, check_l = reduce(lambda acc, t: (acc[0] and len(t[0]) >= acc[1], len(t[0])), ns, (True, -1))
        assert check, check_l
        found_partner = False
        for (prefix, fst) in ns:
            print('**** Next:')
            # assert c_s != 'a' or len(prefix) == 1
            # assert c_s != 'b' or len(prefix) == 2
            # assert c_s != 'c' or len(prefix) == 3
            # Shows all None for the root, since there's "nowhere to go":
            print(uniformize(_path2point(prefix), f.inputs))
            ts = list(terminals_via_bfs_from(fst))
            counter[c_s].append(len(ts))
            tsm = list(map(lambda t: uniformize(_path2point(prefix + t[0][0]), f.inputs), ts))
            # length monotone?
            check, _ = reduce(lambda acc, t: (acc[0] and len(t) >= acc[1], len(t)), tsm, (True, -1))
            assert check
            # let's take one path and try find its independence partner:
            # TODO: will EACH have an i-partner, one, or some?
            partners = map(lambda i: (prefix + i[0], i[1]), chain.from_iterable([independence_day_for_condition(f, fst, t) for t in ts]))
            partners_l = list(partners)
            pss = list(map(lambda p: '{}:{}'.format(ttff(p[1]), uniformize(_path2point(p[0]), f.inputs)), partners_l))
            print(pss)
            # assert not found_partner
            found_partner = found_partner or len(partners_l) > 0
        assert found_partner, "There must be one?"
    assert len(counter['l']) == 6 and all(map(lambda x: x == 2 or x == 3, counter['l'])) and sum(counter['l']) == 14
    assert len(counter['g']) == 6 # and all(map(lambda x: x == 2 or x == 3, counter['g'])) and sum(counter['g']) == 14


@pytest.mark.parametrize("fn", product([UseFirst, Reuser, LongestPath], zip(tcasii.tcas, tcasii.tcas_num_cond)))
def test_volker_has_nplus1(fn):
    h, (f, n) = fn
    _, plot_data, _ = run_experiment((20, 1), [h], [f], [n], run_one_pathsearch)
    (i, rm) = plot_data[0]
    (lowest, _count) = rm[f][0]
    if lowest != len(f.inputs)+1:
        pytest.xfail('{} > {}'.format(lowest, len(f.inputs)+1))


@pytest.mark.parametrize("f,n", zip(tcasii.tcas, tcasii.tcas_num_cond))
#@pytest.mark.parametrize("f,n", zip([tcasii.D1, tcasii.D2], [3, 3]))
#@pytest.mark.parametrize("f,n", zip([tcasii.D1], [3]))
#@pytest.mark.parametrize("f,n", zip([tcasii.D10], [3]))
def test_Dx(f, n):
    print()  # LF for pytest
    #src = Source(f.to_dot())
    # src.render('/tmp/1', view=True)
    test_case, _, _ = run_one_pathsearch(f, UseFirst, Random(42))  # Note: RNG unused.
    assert tcasii.test_mcdc(f, test_case)
    # return test_case, num_test_cases, uniq_test
    # print(tcasii.test_mcdc(f, test_case), tcasii.test_mcdc(f, test_case))

@pytest.mark.parametrize("f,n", zip(tcasii.tcas, tcasii.tcas_num_cond))
def test_better_size(f, n):
    test_case, _, _ = run_one_pathsearch(f, UseFirst, Random(42))  # Note: RNG unused.
    conds = list(test_case.keys())
    # better_size(test_case, (p0, p1))
    bs = better_size(test_case, test_case[conds[0]])
    assert bs > 0
