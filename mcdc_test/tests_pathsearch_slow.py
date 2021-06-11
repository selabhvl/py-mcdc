from itertools import repeat, product, chain
import pytest
from pyeda.boolalg.bdd import BDDNODEZERO, BDDNODEONE
from pyeda.boolalg.bfarray import bddvars

import tcasii
from mcdc_helpers import xor, merge_Maybe
from pathsearch import UseFirst, Reuser, LongestPath, run_one_pathsearch
from mcdctestgen import run_experiment, path_via_node, equal, gen_perm_max
from tcasii import makeLarge, D15


@pytest.mark.parametrize("h,r", zip([UseFirst, Reuser, LongestPath], repeat(10)))
def test_largeD15_has_nplus1(h, r):
    f = makeLarge(D15)
    _, plot_data, _ = run_experiment((r, 1), [h], [f], [len(f.inputs)], run_one_pathsearch)
    (i, rm) = plot_data[0]
    (lowest, _count) = rm[f][0]
    if lowest != len(f.inputs)+1:
        pytest.xfail()  # distinguish unexpected runtime errors from bad results


len_map = map(lambda f: (f, len(f.inputs)), tcasii.tcas)
all_perms = chain.from_iterable(map(lambda xfxl: ((xfxl[0], p) for p in gen_perm_max(100, xfxl[1])), len_map))


@pytest.mark.parametrize("fp", all_perms)
@pytest.mark.xfail
def test_fahi_conjecture(fp):
    f, p = fp
    """Among all the independence-pairs for the root-node, we will find at least one pair,
        which we can use to construct a pair for the next condition.
    """
    fresh_var = 'f'  # apparently there's something weird going on if this name is used before, eg. in tcasii
    assert fresh_var+"[0]" not in map(lambda x: str(x), f.inputs)
    X = bddvars(fresh_var, len(f.inputs))  # Let's hope this ain't too expensive.
    theMap = {sorted(f.inputs, key=lambda c: c.uniqid)[t]: X[p[t]] for t in range(len(f.inputs))}
    f = f.compose(theMap)
    ####
    conditions = sorted(f.support, key=lambda c: c.uniqid)
    root = conditions[0]
    bdd_nodes = list(f.dfs_preorder())
    # all root
    c_nodes = [node for node in bdd_nodes if equal(node, root)]
    paths_to_zero = chain.from_iterable(path_via_node(f.node, vc, BDDNODEZERO, conditions) for vc in c_nodes)
    paths_to_one  = chain.from_iterable(path_via_node(f.node, vc, BDDNODEONE, conditions) for vc in c_nodes)
    cartesian_product = product(paths_to_zero, paths_to_one)
    paths_root = list(((path_zero, path_one) for (path_zero, path_one) in cartesian_product
                    if xor(path_zero, path_one, root)))
    # all 2nd cond
    root = conditions[1]  # ick
    c_nodes = [node for node in bdd_nodes if equal(node, root)]
    paths_to_zero = chain.from_iterable(path_via_node(f.node, vc, BDDNODEZERO, conditions) for vc in c_nodes)
    paths_to_one  = chain.from_iterable(path_via_node(f.node, vc, BDDNODEONE, conditions) for vc in c_nodes)
    cartesian_product = product(paths_to_zero, paths_to_one)
    paths_second = ((path_zero, path_one) for (path_zero, path_one) in cartesian_product
                    if xor(path_zero, path_one, root))
    ps_b = chain.from_iterable([p0, p1] for (p0, p1) in paths_second)
    for b in ps_b:
        # print("B: "+str(b))
        for (a0, a1) in paths_root:
            # Safety net: we were not sure if `==` on paths was doing the right thing, so
            #   we're (also) checking that there are (no) "?"s that we might handle incorrectly.
            assert (merge_Maybe(b, a0) is not None or merge_Maybe(b, a1) is not None) == (b == a0 or b == a1)
            if b == a0 or b == a1:
                # print("Have: {} {} {}".format(b, a0, a1))
                return
    assert False