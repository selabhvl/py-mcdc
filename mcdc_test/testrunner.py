from mcdctestgen import run_experiment, satisfy_mcdc, hi_reuse_long_path, hi_reuse_short_path
import tcasii
from tcasii import makeLarge
import pytest


# Mark a set of functions in a particular way:
def markD(fs, m, ts):
    return map(lambda fn: pytest.param(fn, marks=m) if fn[0] in fs else fn, ts)


@pytest.mark.parametrize("fn", markD({tcasii.D15, tcasii.D17}, pytest.mark.xfail, zip(tcasii.tcas, tcasii.tcas_num_cond)))
def test_H1_has_nplus1(fn):
    f, n = fn
    _, plot_data, _ = run_experiment(20, [hi_reuse_long_path], [f], [n], satisfy_mcdc)
    (i, rm) = plot_data[0]
    (lowest, _count) = rm[f][0]
    assert lowest == len(f.inputs)+1


@pytest.mark.parametrize("fn", markD({tcasii.D15, tcasii.D17}, pytest.mark.xfail, zip(tcasii.tcas, tcasii.tcas_num_cond)))
def test_H2_has_nplus1(fn):
    f, n = fn
    _, plot_data, _ = run_experiment(40, [hi_reuse_short_path], [f], [n], satisfy_mcdc)
    (i, rm) = plot_data[0]
    (lowest, _count) = rm[f][0]
    assert lowest == len(f.inputs)+1


# Tests below are ignored, mostly for/during debugging:
def slow_test_Hshort_D15_has_nplus1():
    f = tcasii.D15
    _, plot_data, _ = run_experiment(100, [hi_reuse_short_path], [f], [len(f.inputs)], satisfy_mcdc)
    (i, rm) = plot_data[0]
    (lowest, _count) = rm[f][0]
    assert lowest == len(f.inputs)+1


def slow_test_trouble():
    f = makeLarge(tcasii.D15)
    _, plot_data, _ = run_experiment(2, [hi_reuse_short_path], [f], [len(f.inputs)], satisfy_mcdc)
    (i, rm) = plot_data[0]
    (lowest, _count) = rm[f][0]
    assert lowest == len(f.inputs)+1
