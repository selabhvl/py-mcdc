import random

from mcdctestgen import run_experiment, h1, h2, faustins_mechanism
import tcasii
from tcasii import makeLarge
import pytest


# @pytest.mark.parametrize("f", tcas)
# def test_all_tcasii(f):
#     test_case, num_test_cases, uniq_test = f.satisfy_mcdc(h1)
#     is_mcdc = tcasii.test_mcdc(f, test_case)
#     assert is_mcdc
#     assert num_test_cases == len(test_case)+1


# Mark a set of functions in a particular way:
def markD(fs, m, ts):
    return map(lambda fn: pytest.param(fn, marks=m) if fn[0] in fs else fn, ts)


@pytest.mark.parametrize("fn", markD(set([tcasii.D15, tcasii.D17]), pytest.mark.xfail, zip(tcasii.tcas, tcasii.tcas_num_cond)))
def test_H1_has_nplus1(fn):
    f, n = fn
    _, plot_data, _ = run_experiment(20, [h1], [f], [n], faustins_mechanism)
    (i, rm) = plot_data[0]
    (lowest, _count) = rm[f][0]
    assert lowest == len(f.inputs)+1


@pytest.mark.parametrize("fn", markD(set([tcasii.D15, tcasii.D17]), pytest.mark.xfail, zip(tcasii.tcas, tcasii.tcas_num_cond)))
def test_H2_has_nplus1(fn):
    f, n = fn
    _, plot_data, _ = run_experiment(40, [h2], [f], [n], faustins_mechanism)
    (i, rm) = plot_data[0]
    (lowest, _count) = rm[f][0]
    assert lowest == len(f.inputs)+1


# Tests below are ignored, mostly for/during debugging:
def slow_test_H1_D15_has_nplus1():
    f = tcasii.D15
    _, plot_data, _ = run_experiment(100, [h1], [f], [len(f.inputs)], faustins_mechanism)
    (i, rm) = plot_data[0]
    (lowest, _count) = rm[f][0]
    assert lowest == len(f.inputs)+1


def slow_test_H3_trouble():
    f = makeLarge(tcasii.D15)
    _, plot_data, _ = run_experiment(2, [h1], [f], [len(f.inputs)], faustins_mechanism)
    (i, rm) = plot_data[0]
    (lowest, _count) = rm[f][0]
    assert lowest == len(f.inputs)+1
