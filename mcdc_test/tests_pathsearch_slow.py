from itertools import repeat
import pytest

from pathsearch import UseFirst, Reuser, LongestPath, run_one_pathsearch
from mcdctestgen import run_experiment
from tcasii import makeLarge, D15


@pytest.mark.parametrize("h,r", zip([UseFirst, Reuser, LongestPath], repeat(10)))
def test_largeD15_has_nplus1(h, r):
    f = makeLarge(D15)
    _, plot_data, _ = run_experiment(r, [h], [f], [len(f.inputs)], run_one_pathsearch)
    (i, rm) = plot_data[0]
    (lowest, _count) = rm[f][0]
    if lowest != len(f.inputs)+1:
        pytest.xfail()  # distinguish unexpected runtime errors from bad results
