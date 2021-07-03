import sys
import numpy as np
from itertools import permutations
from vsplot import load_resultMap


def compareresult(hs):
    BoolTF = map(lambda vals: map(lambda v: v == min(vals), vals), zip(*hs))
    comparison = map(lambda x: sum(x), zip(*BoolTF))
    return list(comparison)


def results_better_equal(hs, criteria=min):
    # Zip groups the 1st experiment of all heuristics, then the 2nd experiment, and so on.
    # hs = [h_1, h_2, ..., h_n] where
    # h_i = [run_i_1, run_i_2, ..., run_i_m]
    # Therefore,
    # zip(*hs) = [run_1, run_2, ..., run_m] where
    # run_1 = [run_1_i for all h_i in hs]

    set_of_runs = ([hi == criteria(run_j) for hi in run_j] for run_j in zip(*hs))
    num_times_hi_is_better_equal = [sum(hi_is_best) for hi_is_best in zip(*set_of_runs)]

    return num_times_hi_is_better_equal


def results_better(hs):
    # type: (list) -> (list, list)
    # hs = [h_1, h_2, ..., h_n] are the results of running n heuristics, where
    # h_i = [run_i_1, run_i_2, ..., run_i_m]

    hs_array = map(np.array, hs)
    # permutations('ABCD', 2) == [AB AC AD BA BC BD CA CB CD DA DB DC]
    per = permutations(hs_array, 2)
    return [sum(np.less(h_i, h_j)) for (h_i, h_j) in per]


def results_equal(hs):
    # type: (list) -> (list, list)
    # hs = [h_1, h_2, ..., h_n] are the results of running n heuristics, where
    # h_i = [run_i_1, run_i_2, ..., run_i_m]

    hs_array = map(np.array, hs)
    # permutations('ABCD', 2) == [AB AC AD BA BC BD CA CB CD DA DB DC]
    per = permutations(hs_array, 2)
    return [sum(np.equal(h_i, h_j)) for (h_i, h_j) in per]


def results_better_two_heuristics(hs):
    # type: (list) -> (list, list)
    # hs = [h_1, h_2] are the results of running two heuristics, where
    # h_i = [run_i_1, run_i_2, ..., run_i_n]

    h1, h2 = np.array(hs[0]), np.array(hs[1])
    return [sum(h1 < h2), sum(h2 < h1)]


def results_equal_two_heuristics(hs):
    # type: (list) -> (list, list)
    # hs = [h_1, h_2] are the results of running two heuristics, where
    # h_i = [run_i_1, run_i_2, ..., run_i_n]

    h1, h2 = np.array(hs[0]), np.array(hs[1])
    return [sum(h1 == h2), sum(h2 == h1)]


def extract_results(resultMap):
    result = []
    for (bool_func, histogram) in resultMap.items():
        expected_value = sum(map(lambda kv: (kv[0] - len(bool_func.inputs)) * kv[1], histogram))
        result += [expected_value]
    return result


if __name__ == "__main__":
    chart_1_csv = sys.argv[1]
    chart_2_csv = sys.argv[2]
    resultMap_chart_1, _ = load_resultMap(chart_1_csv)
    resultMap_chart_2, _ = load_resultMap(chart_2_csv)
    h_1 = extract_results(resultMap_chart_1)
    h_2 = extract_results(resultMap_chart_2)

    # Compare heuristics h_1 and h_2
    heuristics = [h_1, h_2]
    better_equal = results_better_equal(heuristics)
    better = results_better(heuristics)
    equal = results_equal(heuristics)
    print("H1 better or equal than H2: {0}\nH2 better or equal than H1: {1}".format(better_equal[0], better_equal[1]))
    print("H1 better than H2: {0}\nH2 better than H1: {1}".format(better[0], better[1]))
    print("H1 equal than H2: {0}\nH2 equal than H1: {1}".format(equal[0], equal[1]))
