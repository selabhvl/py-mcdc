import sys
from vsplot import load_resultMap


def compareresult(hs):
    BoolTF = map(lambda vals: map(lambda v: v == min(vals), vals), zip(*hs))
    comparison = map(lambda x: sum(x), zip(*BoolTF))
    return list(comparison)


def compareresult_using_list_comprh(hs):
    # Zip groups the 1st experiment of all heuristics, then the 2nd experiment, and so on.
    # hs = [h_1, h_2, ..., h_n] where
    # h_i = [run_i_1, run_i_2, ..., run_i_n]
    # Therefore,
    # zip(*hs) = [run_1, run_2, ..., run_n] where
    # run_1 = [run_1_i for all h_i in hs]

    set_of_runs = ([hi == min(run_j) for hi in run_j] for run_j in zip(*hs))
    num_times_hi_is_best = [sum(hi_is_best) for hi_is_best in zip(*set_of_runs)]

    return num_times_hi_is_best


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
    res = compareresult(heuristics)
    print("H1 better than H2: {0}\nH2 better than H1: {1}".format(res[0], res[1]))
