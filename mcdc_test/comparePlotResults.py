h1=[[1, 2, 3, 1, 2, 6, 3], [2, 4, 5, 8, 0, 5, 18], [1, 1, 1, 8, 0, 140, 3], [0, 1, 0, 8, 0, 10, 13]]
h2=[[1,2,3],[2,4,5],[1,1,1]]


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

# Result = compareresult(h2)
# print(Result)