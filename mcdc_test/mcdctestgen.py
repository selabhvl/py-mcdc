import csv
import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager
from math import factorial
from random import randint, seed
import sys
import time
from pyeda.boolalg.bdd import bddvar, expr2bdd, bdd2expr
from pyeda.boolalg.expr import expr
from mcdc_helpers import *
from sortedcontainers import SortedList
from pyeda.inter import bddvars
from functools import reduce
from itertools import permutations, repeat, product

from vsplot import plot
import tcasii
from tcasii import test_mcdc


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# MP: Things that need to be synced.
# We use `None` to provoke a type-error if we ever get this wrong.
# See `init_globals()`.
# How many rounds:
maxRounds = None
tcas = None
tcas_num_cond = None
mechanism = None


def init_globals(_runs, _tcas, _tcas_num_cond, _mechanism):
    # https://stackoverflow.com/a/28862472/60462
    global maxRounds
    global tcas
    global tcas_num_cond
    global mechanism
    maxRounds = _runs
    sys.setrecursionlimit(1500)  # Required e.g. for makeLarge(D15)
    tcas = list(map(lambda x: expr2bdd(expr(x)), _tcas))
    tcas_num_cond = _tcas_num_cond
    mechanism = _mechanism


def sample_one(l):
    have = []
    while len(have) < l:
        j = randint(0, l - 1)
        if j not in have:
            have.append(j)
    return have


def gen_perm(l):
    global maxRounds
    # If you're asking for more rounds than we have permutations,
    #   we'll give them all to you.
    if maxRounds >= factorial(l):
        # Still hating MP with a vengeance. Force deep eval.
        return list(map(lambda p: list(p), permutations(range(l))))
    # Otherwise, we'll sample sloppily:
    perms = []
    for _ in range(maxRounds):
        # `append`/` += [i]` seems to be the efficient choice here.
        perms.append(sample_one(l))  # ignore duplicates for now XXX
    return perms


# Def. as per paper
def calc_reuse(path, test_case):
    # for p in test_case.values():
    #   print("reuse:\t{0}".format(p))
    # tcs = map(lambda p: (merge_Maybe(conditions,path,p[0]),merge_Maybe(conditions, path, p[1])), test_case.values())
    tcs = filter(lambda p: p[0] == path or p[1] == path, test_case.values())
    return len(list(tcs))


def h1(tcs, c, paths_to_zero, paths_to_one):
    cartesian_product = product(paths_to_zero, paths_to_one)

    # Choose path_zero and path_one that only differs on condition c
    paths = ((path_zero, path_one) for (path_zero, path_one) in cartesian_product
             if xor(path_zero, path_one, c))
    return SortedList(paths, key=lambda path: (-calc_reuse(path[0], tcs) - calc_reuse(path[1], tcs),
            # highest reuse/shortest path
            size(path[0]) + size(path[1])))


def h2(tcs, c, paths_to_zero, paths_to_one):
    cartesian_product = product(paths_to_zero, paths_to_one)

    # Choose path_zero and path_one that only differs on condition c
    paths = ((path_zero, path_one) for (path_zero, path_one) in cartesian_product
             if xor(path_zero, path_one, c))
    return SortedList(paths, key=lambda path: (-calc_reuse(path[0], tcs) - calc_reuse(path[1], tcs),
                                               # highest reuse/longest path
                                               -size(path[0]) - size(path[1])))


def h3(tcs, c, paths_to_zero, paths_to_one):
    # Not "very good", e.g. can't find optimal solution in general, here D3:
    #    Num_Cond=7: min=8, |p|=5040, [(9, 5040)]
    cartesian_product = product(paths_to_zero, paths_to_one)

    # Choose path_zero and path_one that only differs on condition c
    paths = ((path_zero, path_one) for (path_zero, path_one) in cartesian_product
             if xor(path_zero, path_one, c))
    return [paths.__next__()]


def h3s(tcs, c, paths_to_zero, paths_to_one):
    # Try to improve over H3 by pre-sorting. Ideally, we'd want to build the product "diagonally".
    # Shortest path/highest reuse
    # Still "meh" (of course[*]) as in "does not find n+1 at all": Num_Cond=7: min=8, |p|=5040, [(9, 5040)]
    # Re: [*] "of course" -- or is it? Of course ORDER of solutions plays a role, since we're only
    #    taking the first one once we return results here!
    paths_to_zero = SortedList(paths_to_zero, key=lambda path: (size(path), -calc_reuse(path, tcs)))
    paths_to_one = SortedList(paths_to_one, key=lambda path: (size(path), -calc_reuse(path, tcs)))
    cartesian_product = product(paths_to_zero, paths_to_one)

    # Choose path_zero and path_one that only differs on condition c
    paths = ((path_zero, path_one) for (path_zero, path_one) in cartesian_product
             if xor(path_zero, path_one, c))
    return [paths.__next__()]


def h3h(tcs, c, paths_to_zero, paths_to_one):
    # highest reuse/shortest path
    # Still fails D3
    paths_to_zero = SortedList(paths_to_zero, key=lambda path: (-calc_reuse(path, tcs), size(path)))
    paths_to_one = SortedList(paths_to_one, key=lambda path: (-calc_reuse(path, tcs), size(path)))
    cartesian_product = product(paths_to_zero, paths_to_one)

    # Choose path_zero and path_one that only differs on condition c
    paths = ((path_zero, path_one) for (path_zero, path_one) in cartesian_product
             if xor(path_zero, path_one, c))
    return [paths.__next__()]


def h3hl(tcs, c, paths_to_zero, paths_to_one):
    # highest reuse/longest path
    # Still fails D3
    paths_to_zero = SortedList(paths_to_zero, key=lambda path: (-calc_reuse(path, tcs), -size(path)))
    paths_to_one = SortedList(paths_to_one, key=lambda path: (-calc_reuse(path, tcs), -size(path)))
    cartesian_product = product(paths_to_zero, paths_to_one)

    # Choose path_zero and path_one that only differs on condition c
    paths = filter(lambda p: xor(p[0], p[1], c), cartesian_product)
    return [paths.__next__()]


def processFP_with_time(args):
    tic = time.process_time_ns()
    i = args[0]
    perms = args[1]
    h = args[2]
    thread_time = args[3]
    new_args = (i, perms, h)
    value = processFP(new_args)
    toc = time.process_time_ns()
    elapsed_time = toc - tic
    thread_time.append(elapsed_time)
    return value


def processFP(args):
    global mechanism
    i, p, heuristic = args
    f1 = tcas[i]
    fresh_var = 'f'  # apparently there's something weird going on if this name is used before, eg. in tcasii
    assert fresh_var+"[0]" not in map(lambda x: str(x), f1.inputs)
    X = bddvars(fresh_var, len(f1.inputs))  # Let's hope this ain't too expensive.
    theMap = {sorted(f1.inputs, key=lambda c: c.uniqid)[t]: X[p[t]] for t in range(len(f1.inputs))}
    f2 = f1.compose(theMap)
    test_case, num_test_cases, uniq_test = mechanism(f2, heuristic)
    is_mcdc = test_mcdc(f2, test_case)
    # print('Round: {0} Number of Conditions: {1} Number of TCs: {2}'.format(RoundN, len(f1.inputs), num_test_cases))
    assert len(f1.inputs) == len(f2.inputs)
    if num_test_cases <= len(f1.inputs) or not is_mcdc:
        # inv_map = {v: k for k, v in theMap.items()}
        num_test_cases = -1  # indicate that this set is useless
        # TODO: I think this disappears in the final result?
        # Not a problem since we show % in relation to maxRounds.
    return num_test_cases


def process_one(arg):
    results = {}
    i, (f1, num_cond) = (arg[0][0], (arg[0][1][0], arg[0][1][1]))
    # Pity that with mp we can't share/reuse perms without more complications.
    # TODO: Actually, after the "global pool" refactoring, we probably can again!
    perms = arg[1]
    h = arg[2]
    pool = arg[3]
    thread_time = arg[4]
    ntcs = pool.map(processFP_with_time, zip(repeat(i), perms, repeat(h), repeat(thread_time)))
    count = 0  # track down a presumed glitch
    for num_test_cases in ntcs:
        try:
            old = results[num_test_cases]
        except KeyError:
            old = 0
        results[num_test_cases] = old + 1
        count = count + 1
    assert count > 0, perms
    colour = ''
    # We keep these around for the CSV:
    sr = sorted(results)
    resultMap = [(k, results[k]) for k in sr]

    if len(resultMap) == 0 or len(f1.inputs) + 1 < resultMap[0][0]:
        # We didn't find any n+1 solution.
        colour = bcolors.FAIL
    print('{}Num_Cond={}: min={}, |p|={}, {}'.format(colour, num_cond, len(f1.inputs) + 1, factorial(len(f1.inputs)),
                                                     resultMap), bcolors.ENDC)
    # TODO: I don't think we need this print statement below in general, both values are included in the line above.
    if num_cond != len(f1.inputs):
        print(bcolors.WARNING + 'Decision with Masked conditions: D_' + str(i + 1), bcolors.ENDC)
    # TODO: could (should?) be done outside? OTOH, doesn't really matter
    # "Normalize" result; we don't want to know that we have 42 solutions of size 6,
    # but rather that these are 42 of size (n+1)+m (where m=0 is the best).
    myKeys = set(map(lambda v: v - len(f1.inputs) - 1, sr))
    return myKeys, resultMap


def faustins_mechanism(f, h):
    return f.satisfy_mcdc(h)


def run_experiment(_maxRounds, hs, tcas, tcas_num_cond, mechanism):
    global maxRounds
    resultMapx = None  # Python doesn't do functional `reduce()` below, but destructive:

    def red(acc, val):
        resultMapx.append(val[1])
        return acc.union(val[0])

    wall_clock_list = []
    plot_data = []
    allKeys = set()
    # Generating permutations is faster than I thought.
    maxRounds = _maxRounds
    perms = list(map(gen_perm, map(lambda f: len(f.inputs), tcas)))
    # Permutations are deterministic due to the fixed seed between runs.
    for (hi, h) in enumerate(hs):
        thread_time = Manager().list()
        resultMapx = []  # reset

        assert sys.version_info >= (3, 7), "Python 3.7 required for execution time measuring"
        # time.time() measures wall clock time
        # In Python 2.7, time.clock() measures process time in seconds (float)
        # In Python >= 3.7, time.process_time_ns() measures process time in nanoseconds (int)
        startTime = time.process_time_ns()
        startClock = time.monotonic()

        # Design considerations: we don't want to depend on the content of `tcasii.tcas`, but take this as a parameter.
        # BDDs are not picklable and hence unsuitable for MP, so we reconstitute BDDs via strings in pool members.
        # We only do that once and create a pool that we reuse multiple times.
        num_proc = 4
        with Pool(num_proc, initializer=init_globals,
                  initargs=(_maxRounds, list(map(lambda x: str(bdd2expr(x)), tcas)), tcas_num_cond, mechanism)) as p:
            # BDDs ain't picklable, so we pass the index, not the BDD:
            # TODO: isn't that problem now solved with the "global pool"?
            # Pass our pool further down:
            ps = map(process_one, zip(enumerate(zip(tcas, tcas_num_cond)), perms, repeat(h), repeat(p), repeat(thread_time)))
            # Reduce inside `with` to force evaluation of `map` above, otherwise MP will fail:
            allKeys = reduce(red, ps, set())
        resultMap = dict(zip(tcas, resultMapx))
        endTime = time.process_time_ns()
        endClock = time.monotonic()
        # TODO: XXX Needs comments
        t = (endTime - startTime) + sum(thread_time)
        wall_clock = endClock - startClock
        wall_clock_list.append(wall_clock)
        print("Usedtime: %s CPU seconds" % (t/1e9), file=sys.stderr)
        print("Usedtime: %s Wall seconds" % wall_clock, file=sys.stderr)
        plot_data.append((hi, resultMap))
        # plot_data and wall_clock_list must have the same length
        assert len(wall_clock_list) == len(plot_data)
    return allKeys, plot_data, wall_clock_list


if __name__ == "__main__":
    try:
        maxRounds = int(sys.argv[1])
    except IndexError:
        maxRounds = 42

    RNGseed = 42
    # XXX Oh wow, it's even worse; MP uses a global random state?!
    #     https://github.com/numpy/numpy/issues/9650#issuecomment-327144993
    seed(RNGseed)

    hs = [h1, h2]
    allKeys, plot_data, t_list = run_experiment(maxRounds, hs, tcasii.tcas, tcasii.tcas_num_cond, faustins_mechanism)

    # plot_data and wall_clock_list must have the same length
    assert len(t_list) == len(plot_data)

    # TODO: Pity, for now you'll have to wait again for plotting.
    # Probably we could sneak in a callback again if we really need it.
    for (hi, resultMap), t in zip(plot_data, t_list):
        # Gnuplot:
        chart_name = '{}.{}-{}'.format(hs[hi].__name__, RNGseed, maxRounds)

        with open('{}_resultMap.csv'.format(chart_name), 'w') as csvfile:
            result_map_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for col, rm in enumerate(resultMap.values()):
                result_map_writer.writerow([col, rm])
        plot(allKeys, chart_name, resultMap, t)
    plt.show()  # Wait for windows
