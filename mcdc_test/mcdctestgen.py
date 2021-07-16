import logging
import csv
import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager
from math import factorial
from random import randint, seed, Random
import sys
import time
from pyeda.boolalg.bdd import expr2bdd, bdd2expr, _iter_all_paths, _path2point, BDDNode, BDDNODEZERO, BDDNODEONE, BDDVariable, BinaryDecisionDiagram
from pyeda.boolalg.expr import expr
from mcdc_helpers import *
from sortedcontainers import SortedList
from pyeda.inter import bddvars
from functools import reduce
from itertools import permutations, repeat, product, chain, tee

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
rngRounds = None
tcas = None
tcas_num_cond = None


# noinspection PyUnreachableCode
def no_mechanism(_f, _h):
    assert False, "MP maybe not initialized correctly?"
    exit(1)


mechanism = no_mechanism


def path_via_node(fr, vc, to, conditions):
    # type: (BDDNode, BDDNode, BDDNode, iter) -> list
    # list of paths from the root to terminal node via intermediate node vc

    # temp_list_of_paths = list(_iter_all_paths(fr, to))
    # list_of_paths = [path for path in temp_list_of_paths if vc in path]

    list_of_paths = [uniformize(_path2point(path), conditions)
                     for path in _iter_all_paths(fr, to) if vc in path]
    return list_of_paths


def equal(bddnode, condition):
    # type: (BDDNode, BDDVariable) -> bool
    return bddnode.root == condition.uniqid


def satisfy_mcdc(f, heuristic, _rng):
    # type: (BinaryDecisionDiagram, callable) -> (dict, int, list)

    def select_paths_bdd(f):
        # type: (BinaryDecisionDiagram) -> dict

        # Logging configuration
        # TODO: doesn't really work with MP (of course it doesn't) :-/
        logger = logging.getLogger("MCDC")
        # logger.setLevel(logging.DEBUG)

        # set that will store the paths selected from the BDD in tuple format,
        # i.e., psi = set(elem_1, elem_2,...)
        # where:
        # elem_1 != elem_2, and
        # elem_i = (bddnode_1, bddnode_2, ....)
        # psi = set()

        # Dictionary that translates a path in tuple format:
        # path = (bddnode_1, ..., bddnode_n)
        # into:
        # path = {a: 1, ..., c: None}
        # The use of a dictionary guarantees singleton references to the same path, which is useful when instantiating
        # None values in the path (e.g., c: None) and the path is shared by several case of studies
        tuple_2_dict = dict()

        # test_case is a dictionary of pairs
        test_case = dict()

        # Format:
        # test_case[cond] = (path_zero, path_one)
        # where:
        # c is the condition, and
        # path_zero/path_one go to terminal node 0/1 in the BDD respectively

        root = f.node
        bdd_nodes = list(f.dfs_preorder())

        # TODO: could be adjustable.
        conditions = sorted(f.support, key=lambda c: c.uniqid)
        for c in conditions:
            logger.debug("\n")
            logger.debug("current variable: {0}".format(c))
            logger.debug("______________________________________")

            paths_to_zero = []
            paths_to_one = []

            c_nodes = [node for node in bdd_nodes if equal(node, c)]
            # TODO: Could be interesting to make the following loop a generator.
            # for vc in c_nodes:
            #     # Paths to terminal nodes that are sorted by length
            #     # Note: Search for the shortest paths_to_zero & paths_to_one for all c_nodes
            #     # If several paths have the same length, then choose the one with highest reuse factor
            #     # Paths start from the root node in the BDD, and they must explicitly cross node vc (which has cond c)
            #     paths_to_zero += path_via_node(root, vc, BDDNODEZERO, conditions)
            #     paths_to_one += path_via_node(root, vc, BDDNODEONE, conditions)

            # WARNING: iterators are exhausted once they are read
            paths_to_zero, paths_to_zero_log = tee(chain.from_iterable(path_via_node(root, vc, BDDNODEZERO, conditions) for vc in c_nodes))
            paths_to_one, paths_to_one_log = tee(chain.from_iterable(path_via_node(root, vc, BDDNODEONE, conditions) for vc in c_nodes))

            logger.debug("paths_to_zero:\t{0}".format(paths_to_zero_log))
            logger.debug("paths_to_one:\t{0}".format(paths_to_one_log))

            # paths_to_zero = {P0, P1}, and
            # paths_to_one = {P2, P3} with
            # P0 = (bddnode_1, ..., bddnode_n),
            # P1 = (bddnode_1, ..., bddnode_m), and n != m

            logger.debug("sorting...")
            fcp_list = heuristic(test_case, c, paths_to_zero, paths_to_one)
            logger.debug("# paths differing in condition {0}: {1}\n".format(c, len(fcp_list)))
            # logger.debug(list(fcp_list))

            if len(fcp_list) == 0:
                print("Warning: suspecting masking(?)")
                assert False
            else:
                # Choose the first pair = (path_zero, path_one) that satisfy the restrictions, if any
                (path_zero, path_one) = fcp_list[0]

                # Assign test cases path_zero and path_one to condition c
                p0, p1 = Path(dict(path_zero)), Path(dict(path_one))
                # TODO: seems to be missing a merge here; needs a variation on
                #  p0 = merge(p0,p1), p1 = merge(p1,p0)
                test_case[c] = (p0, p1)
                logger.debug("test_case[{0}] = ({1})".format(c, test_case[c]))

                # continue with the next condition

        # TODO: If code breaks, indent code until here

        # test_case[a] = ({a: 0, b: None, c: 0}, {a: 1, b: 1, c: None})
        # test_case[b] = ({a: 1, b: 0, c: 0}, {a: 1, b: 1, c: None})
        # test_case[c] = ({a: 0, b: None, c: 0}, {a: 0, b: None, c: 1})

        return test_case

    # First loop
    # Select pair of paths p0/p1 from the BDD
    # test_case, tuple_2_dict = select_paths_bdd(self)

    test_case = select_paths_bdd(f)

    # psi = compact_truth_table(test_case, tuple_2_dict)
    # psi[p0] = {a, c}, where tuple_2_dict(p0) = {a: 0, b: None, c: 0}
    # psi[p1] = {b},    where tuple_2_dict(p1) = {a: 1, b: 1, c: None}

    # Second loop
    # Instantiate '?' for each path in test_case
    test_case = instantiate(test_case)
    replace_final_question_marks(test_case)
    uniq_test = unique_tests(test_case)
    # TODO: num_test_cases is kind of redundant
    num_test_cases = len(uniq_test)
    return test_case, num_test_cases, uniq_test


def init_globals(_runs, _tcas, _tcas_num_cond, _mechanism):
    # type: (int, dict, int, callable) -> None
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
    # type: (int) -> list
    have = []
    while len(have) < l:
        j = randint(0, l - 1)
        if j not in have:
            have.append(j)
    return have


def gen_perm(l):
    # type: (int) -> list
    global maxRounds
    return gen_perm_max(maxRounds, l)


def gen_perm_max(maxRounds, l):
    # type: (int, int) -> list
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
def calc_reuse(path, test_case_pairs):
    # type: (dict, dict) -> int
    # for p in test_case.values():
    #   print("reuse:\t{0}".format(p))
    # tcs = map(lambda p: (merge_Maybe(conditions,path,p[0]),merge_Maybe(conditions, path, p[1])), test_case.values())
    tcs = filter(lambda p: p[0] == path or p[1] == path, test_case_pairs.values())
    return len(list(tcs))


def calc_may_reuse(path, test_case_pairs):
    # type: (dict, dict) -> int
    tcs = filter(lambda p: merge_Maybe_except_c_bool(None, path, p[0]) is not None
                        or merge_Maybe_except_c_bool(None, path, p[1]) is not None, test_case_pairs.values())
    return len(list(tcs))


def hi_reuse_short_path(tcs, c, paths_to_zero, paths_to_one):
    # type: (dict, BDDVariable, list, list) -> list
    cartesian_product = product(paths_to_zero, paths_to_one)

    # Choose path_zero and path_one that only differs on condition c
    paths = ((path_zero, path_one) for (path_zero, path_one) in cartesian_product
             if xor(path_zero, path_one, c))
    return SortedList(paths, key=lambda path: (-calc_reuse(path[0], tcs) - calc_reuse(path[1], tcs),
            # highest reuse/shortest path
            size(path[0]) + size(path[1])))


def hi_reuse_long_path(tcs, c, paths_to_zero, paths_to_one):
    # type: (dict, BDDVariable, list, list) -> list
    cartesian_product = product(paths_to_zero, paths_to_one)

    # Choose path_zero and path_one that only differs on condition c
    paths = ((path_zero, path_one) for (path_zero, path_one) in cartesian_product
             if xor(path_zero, path_one, c))
    return SortedList(paths, key=lambda path: (-calc_reuse(path[0], tcs) - calc_reuse(path[1], tcs),
                                               # highest reuse/longest path
                                               -size(path[0]) - size(path[1])))


def hi_reuse_long_merged_path(tcs, c, paths_to_zero, paths_to_one):
    # type: (dict, BDDVariable, list, list) -> list
    cartesian_product = product(paths_to_zero, paths_to_one)

    # Choose path_zero and path_one that only differs on condition c
    paths = ((path_zero, path_one) for (path_zero, path_one) in cartesian_product
             if xor(path_zero, path_one, c))
    return SortedList(paths, key=lambda path: (-calc_may_reuse(path[0], tcs) - calc_may_reuse(path[1], tcs),
                                               # highest reuse/longest path
                                               -size(path[0]) - size(path[1])))


def h3(tcs, c, paths_to_zero, paths_to_one):
    # type: (dict, BDDVariable, list, list) -> list
    # Not "very good", e.g. can't find optimal solution in general, here D3:
    #    Num_Cond=7: min=8, |p|=5040, [(9, 5040)]
    cartesian_product = product(paths_to_zero, paths_to_one)

    # Choose path_zero and path_one that only differs on condition c
    paths = ((path_zero, path_one) for (path_zero, path_one) in cartesian_product
             if xor(path_zero, path_one, c))
    return [paths.__next__()]


def h3s(tcs, c, paths_to_zero, paths_to_one):
    # type: (dict, BDDVariable, list, list) -> list
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
    # type: (dict, BDDVariable, list, list) -> list
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
    # type: (dict, BDDVariable, list, list) -> list
    # highest reuse/longest path
    # Still fails D3
    paths_to_zero = SortedList(paths_to_zero, key=lambda path: (-calc_reuse(path, tcs), -size(path)))
    paths_to_one = SortedList(paths_to_one, key=lambda path: (-calc_reuse(path, tcs), -size(path)))
    cartesian_product = product(paths_to_zero, paths_to_one)

    # Choose path_zero and path_one that only differs on condition c
    paths = filter(lambda p: xor(p[0], p[1], c), cartesian_product)
    return [paths.__next__()]


def processFP_with_time(args):
    # type: (list) -> int
    tic = time.process_time_ns()
    i = args[0]
    p = args[1]
    h = args[2]
    thread_time = args[3]
    rng = Random(args[4])
    value = processFP(i, p, h, rng)
    toc = time.process_time_ns()
    elapsed_time = toc - tic
    thread_time.append(elapsed_time)
    return value


def paths_from_pair_is_reused(tcs, pair):
    # type: (dict, (dict, dict)) -> bool
    """Each path occurs only once, exactly for the pair we're looking at,
       hence sum of reuse is larger than 2 (1 each).
    """
    r0 = calc_reuse(pair[0], tcs)
    r1 = calc_reuse(pair[1], tcs)
    return r0 + r1 > 2


def processFP(i, p, heuristic, rng):
    # type: () -> int
    global mechanism
    f1 = tcas[i]  # Back into the global data (MP)
    fresh_var = 'f'  # apparently there's something weird going on if this name is used before, eg. in tcasii
    assert fresh_var+"[0]" not in map(lambda x: str(x), f1.inputs)
    X = bddvars(fresh_var, len(f1.inputs))  # Let's hope this ain't too expensive.
    theMap = {sorted(f1.inputs, key=lambda c: c.uniqid)[t]: X[p[t]] for t in range(len(f1.inputs))}
    f2 = f1.compose(theMap)
    test_case, num_test_cases, uniq_test = mechanism(f2, heuristic, rng)
    is_mcdc = test_mcdc(f2, test_case)
    assert is_mcdc
    # print('Round: {0} Number of Conditions: {1} Number of TCs: {2}'.format(RoundN, len(f1.inputs), num_test_cases))
    assert len(f1.inputs) == len(f2.inputs)
    if num_test_cases > len(f1.inputs)+1:
        # Check if first condition was ever reused:
        fs = sorted(f2.inputs, key=lambda c: c.uniqid)
        first_cond = fs[0]
        if not paths_from_pair_is_reused(test_case, test_case[first_cond]):
            tcs_s = [(lrlr(fs, test_case[c][0]), lrlr(fs, test_case[c][1])) for c in fs]
            # TODO: doesn't really work within MP :-/
            logger = logging.getLogger(__name__)
            logger.debug("Failed to reuse root node (h:"+str(heuristic)+":\n"+str(i)+'/'+str(num_test_cases)+':'+str(bdd2expr(f2))+'\n'+str(tcs_s))
            # assert False

    if num_test_cases <= len(f1.inputs) or not is_mcdc:
        # inv_map = {v: k for k, v in theMap.items()}
        num_test_cases = -1  # indicate that this set is useless
        # TODO: I think this disappears in the final result?
        # Not a problem since we show % in relation to maxRounds.
    return num_test_cases


def process_one(arg):
    # type: ((int, (BinaryDecisionDiagram, int), int, Pool, list)) -> (set, list)
    # enumerate(zip(tcas, tcas_num_cond)), perms, repeat(h), repeat(p), repeat(thread_time))
    results = {}
    i, (f1, num_cond) = (arg[0][0], (arg[0][1][0], arg[0][1][1]))
    # Pity that with mp we can't share/reuse perms without more complications.
    # TODO: Actually, after the "global pool" refactoring, we probably can again!
    perms = arg[1]
    h = arg[2]
    pool = arg[3]
    # thread_time = arg[4]
    thread_time = Manager().list()
    ntcs = []
    # We run several rounds for each permutation, using positions from the first permutation as seed.
    # Blows numbers a bit out of the water though...
    for p in perms[0][:rngRounds]:
        rng_seed = p
        ntcs += pool.map(processFP_with_time, zip(repeat(i), perms, repeat(h), repeat(thread_time), repeat(rng_seed)))
    for num_test_cases in ntcs:
        try:
            old = results[num_test_cases]
        except KeyError:
            old = 0
        results[num_test_cases] = old + 1
    colour = ''
    # We keep these around for the CSV:
    sr = sorted(results)
    resultMap = [(k, results[k]) for k in sr]

    if len(resultMap) == 0 or len(f1.inputs) + 1 < resultMap[0][0]:
        # We didn't find any n+1 solution.
        colour = bcolors.FAIL
    print('{}Num_Cond={}: min={}, t_avg={} |p|={}, {}'.format(colour, num_cond, len(f1.inputs) + 1
                                                              , (sum(thread_time)/(len(perms)*rngRounds))/1e9
                                                              , factorial(len(f1.inputs))
                                                              , resultMap), bcolors.ENDC)
    arg[4].extend(thread_time)  # report time up the chain
    # TODO: I don't think we need this print statement below in general, both values are included in the line above.
    if num_cond != len(f1.inputs):
        print(bcolors.WARNING + 'Decision with Masked conditions: D_' + str(i + 1), bcolors.ENDC)
    # TODO: could (should?) be done outside? OTOH, doesn't really matter
    # "Normalize" result; we don't want to know that we have 42 solutions of size 6,
    # but rather that these are 42 of size (n+1)+m (where m=0 is the best).
    myKeys = set(map(lambda v: v - len(f1.inputs) - 1, sr))
    return myKeys, resultMap


def run_experiment(rounds_config, hs, tcas, tcas_num_cond, mechanism):
    # type: ((int, int), dict, dict, int, callable) -> (set, List[Tuple[int, dict]], List[float])
    global maxRounds
    global rngRounds
    resultMapx = None  # Python doesn't do functional `reduce()` below, but destructive:

    def red(acc, val):
        # type: (set, (set, list)) -> set
        resultMapx.append(val[1])
        return acc.union(val[0])

    wall_clock_list = []
    plot_data = []
    allKeys = set()
    # Generating permutations is faster than I thought.
    (maxRounds, rngRounds) = rounds_config
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
                  initargs=(maxRounds, list(map(lambda x: str(bdd2expr(x)), tcas)), tcas_num_cond, mechanism)) as p:
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
    try:
        rngRounds = int(sys.argv[2])
    except IndexError:
        rngRounds = 3


    RNGseed = 42
    # XXX Oh wow, it's even worse; MP uses a global random state?!
    #     https://github.com/numpy/numpy/issues/9650#issuecomment-327144993
    seed(RNGseed)

    hs = [hi_reuse_short_path, hi_reuse_long_path]
    allKeys, plot_data, t_list = run_experiment((maxRounds, rngRounds), hs, tcasii.tcas, tcasii.tcas_num_cond, satisfy_mcdc)

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
