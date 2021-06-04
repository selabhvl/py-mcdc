# The decisions considered here are the Traffic Alert and Collision Avoidance System (TCAS II) benchmarks used in avionics.
# The were presented in [https://doi.org/10.1002/qre.1934, https://link.springer.com/chapter/10.1007/978-3-319-99130-6_9, https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=286420,https://dl.acm.org/doi/pdf/10.1145/3167132.3167335 ]

from more_itertools import partition

from pyeda.boolalg.bdd import bddvar, expr2bdd, bdd2expr, BinaryDecisionDiagram, BDDZERO, BDDONE
from pyeda.boolalg.expr import expr
from pyeda.boolalg.boolfunc import *
from sortedcontainers import SortedDict
from pyeda.inter import *
import logging

a, b, c, d, e, f, g, h, i, j, k, l, m, n = map(bddvar, 'abcdefghijklmn')
D1 = a & (~b | ~c) & d | e
D2 = ~(a & b) & ((d & ~e & ~f) | (~d & e & ~f) | ~d & ~e & ~f) & (
            (a & c & (d | e) & h) | (a & (d | e) & ~h) | (b & (e | f)))
D3 = ~(c & d) & (~e & f & ~g & ~a & (b & c | ~b & d))
D4 = a & c & (d | e) & h | a & (d | e) & ~h | b & (e | f)
D5 = ~e & f & ~g & ~a & (b & c | ~b & d)
D6 = (~a & b | a & ~b) & ~(c & d) & ~(g & h) & ((a & c | b & d) & e & (f & g | ~f & h))
D7 = (a & c | b & d) & e & (f & g | ~f & h)
D8 = (a & ((c | d | e) & g | a & f | c & (f | g | h | i)) | (a | b) & (c | d | e) & i) & ~(a & b) & ~(c & d) & ~(
            c & e) & ~(d & e) & ~(f & g) & ~(f & h) & ~(f & i) & ~(g & h) & ~(h & i)
D9 = a & (~b | ~c | b & c & ~(~f & g & h & ~i | ~g & h & i) & ~(~f & g & l & k | ~g & ~i & k)) | f
D10 = a & ((c | d | e) & g | a & f | c & (f | g | h | i)) & (a | b) & (c | d | e) & i
D11 = (~a & b | a & ~b) & ~(c & d) & ~(g & h) & ~(j & k) & (a & c | b & d) & e & (i | ~g & ~k | ~j & (~h | ~k))
D12 = (a & c | b & d) & e & (i | ~g & ~k | ~j & (~h | ~k)) & (a & c | b & d) & e & (i | ~g & ~k | ~j & (~h | ~k))
D13 = (~a & b | a & ~b) & ~(c & d) & (f & ~g & ~h | ~f & g & ~h | ~f & ~g & ~h) & (~(j & k)) & (
            (a & c | b & d) & e & (f | (i & (g & j | h & k))))
D14 = (a & c | b & d) & e & (f | (i & (g & j | h & k)))
D15 = (a & (~d | ~e | d & e & ~(~f & g & h & ~i | ~g & h & i) & ~(~f & g & l & k | ~g & ~i & k)) | ~(
            ~f & g & h & ~i | ~g & h & i) & ~(~f & g & l & k | ~g & ~i & k) & (b | c & ~m | f)) & (
                  a & ~b & ~c | ~a & b & ~c | ~a & ~b & c)
D16 = a | b | c | ~c & ~d & e & f & ~g & ~h | i & (j | k) & l
D17 = a & (~d | ~e | d & e & ~(~f & g & h & ~i | ~g & h & i) & ~(~f & g & l & k | ~g & ~i & ~k)) | ~(
            f & g & h & ~i | ~g & h & i) & ~(~f & g & l & k | ~g & ~i & k) & (b | c & ~m | f)
D18 = a & ~b & ~c & ~d & ~e & f & (g | ~g & (h | i)) & ~(j & k | ~j & l | m)
D19 = a & ~b & ~c & (~f & (g | ~f & (h | i))) | f & (g | ~g & (h | i) & ~d & ~e) & ~(j & k | ~j & l & ~m)
D20 = a & ~b & ~c & (~f & (g | ~g & (h | i))) & (~e & ~n | d) | ~n & (j & k | ~j & l & ~m)

def makeLarge(f):
    l = len(f.inputs)
    vars = list(f.inputs)
    X = bddvars('X', l)
    Y = bddvars('Y', l)
    Z = bddvars('Z', l)
    fx = f.compose({vars[i]: X[i] for i in range(l)})
    # fy = f.compose({vars[i]: Y[i] for i in range(l)})
    # fz = f.compose({vars[i]: Z[i] for i in range(l)})
    Dlarge = f & fx # & fy & fz
    return Dlarge

# D2 = "~(a & b) & ((d & ~e & ~f) | (~d & e & ~f) | ~d & ~e & ~f) & ((a & c & (d | e) & h) | (a & (d | e) & ~h) | (b & (e | f)))"
# D4 = "a & c & (d | e) & h | a & (d | e) & ~h | b & (e | f)"
# D6 = "(~a & b | a & ~b) & ~(c & d) & ~(g & h) & ((a & c | b & d) & e & (f & g | ~f & h))"
# D8 = "(a & ((c | d | e) & g | a & f | c & (f | g | h | i)) | (a | b) & (c | d | e) & i) & ~(a & b) & ~(c & d)& ~(c & e) & ~(d & e) & ~(f & g) & ~(f & h) & ~(f & i) & ~(g & h) & ~(h & i)"
# D9 = "a & (~b | ~c | b & c & ~(~f & g & h & ~i | ~g & h & i) & ~(~f & g & l & k | ~g & ~i & k)) | f"
# D11 = "(~a & b | a & ~b) & ~(c&d) & ~(g&h) & ~(j & k)&(a & c | b & d) & e & (i | ~g & ~k | ~j & (~h | ~k))"
# D12 = "(a & c | b & d) & e & (i | ~g & ~k | ~j & (~h | ~k)) & (a & c | b & d) & e & (i | ~g & ~k | ~j & (~h | ~k))"
# D13 = "(~a & b | a & ~b) & ~(c & d) & (f & ~g & ~h | ~f & g & ~h | ~f & ~g & ~h) & (~(j & k)) & ((a & c | b & d) & e & (f | (i & (g & j | h & k))))"
# D15 = "(a & (~d | ~e | d & e & ~(~f & g & h & ~i | ~g & h & i) & ~(~f & g & l & k | ~g & ~i & k)) | ~(~f & g & h & ~i | ~g & h & i)& ~(~f & g & l & k | ~g & ~i & k) & (b | c & ~m | f)) & (a & ~b & ~c | ~a & b & ~c | ~a & ~b & c)"
# D17 = "a & (~d | ~e | d & e & ~(~f & g & h & ~i | ~g & h & i) & ~(~f & g & l & k | ~g & ~i & ~k)) | ~(f & g & h & ~i | ~g & h & i) & ~( ~f & g & l & k | ~g & ~i & k) & (b | c & ~m | f)"

# TODO: example for "larger" formula
# These take too long with sorting:
# tcas = [makeLarge(D15), makeLarge(D17), makeLarge(D19), makeLarge(D20)]
tcas = [D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15, D16, D17, D18, D19, D20]
# tcas_dict = {expr2bdd(expr(D_i)): D_i for D_i in tcas}
# tcas_dict_ordered = SortedDict({D_i: expr2bdd(expr(D_i)) for D_i in tcas})

tcas1 = [D2, D8, D9, D13, D15, D17, D19]  # failed the first test
tcas2 = [D6, D11, D19]  # passed with Order : d, e, f, g, h, i, j, k, l, m, n, a, b, c = map(bddvar, 'defghijklmnabc')
tcas3 = [D11, D12, D13]  # passed with Order : i, j, k, l, m, n, a, b, c, d, e, f, g, h = map(bddvar, 'ijklmnabcdefgh')
tcas4 = [D4, D6, D9,
         D11]  # passed with Order : f, i, j, k, l, m, n, a, b, c, d, e, g, h = map(bddvar, 'fijklmnabcdegh') and In general 7 failed, 13 passed in 0.75s
tcas5 = [D4, D9, D12]  # passed with Order : a, d, c, e, i, j, b, f, g, m, n, k, h, l = map(bddvar, 'adceijbfgmnkhl')
tcas6 = [D15, D17]  #

tcas_names = ["D"+str(i) for i in range(1, 21)]
tcas_num_cond = [5, 7, 7, 7, 7, 8, 8, 9, 9, 9, 10, 10, 11, 11, 12, 12, 12, 13, 13, 14]
# tcas_dict_name = {ds_i: value for ds_i, value in zip(tcas_names,  tcas_num_cond)}
tcas_dict = dict(zip(tcas, tcas_num_cond))

# print(D4.support, D4.top, D4.usupport, D4.inputs)
# gv = Source(D4.to_dot())
# gv.render('D4fijklmnabcdegh', view=True)
# print(RoundN, D_i, num_test_cases)
# Order : abcdefghijklmn
# [D1, D2, D3, D5, D7, D8, D10, D14, D16, D18, D20]

# Order : defghijklmnabc
# [D6, D11, D19]
# Order : ijklmnabcdefgh
# [D12, D13]
# Order : fijklmnabcdegh
# [D4]
# Order : adceijlbfghmnk
# [D9]


# Logging configuration
logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def test_to_str(test):
    # type: (dict) -> str
    # print(sorted(test))
    # return "".join(str(c) for c in test.values())
    return "".join(str(test[c]) if test[c] is not None else '?' for c in sorted(test))


def printer(test_case, is_mcdc, num_test_cases, uniq_test):
    # type: (dict, bool, int, list) -> None
    n = len(test_case)
    print("is MCDC?: {0}".format(is_mcdc))
    print("#conditions: {0}".format(n))
    print("#test_cases: {0}".format(num_test_cases))
    print("#test_cases-(#n+1): {0}".format(num_test_cases - (n + 1)))
    print("test_cases:")

    s = "|" + "|".join(str(i) for i in sorted(test_case.keys())) + "|"

    print(s)
    print("_" * len(s))
    for cond in sorted(test_case):
        p0, p1 = test_case[cond]
        # s0 = "".join(str(e) for e in p0.values())
        # s1 = "".join(str(e) for e in p1.values())
        s0 = test_to_str(p0)
        s1 = test_to_str(p1)

        print("|{1}|0|\t\t({0})".format(cond, s0))
        print("|{1}|1|\t\t({0})".format(cond, s1))

    psi = psi_gen(test_case, uniq_test)
    psi_printer(psi, uniq_test)


def psi_gen(test_case, uniq_test):
    # type: (dict, list) -> dict
    psi = {test_to_str(test): set() for test in uniq_test}
    for cond in sorted(test_case):
        (p0, p1) = test_case[cond]
        psi[test_to_str(p0)].add(cond)
        psi[test_to_str(p1)].add(cond)
    return psi


def psi_printer(psi, uniq_test):
    # type: (dict, list) -> None
    print("Psi:")
    for test in sorted(psi.keys()):
        print("|{0}| {1}".format(test, psi[test]))
    print("\n")


def test_mcdc(f, test_case):
    # type: (BinaryDecisionDiagram, dict) -> (dict, bool, int)

    # test_case[a] = ({a: 0, b: None, c: 0}, {a: 1, b: 1, c: None})
    # test_case[b] = ({a: 1, b: 0, c: 0}, {a: 1, b: 1, c: None})
    # test_case[c] = ({a: 0, b: None, c: 0}, {a: 0, b: None, c: 1})
    # where:
    # test_case[c_i] = (p0, p1) are paths to terminal node 0 and 1 from the BDD respectively

    # MCDC is satisfied if:
    # 1. all test are instantiated (i.e., None not in p0 nor in p1 for test_case[c_i] for all c_i in conditions)
    # is_mcdc = True
    # for cond in test_case:
    #     path_zero, path_one = psi[cond]
    #     is_mcdc = is_mcdc and (None not in path_zero.values())
    #     is_mcdc = is_mcdc and (None not in path_one.values())
    def check_condition_1(p0, p1):
        return (None not in p0.values()) and (None not in p1.values())
    # TODO: directly partition for better error handling?
    cond_1 = all(check_condition_1(p0, p1) for (p0, p1) in test_case.values())
    # The logic below does not work correctly if there are still "?", so let's stop here if necessary.
    assert cond_1, list(filter(lambda p: not check_condition_1(p[0], p[1]), test_case.values()))

    # 2. for condition 'a', p0, p1 in test_case then
    # p0[c_i] != p1[c_i] for c_i = 'a' and p0[c_i] == p1[c_i] for the rest of conditions
    def check_condition_2(cp):
        (c_1, (p0, p1)) = cp
        return (p0[c_1] != p1[c_1]) and all((p0[c_2] == p1[c_2]) for c_2 in test_case.keys() - {c_1})
    (notok, ok) = partition(check_condition_2, test_case.items())
    notok_l = list(notok)
    cond_2 = len(notok_l) == 0
    assert cond_2, notok_l
    # 3. for each condition c_i in test_case, (p0, p1) = test_case[c_i] and (f(p0) = 0 and f(p1) = 1)
    # TODO: fails if restrict does not produce a final result without "?"s.

    cond_3 = True
    for (p0, p1) in test_case.values():
        rp0 = f.restrict(p0)
        rp1 = f.restrict(p1)
        assert rp0 == BDDZERO or rp0 == BDDONE, "Bool expression using instance p0({0}): {1}".format(p0, rp0)
        assert rp1 == BDDZERO or rp1 == BDDONE, "Bool expression using instance p1({0}): {1}".format(p1, rp1)
        cond_3 = cond_3 and int(rp0) == 0 and int(rp1) == 1
        if not cond_3:
            assert not (int(rp0) == 1 and int(rp1) == 0), "Please construct your pairs more carefully!."
            break  # quick exit.
    if cond_1 and cond_2 and cond_3:
        is_mcdc = True
    else:
        is_mcdc = (cond_1, cond_2, cond_3)

    logger.debug("cond_1: {0}\ncond_2: {1}\ncond_3: {2}\nis_mcdc: {3}\n".format(cond_1, cond_2, cond_3, is_mcdc))
    return is_mcdc
