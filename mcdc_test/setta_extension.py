# -*- coding: utf-8 -*-
# apt-get install gcc
# apt-get install make
# apt-get install libgmp3-dev
# pip install pysmt
# pysmt-install --check
# pysmt-install --msat

from pysmt.shortcuts import Symbol, LE, GE, Int, And, Equals, Plus, Solver, serialize, Not
from pysmt.typing import INT, BOOL
from pysmt.parsing import parse
from pathsearch import run_one_pathsearch, LongestMayMerge
from pyeda.boolalg.bdd import expr2bdd


def solve(eq, reuse_h, rng):
    # type: (str) -> list

    def preprocess(test, decrypt_dict):
        # type: (dict, dict) -> iter
        # test = {a0: 0, a1: 0}
        # atoms = {a0: (1 <= h), a1: (h <= 10)}
        # new_test = {Not(1 <= h), Not(h <= 10)}

        for key, val in test.items():
            if val:
                yield decrypt_dict[key]
            else:
                yield Not[decrypt_dict[key]]

    def sat_solve(test, variables):
        # type: (iter, iter) -> dict
        # atoms = iter of pysmt.fnode.FNode = {1 <= h, h >= 10, ...}
        solution = dict()
        with Solver(logic="QF_LIA") as solver:
            for atom in test:
                solver.add_assertion(atom)
            if not solver.solve():
                print("Domain is not SAT!!!")
                exit()
            if solver.solve():
                for v in variables:
                    solution[v] = solver.get_value(v)
            else:
                print("No solution found")
        # solution = {h: 10, ....}
        return solution

    # Replacement of integer expressions in equation by boolean expressions:
    # Before:
    #   equation = (1 <= h) & (10 >= h)
    # After:
    #   equation = a0 & a1

    # equation = "(1 <= h) & (10 >= h)".lower()
    equation = eq.lower()
    eq_variables = set(v for v in equation if v.isalpha())
    int_variables = set(Symbol(s, INT) for s in eq_variables)

    # Equation
    # formula = (1 <= h) & (10 >= h)
    formula = parse(equation)

    # Replace atoms by symbolic variables.
    # E.g.:
    #  - "(1 <= h)" by "a0"
    #  - "(10 >= h)" by "a1"

    atoms = formula.get_atoms()
    sorted_atoms = sorted(atoms)
    bool_variables = set(Symbol("a" + str(i), BOOL) for (i, a) in enumerate(sorted_atoms))

    # Encrypt
    # formula           = (1 <= h) & (h <= 10)
    # abstract_formula  = a0 & a1
    encrypt_dict = dict(zip(bool_variables, atoms))
    abstract_formula = formula.substitute(encrypt_dict)

    # Convert formula to BDD (pyeda) format
    f = expr2bdd(abstract_formula.serialize())

    # Call to SETTA method / Compute the test case
    # allKeys, plot_data, t_list = run_experiment((maxRounds, rngRounds), hs, tcasii.tcas, tcasii.tcas_num_cond, run_one_pathsearch)
    test_case_pairs, num_test_cases, uniq_test = run_one_pathsearch(f, reuse_h, rng)

    # Decrypt
    # abstract_formula  = a0 & a1
    # fp           = (1 <= h) & (h <= 10)
    decrypt_dict = dict(zip(atoms, bool_variables))
    # fp = abstract_formula.substitute(decrypt_dict)

    # Map atoms to tests
    # unique_tests = [{a0: 0, a1: 0}, {a0: 1, a1: 1}, ...]
    solutions = []
    for test in uniq_test:
        # test = {a0: 0, a1: 0}
        test = preprocess(test, decrypt_dict)
        # test = {Not(1 <= h), Not(h <= 10)}
        solutions += sat_solve(test, formula.get_free_variables())

    return solutions


if __name__ == "__main__":
    eq = "(1 <= h) & (10 >= h)"
    rng = 10
    solutions = solve(eq, LongestMayMerge, rng)

    for test in solutions:
        print("Test\n")
        for var, value in test.items():
            print("{0}: {1}".format(var, value))
