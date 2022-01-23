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

# Replacement of integer expressions in equation by boolean expressions:
# Before:
#   equation = (1 <= h) & (10 >= h)
# After:
#   equation = A & B
equation = "(1 <= h) & (10 >= h)".lower()
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
form_2_abstr = dict(zip(bool_variables, atoms))
# formula           = (1 <= h) & (h <= 10)
# abstract_formula  = a0 & a1
abstract_formula = formula.substitute(form_2_abstr)

# Call to SETTA method

abstr_2_form = dict(zip(atoms, bool_variables))
fp = abstract_formula.substitute(abstr_2_form)
# abstract_formula  = a0 & a1
# fp           = (1 <= h) & (h <= 10)

print(formula)
print(fp)
print(abstract_formula)

with Solver(logic="QF_LIA") as solver:
    solver.add_assertion(formula)
    if not solver.solve():
        print("Domain is not SAT!!!")
        exit()
    if solver.solve():
        for l in int_variables:
            print("%s = %s" %(l, solver.get_value(l)))
    else:
        print("No solution found")

print(sorted_atoms[0])
print(Not(sorted_atoms[0]))

with Solver(logic="QF_LIA") as solver:
    solver.add_assertion(Not(sorted_atoms[0]))
    if not solver.solve():
        print("Domain is not SAT!!!")
        exit()
    if solver.solve():
        for l in int_variables:
            print("%s = %s" %(l, solver.get_value(l)))
    else:
        print("No solution found")