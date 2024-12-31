import re
from typing import NamedTuple, Sequence, Tuple

"""
below is an example input:
    \[
        x^11
        - x^10
        + (30j[0]^2 -30[0] - 30) x^8
        + (-30[1]^5 - 30j[1]^3 + 30j[1]^2 - 30j[1] + 30) x^6
        + (30j[2]^2 + 30j[2] - 30) x^5
    \]

will assign a flattened version to a string for simplicity.

We want to parse this into the below data structures:
"""


class Coefficient(NamedTuple):
    constant: float  # part to multiply the t by, if no t then multiply by 1
    t: Tuple[int, int] | None  # index into and power of the varying or static t


class Term(NamedTuple):
    degree: int  # degree of this term, aka what x^n it is multiplied with
    coefficient_parts: Sequence[Coefficient]  # list of all parts to the coefficient, together


class Equation(NamedTuple):
    terms: Sequence[Term]  # sequence of all the terms, the max degree among all terms is the degree of the polynomial


equation_str = "x^11 - x^10 + (30j[0]^2 -30[0] - 30) x^8 + (-30[1]^5 - 30j[1]^3 + 30j[1]^2 - 30j[1] + 30) x^6 + (30j[2]^2 + 30j[2] - 30) x^5"

# TODO(afmck): these regexes are likely not robust, but good enough for now
# TODO(afmck): could probably get them all in one pass too, but let's not bother
all_term_str = re.findall(r"[-+]?\s?\(?.*?\)?x\^\d+", equation_str)
all_term_str = [x.lstrip() for x in all_term_str]
all_term_str = ["+ " + x if x[0] not in "+-" else x for x in all_term_str]

print(equation_str)
print()

find_degree_re = re.compile(r"x\^(?P<degree>\d+)")
find_coeff_parts_re = re.compile(r"\s?[+-]?\s?\d+?j?(?:\[\d+\])(?:\^\d+)?")

equation = Equation(terms=[])

for term_str in all_term_str:
    print(term_str)
    # term_degree = find_degree_re.findall(term_str)
    # term = Term(degree=term_degree[0], coefficient_parts=[])
    # print(term)

    coeff_parts = find_coeff_parts_re.findall(term_str)
    print(coeff_parts)
