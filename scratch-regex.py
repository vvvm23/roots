import ast
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
print(equation_str)
print()

# TODO(afmck): these regexes are likely not robust, but good enough for now
# TODO(afmck): could probably get them all in one pass too, but let's not bother
all_term_str = re.findall(r"[-+]?\s?\(?.*?\)?x\^\d+", equation_str)
all_term_str = [x.lstrip() for x in all_term_str]
all_term_str = ["+ " + x if x[0] not in "+-" else x for x in all_term_str]

# TODO: can we simplify these regexes? I am a noob
find_degree_re = re.compile(r"(\s?[-+]?\s?)x\^(?P<degree>\d+)")
find_coeff_parts_re = re.compile(r"[^\^x]\s?[+-]?\s?[^\^]\d+j?(?:(?:\[\d+\])(?:\^\d+)?)?")
parse_coeff_part_re = re.compile(r"(?P<constant>[+-]?\d+j?)?(?:\[(?P<index>\d+)\])?(?:\^(?P<power>\d+))?$")

equation = Equation(terms=[])

for term_str in all_term_str:
    maybe_leading_sign, term_degree = find_degree_re.findall(term_str)[0]
    maybe_leading_sign = re.sub(r"\s", "", maybe_leading_sign)

    leading_sign = 1
    if maybe_leading_sign == "-":
        leading_sign = -1

    term = Term(degree=int(term_degree), coefficient_parts=[])
    coeff_parts = find_coeff_parts_re.findall(term_str)

    if len(coeff_parts) == 0 and leading_sign == -1:
        term.coefficient_parts.append(Coefficient(constant=-1, t=None))

    for coeff_part in coeff_parts:
        coeff_part = re.sub(r"[()\s]", "", coeff_part)

        coeff = parse_coeff_part_re.findall(coeff_part)[0]

        constant = ast.literal_eval(coeff[0])
        index = int(coeff[1]) if coeff[1] else None
        if index is None:
            term.coefficient_parts.append(Coefficient(constant=constant, t=None))
            continue
        power = int(coeff[2]) if coeff[2] else 1

        term.coefficient_parts.append(Coefficient(constant=leading_sign * constant, t=(index, power)))

    equation.terms.append(term)

print("\n".join(map(str, equation.terms)))
