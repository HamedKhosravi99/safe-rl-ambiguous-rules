"""Predicate SHAPE: structure with whole terms as single slots.

The enumeration explosion came from a category error, not from a budget
being too small. Term structure and predicate structure were enumerated in
one space, so `sum(rate(x[5m]))` cost four nodes and a two-window
burn-rate conjunction cost fourteen -- and the number of fourteen-node
structures is astronomically larger than any candidate set. But the term
is not something the structure generator should be building: the
development vocabulary already supplies it as one reusable atom, and that
vocabulary covers 97.9% of the term occurrences in test.

So a reading factorizes into two independent questions:

    SHAPE      and(cmp(T, N), cmp(T, N))     -- which predicate structure
    ASSIGNMENT T := sum(rate(?[?w]))          -- which term fills each slot

A shape counts each term operand as one slot regardless of what is inside
it, which is what makes the burn-rate template small rather than huge. The
space of shapes below a fixed cost is then a few hundred thousand rather
than astronomically many, and it can be enumerated exhaustively.

`T` is a vector-valued slot, `N` a scalar-valued one. This module never
reads the hidden store.
"""
from __future__ import annotations

import json
from typing import Any

T_SLOT = ["T"]
N_SLOT = ["N"]

CMP = (">", "<", ">=", "<=", "==", "!=")
SETOP = ("and", "unless", "or")


def _is_num(a: Any) -> bool:
    return isinstance(a, list) and len(a) >= 1 and a[0] == "num"


def predicate_shape(a: Any) -> Any:
    """Canonical AST -> shape, replacing each whole term with a slot."""
    if not isinstance(a, list) or not a:
        return T_SLOT
    if a[0] == "bin" and a[1] in SETOP:
        return ["bin", a[1], predicate_shape(a[2]), predicate_shape(a[3]),
                None if a[4] is None else [a[4][0], "?", a[4][2], "?"], a[5]]
    if a[0] == "bin" and a[1] in CMP:
        lhs = N_SLOT if _is_num(a[2]) else T_SLOT
        rhs = N_SLOT if _is_num(a[3]) else T_SLOT
        return ["bin", a[1], lhs, rhs,
                None if a[4] is None else [a[4][0], "?", a[4][2], "?"], a[5]]
    # a bare term in predicate position (absent(), a set-op operand)
    return T_SLOT


def shape_cost(shape: Any) -> int:
    """Nodes, with each slot costing one."""
    if not isinstance(shape, list) or not shape:
        return 1
    if shape in (T_SLOT, N_SLOT):
        return 1
    if shape[0] == "bin":
        return 1 + shape_cost(shape[2]) + shape_cost(shape[3])
    return 1


def shape_key(a: Any) -> str:
    return json.dumps(predicate_shape(a), separators=(",", ":"))


def terms_of(a: Any, out: list) -> None:
    """The whole terms occupying the slots of a predicate."""
    if not isinstance(a, list) or not a:
        return
    if a[0] == "bin" and a[1] in SETOP:
        terms_of(a[2], out)
        terms_of(a[3], out)
        return
    if a[0] == "bin" and a[1] in CMP:
        for side in (a[2], a[3]):
            if not _is_num(side):
                out.append(side)
        return
    out.append(a)


def derivable(shape: Any, caps, max_cost: int) -> bool:
    """Is this shape derivable from the declared productions within budget?

    Membership does not require enumerating the space. Enumerating every
    shape below the dev-derived cost bound would produce millions of
    objects to answer a question that is a recursive check on one shape,
    so the availability measurement asks derivability directly. A per-unit
    generator still enumerates, but only over the operands it retrieved.
    """
    if shape_cost(shape) > max_cost:
        return False
    return _derivable(shape, set(caps))


def _derivable(s: Any, caps: set) -> bool:
    if s == T_SLOT:
        return "no_comparison" in caps
    if s == N_SLOT:
        return False
    if not isinstance(s, list) or s[0] != "bin":
        return False
    op, lhs, rhs, matching = s[1], s[2], s[3], s[4]
    if matching is not None and "vector_matching" not in caps:
        return False
    if op in SETOP:
        if f"setop_{op}" not in caps:
            return False
        return _derivable(lhs, caps) and _derivable(rhs, caps)
    if op not in CMP:
        return False
    # a comparison: term against scalar, or term against term
    if lhs == N_SLOT and rhs == N_SLOT:
        return False
    if N_SLOT in (lhs, rhs):
        return T_SLOT in (lhs, rhs)
    if lhs == T_SLOT and rhs == T_SLOT:
        return ("threshold_reference" in caps
                or "threshold_expr_vector" in caps)
    return False
