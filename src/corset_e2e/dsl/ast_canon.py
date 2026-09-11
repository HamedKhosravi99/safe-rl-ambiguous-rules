"""Canonical form for a parsed expression.

Two expressions that mean the same thing must have the same canonical
form, or proposal recall would be measured against incidental spelling.
The normalisations are deliberately conservative -- only rewrites that are
semantically inert in PromQL:

  * parentheses are dropped (they carry no meaning after parsing);
  * label matchers, aggregation grouping labels and vector-matching labels
    are sorted, since each is a set;
  * a subtree with no data reference is folded to its numeric value, so
    `14.4 * 0.001` and `0.0144` agree;
  * `x offset 0` is `x`.

Operand ORDER is never changed. `a / b` is not `b / a`, and while `and`
commutes over which series survive, it does not commute over which side
supplies the output labels, so reordering it would not be inert.

Two keys are produced for every expression. The SHAPE key omits label
matchers; the STRICT key includes them. A reading has to name a metric, an
operator, a threshold, a window and a structure, and those are what the
unit's text can plausibly determine -- `{job="node-exporter"}` is a
property of where the rule is deployed, not of what it says. Recall is
therefore reported on shape, with strict alongside it so the gap is
visible rather than assumed away.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from corset_e2e.dsl.promql_ast import (
    Aggregate, Binary, Call, MatrixSelector, Node, NumberLit, Paren,
    StringLit, Subquery, Unary, VectorSelector, fold_constant, unparen)


def _num(v: float) -> Any:
    if v != v:
        return "nan"
    if v in (float("inf"), float("-inf")):
        return "inf" if v > 0 else "-inf"
    r = round(float(v), 9)
    return int(r) if r == int(r) and abs(r) < 1e15 else r


def canon(node: Node, with_matchers: bool = True) -> Any:
    """Nested JSON-able canonical form."""
    n = unparen(node)

    folded = fold_constant(n)
    if folded is not None and not isinstance(n, NumberLit):
        return ["num", _num(folded)]

    if isinstance(n, NumberLit):
        return ["num", _num(n.value)]
    if isinstance(n, StringLit):
        return ["str", n.value]
    if isinstance(n, VectorSelector):
        out = ["sel", n.metric]
        if with_matchers:
            out.append(sorted([m.name, m.op, m.value] for m in n.matchers))
        if n.offset_s:
            out.append(["offset", _num(n.offset_s)])
        if n.at_:
            out.append(["at", n.at_])
        return out
    if isinstance(n, MatrixSelector):
        return ["range", canon(n.vs, with_matchers), _num(n.range_s)]
    if isinstance(n, Subquery):
        return ["subquery", canon(n.expr, with_matchers), _num(n.range_s),
                None if n.step_s is None else _num(n.step_s),
                None if not n.offset_s else _num(n.offset_s)]
    if isinstance(n, Call):
        return ["call", n.func] + [canon(a, with_matchers) for a in n.args]
    if isinstance(n, Aggregate):
        return ["agg", n.op,
                canon(n.expr, with_matchers),
                None if n.param is None else canon(n.param, with_matchers),
                "without" if n.without else "by",
                sorted(n.grouping)]
    if isinstance(n, Unary):
        return ["unary", n.op, canon(n.expr, with_matchers)]
    if isinstance(n, Binary):
        m = n.matching
        mm = None if m is None else [
            "on" if m.on else "ignoring", sorted(m.labels), m.card,
            sorted(m.include)]
        return ["bin", n.op, canon(n.lhs, with_matchers),
                canon(n.rhs, with_matchers), mm, bool(n.bool_)]
    return ["unknown", type(n).__name__]


def canon_key(node: Node, with_matchers: bool = True) -> str:
    return json.dumps(canon(node, with_matchers), separators=(",", ":"),
                      sort_keys=True)


def canon_hash(node: Node, with_matchers: bool = True) -> str:
    return hashlib.sha256(
        canon_key(node, with_matchers).encode()).hexdigest()[:16]


def shape_key(node: Node) -> str:
    """Canonical form with label matchers removed."""
    return canon_key(node, with_matchers=False)


def same_shape(a: Node, b: Node) -> bool:
    return shape_key(a) == shape_key(b)


def same_strict(a: Node, b: Node) -> bool:
    return canon_key(a, True) == canon_key(b, True)
