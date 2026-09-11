"""Expression SKELETONS: the structure of a reading with its operands blanked.

Enumerating ASTs as a product of axes is what makes candidate sets
explode. A skeleton avoids that by separating the two questions a reading
answers. The skeleton says WHAT SHAPE the rule has -- a ratio of two rates
compared to a constant, a burn-rate conjunction over two windows, a
selector compared to a reference series -- and the operands say WHICH
series and WHICH numbers fill it. Candidates are then skeletons paired
with retrieved operands, not the Cartesian product of every axis.

A skeleton keeps operators, function names, aggregation operators,
comparison direction and the on/ignoring/group cardinality, and blanks
metric names, label matchers, scalars, windows and grouping label lists,
since those are what retrieval and the axes supply.

PROVENANCE. Skeletons may only be mined from the development split. The
generator is never allowed to read rule expressions from the calibration
or test deployments -- that is the leak the audits exist to prevent -- and
the repository catalogues are deliberately harvested from non-rule files
for the same reason. Dev is the declared design surface, the same one that
fixed the frozen grids, so mining structure there is the same kind of
prior, and it is frozen once and reused.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, List

HOLE = "?"
HOLE_NUM = "?n"
HOLE_WIN = "?w"


def skeletonize(canon_ast: Any) -> Any:
    """Blank the operands of a canonical AST, keep its structure."""
    a = canon_ast
    if not isinstance(a, list) or not a:
        return a
    tag = a[0]
    if tag == "sel":
        return ["sel", HOLE]
    if tag == "num":
        return ["num", HOLE_NUM]
    if tag == "str":
        return ["str", HOLE]
    if tag == "range":
        return ["range", skeletonize(a[1]), HOLE_WIN]
    if tag == "subquery":
        return ["subquery", skeletonize(a[1]), HOLE_WIN, HOLE_WIN, None]
    if tag == "call":
        return ["call", a[1]] + [skeletonize(x) for x in a[2:]]
    if tag == "agg":
        return ["agg", a[1], skeletonize(a[2]),
                None if a[3] is None else skeletonize(a[3]), a[4], HOLE]
    if tag == "unary":
        return ["unary", a[1], skeletonize(a[2])]
    if tag == "bin":
        m = a[4]
        mm = None if m is None else [m[0], HOLE, m[2], HOLE]
        return ["bin", a[1], skeletonize(a[2]), skeletonize(a[3]), mm, a[5]]
    return a


def skeleton_key(canon_ast: Any) -> str:
    return json.dumps(skeletonize(canon_ast), separators=(",", ":"))


def holes(skel: Any, kind: str = "sel") -> int:
    """How many operand slots of a kind a skeleton has."""
    n = 0
    if isinstance(skel, list):
        if skel and skel[0] == kind:
            return 1
        for x in skel:
            if isinstance(x, list):
                n += holes(x, kind)
    return n


def size(skel: Any) -> int:
    if not isinstance(skel, list):
        return 1
    return 1 + sum(size(x) for x in skel if isinstance(x, list))


def mine(gold: dict, splits=("dev",)) -> Counter:
    """Skeleton -> frequency, over the allowed splits only."""
    out = Counter()
    for rec in gold.values():
        if rec.get("split") not in splits or "ast" not in rec:
            continue
        out[skeleton_key(rec["ast"])] += 1
    return out
