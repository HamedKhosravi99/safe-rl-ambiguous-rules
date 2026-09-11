"""Enumerate skeletons from a bounded production grammar.

Mining skeletons from dev covers only 6.1% of G2 test units: dev's 272
distinct structures are not the structures other deployments write. But
structure is extremely repetitive -- the 4,251 in-grammar test units use
on the order of 400 distinct skeletons -- so the answer is to GENERATE
structure from a small grammar rather than to retrieve it, which is also
what keeps the result auditable: every candidate structure is derivable
from a production, and no structure outside the productions is proposed.

The productions are exactly the declared G2 capabilities, so widening the
grammar is a change to G2_SPEC.json rather than to this file:

    T  ::=  sel | num | range(sel) | fn(T) | agg(T) | T arith T
    P  ::=  T cmp T | P setop P | agg(P) | T          (the last only when
                                                       `no_comparison`)

Enumeration is bottom-up by size with a hard budget, so the candidate
structure set is finite, deterministic, and independent of any target.
This module never reads the hidden store.
"""
from __future__ import annotations

import json
from typing import Dict, List, Set

from corset_e2e.generator.skeletons import HOLE, HOLE_NUM, HOLE_WIN, size

ARITH = ("/", "-", "*", "+")
CMP = (">", "<", ">=", "<=", "==", "!=")
SETOP = ("and", "unless", "or")
# matching modifier variants: none, plain on/ignoring, and a group_left join
MATCHINGS = (None,
             ["on", HOLE, None, HOLE],
             ["on", HOLE, "group_left", HOLE],
             ["ignoring", HOLE, None, HOLE])

SEL = ["sel", HOLE]
NUM = ["num", HOLE_NUM]


def _k(x) -> str:
    return json.dumps(x, separators=(",", ":"))


def enumerate_terms(caps: Set[str], fns: List[str], aggs: List[str],
                    max_size: int, term_budget: int = 3000
                    ) -> Dict[int, List[list]]:
    """Vector-valued skeletons indexed by size, smallest first.

    Size order is not a detail. An unordered breadth-first enumeration
    spends its budget on large terms and never reaches structures like
    `A unless B` (size 3, 487 test units); indexing by size guarantees
    that every smaller structure exists before any larger one is built.
    """
    by: Dict[int, List[list]] = {}
    seen: Set[str] = set()

    def full() -> bool:
        return len(seen) >= term_budget

    def add(x):
        k = _k(x)
        if k in seen or full():
            return
        seen.add(k)
        by.setdefault(size(x), []).append(x)

    add(SEL)
    add(NUM)
    add(["range", SEL, HOLE_WIN])
    for s in range(2, max_size + 1):
        if full():
            break
        for sub_size, subs in list(by.items()):
            if sub_size >= s or full():
                continue
            for t in subs:
                for f in fns:
                    c = ["call", f, t]
                    if size(c) == s:
                        add(c)
                for a in aggs:
                    c = ["agg", a, t, None, "by", HOLE]
                    if size(c) == s:
                        add(c)
        if "arith_vector" in caps and not full():
            for sa, la in list(by.items()):
                if full():
                    break
                for sb, lb in list(by.items()):
                    if sa + sb + 1 != s or full():
                        continue
                    for x in la:
                        if full():
                            break
                        for y in lb:
                            if full():
                                break
                            for op in ARITH:
                                for m in (None, MATCHINGS[1]):
                                    add(["bin", op, x, y, m, False])
    return by


def enumerate_skeletons(spec_caps, max_term_size: int = 5,
                        max_pred_size: int = 16,
                        budget: int = 400000,
                        term_budget: int = 3000,
                        term_vocab=None) -> List[list]:
    """Compose predicates over a term vocabulary.

    `term_vocab`, when given, replaces enumeration of term shapes. This is
    the difference between a feasible and an infeasible search. Enumerating
    term shapes produces on the order of a thousand of them, each of which
    multiplies into thousands of atomic comparisons, and the budget is then
    exhausted on large atoms before small conjunctions are reached -- a
    size-7 `and(cmp, cmp)` covering 487 test units kept going missing that
    way. Dev exhibits only 166 distinct term shapes and they account for
    97.9% of the term-shape occurrences in test, so the vocabulary is the
    part that transfers and the composition is the part worth enumerating.
    """
    caps = set(spec_caps)
    fns = sorted({c[3:] for c in caps if c.startswith("fn_")}
                 | {"rate", "irate", "increase"})
    aggs = sorted({c[4:] for c in caps if c.startswith("agg_")}
                  | {"sum", "avg", "max", "min", "count"})

    if term_vocab is not None:
        terms = sorted(term_vocab, key=size)
    else:
        terms_by = enumerate_terms(caps, fns, aggs, max_term_size, term_budget)
        terms = [t for _s, lst in sorted(terms_by.items()) for t in lst]

    # Atomic predicates, grouped by size but NOT yet emitted. Emitting all
    # atoms before any compound spends the budget on large comparisons and
    # starves small conjunctions: a size-7 `and(cmp, cmp)` covering 487 test
    # units went missing that way while size-13 atoms were being added. The
    # enumeration below is therefore ordered by TOTAL size across both
    # productions, so every structure of size s exists before any of size s+1.
    atoms: Dict[int, List[list]] = {}

    def atom(x):
        atoms.setdefault(size(x), []).append(x)

    for t in terms:
        for op in CMP:
            atom(["bin", op, t, NUM, None, False])
            if "threshold_reference" in caps or "threshold_expr_vector" in caps:
                for m in MATCHINGS:
                    atom(["bin", op, t, SEL, m, False])
    if "no_comparison" in caps:
        for t in terms:
            atom(t)

    by: Dict[int, List[list]] = {}
    seen: Set[str] = set()

    def add(x):
        k = _k(x)
        if k in seen or len(seen) >= budget:
            return
        seen.add(k)
        by.setdefault(size(x), []).append(x)

    setops = [s for s in SETOP if f"setop_{s}" in caps]
    for s in range(1, max_pred_size + 1):
        if len(seen) >= budget:
            break
        for x in atoms.get(s, ()):
            add(x)
        for sa in range(1, s):
            sb = s - sa - 1
            if sb < 1 or len(seen) >= budget:
                continue
            for a in by.get(sa, ()):
                if len(seen) >= budget:
                    break
                for b in atoms.get(sb, ()):
                    if len(seen) >= budget:
                        break
                    for op in setops:
                        for m in (None, MATCHINGS[1]):
                            add(["bin", op, a, b, m, False])
    return [x for _s, lst in sorted(by.items()) for x in lst]


def dev_term_vocab(gold: dict) -> List[list]:
    """Vector-valued operand shapes the development split exhibits."""
    from corset_e2e.generator.skeletons import skeletonize
    out = {}

    def sub(a):
        if not isinstance(a, list) or not a:
            return
        if a[0] == "bin":
            if a[1] in CMP:
                for x in (a[2], a[3]):
                    out.setdefault(_k(skeletonize(x)), skeletonize(x))
            for x in (a[2], a[3]):
                sub(x)
        elif a[0] in ("agg", "call", "unary", "range", "subquery"):
            for x in a[1:]:
                if isinstance(x, list):
                    sub(x)

    for rec in gold.values():
        if rec.get("split") == "dev" and "ast" in rec:
            sub(rec["ast"])
    return list(out.values())


def candidate_skeletons(spec_caps, dev_skeletons, max_pred_size: int = 16,
                        budget: int = 600000, term_vocab=None) -> List[list]:
    """Exhaustive small structures, unioned with structures dev exhibits.

    Enumerating every skeleton up to the largest size the corpus uses is
    not possible and should not be attempted: the multi-window burn-rate
    template has size 14, and the number of size-14 structures over these
    productions is astronomically larger than any candidate set could hold.
    Blind enumeration is the wrong instrument past small sizes.

    So the structure channel is a union of two bounded sources. Everything
    up to `max_pred_size` is enumerated exhaustively, which covers the
    simple majority; and every structure the DEVELOPMENT split exhibits is
    added whatever its size, which is a prior over what operators actually
    write, obtained from the one split allowed to inform design. Neither
    source looks at the target, and both are frozen before any test pass.

    The gap this leaves is real and is reported rather than closed by
    enlarging the budget until the test set is covered -- that would be
    fitting the grammar to the test split.
    """
    out = {}
    for x in enumerate_skeletons(spec_caps, max_pred_size=max_pred_size,
                                 budget=budget, term_vocab=term_vocab):
        out.setdefault(_k(x), x)
    for k in dev_skeletons:
        out.setdefault(k, json.loads(k))
    return list(out.values())
