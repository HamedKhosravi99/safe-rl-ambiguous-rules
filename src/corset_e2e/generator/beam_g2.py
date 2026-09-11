"""A factorized beam generator over G2 readings.

Availability said the ingredients exist; this has to actually emit the
reading. The search is factorized rather than Cartesian, because the
Cartesian product of the channels is what produced the old
twenty-seven-million-reading pools:

    shape  ->  term shape per slot  ->  metric per selector  ->
    scalars and windows  ->  label sets

with a beam kept after each stage, so a partial reading that is already
implausible never has its remaining slots filled.

Shapes are RANKED, not memorised. Dev exhibits 67 distinct shapes and they
account for only 35.1% of test units, so a generator that proposes dev's
shapes would be capped there. Instead a small PCFG over the shape
productions is estimated on dev -- how often a predicate expands to a set
operation rather than a comparison, which comparator, whether a matching
modifier is present, whether the right-hand side is a scalar or a series --
and shapes are enumerated in order of probability. That assigns mass to
shapes dev never wrote but whose parts dev did.

Label sets are ranked compositionally rather than uniformly. The
availability measurement enumerated all 3,301 subsets of a repository's
top keys, which inflated the label column by construction; here a set
scores as the sum of its keys' repository evidence with a size penalty, so
`{component, environment, tier}` beats an arbitrary triple because those
keys actually co-occur in the deployment's selectors.

EVERY search constant -- beam width, per-slot top-k, pool cap, and the
scoring weights -- is frozen from dev in freeze_policy() and must not be
adjusted after a test measurement.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from corset_e2e.dsl.ast_canon import _num, canon_key
from corset_e2e.generator.pred_shape import (
    CMP, SETOP, T_SLOT, N_SLOT, derivable, predicate_shape, shape_cost)
from corset_e2e.generator.skeletons import HOLE, HOLE_NUM, HOLE_WIN

LOG0 = -1e9


# ------------------------------------------------------------------ policy

DEFAULT_POLICY = dict(
    beam=64, k_shape=32, k_term=6, k_metric=8, k_thr=6, k_win=4, k_label=6,
    pool_cap=5000,
    w_shape=1.0, w_term=1.0, w_metric=1.0, w_scalar=1.0, w_label=1.0,
    label_size_penalty=0.35)


# --------------------------------------------------------------- the PCFG

class ShapePCFG:
    """Production probabilities for predicate shapes, estimated on dev."""

    def __init__(self):
        self.p_setop = 0.3
        self.setop = Counter()
        self.cmp = Counter()
        self.p_match_setop = 0.3
        self.p_match_cmp = 0.3
        self.p_rhs_num = 0.8

    def fit(self, shapes: Counter) -> "ShapePCFG":
        n_pred = n_setop = 0
        n_cmp = 0
        m_setop = m_cmp = 0
        rhs_num = 0

        def visit(s):
            nonlocal n_pred, n_setop, n_cmp, m_setop, m_cmp, rhs_num
            if not isinstance(s, list) or not s or s[0] != "bin":
                n_pred += 1
                return
            n_pred += 1
            op = s[1]
            if op in SETOP:
                n_setop += 1
                self.setop[op] += 1
                m_setop += s[4] is not None
                visit(s[2])
                visit(s[3])
            else:
                n_cmp += 1
                self.cmp[op] += 1
                m_cmp += s[4] is not None
                rhs_num += s[3] == N_SLOT

        for k, c in shapes.items():
            s = json.loads(k)
            for _ in range(c):
                visit(s)
        self.p_setop = n_setop / max(n_pred, 1)
        self.p_match_setop = m_setop / max(n_setop, 1)
        self.p_match_cmp = m_cmp / max(n_cmp, 1)
        self.p_rhs_num = rhs_num / max(n_cmp, 1)
        return self

    def _lp_choice(self, counter: Counter, key, universe) -> float:
        tot = sum(counter.values()) + len(universe)
        return math.log((counter.get(key, 0) + 1) / tot)

    def logprob(self, s) -> float:
        if not isinstance(s, list) or not s or s[0] != "bin":
            return math.log(max(1e-6, 1.0 - self.p_setop))
        op = s[1]
        if op in SETOP:
            lp = math.log(max(1e-6, self.p_setop))
            lp += self._lp_choice(self.setop, op, SETOP)
            p = self.p_match_setop if s[4] is not None else 1 - self.p_match_setop
            lp += math.log(max(1e-6, p))
            return lp + self.logprob(s[2]) + self.logprob(s[3])
        lp = math.log(max(1e-6, 1.0 - self.p_setop))
        lp += self._lp_choice(self.cmp, op, CMP)
        p = self.p_match_cmp if s[4] is not None else 1 - self.p_match_cmp
        lp += math.log(max(1e-6, p))
        q = self.p_rhs_num if s[3] == N_SLOT else 1 - self.p_rhs_num
        lp += math.log(max(1e-6, q))
        return lp


MATCHINGS = (None, ["on", "?", None, "?"], ["on", "?", "group_left", "?"],
             ["ignoring", "?", None, "?"])


def enumerate_shapes(pcfg: ShapePCFG, caps, max_cost: int, k: int,
                     level_beam: int = 4000) -> List[list]:
    """Top-k shapes by PCFG probability, enumerated by increasing cost.

    Pruning to the best `level_beam` shapes at each cost keeps the search
    bounded without the insertion-order bias that let large structures
    crowd out small ones in the earlier skeleton enumeration.
    """
    atoms: List[list] = []
    for op in CMP:
        for m in MATCHINGS:
            atoms.append(["bin", op, T_SLOT, N_SLOT, m, False])
            atoms.append(["bin", op, T_SLOT, T_SLOT, m, False])
    if "no_comparison" in caps:
        atoms.append(T_SLOT)
    atoms = [a for a in atoms if derivable(a, caps, max_cost)]

    by_cost: Dict[int, List[list]] = defaultdict(list)
    for a in atoms:
        by_cost[shape_cost(a)].append(a)
    setops = [s for s in SETOP if f"setop_{s}" in caps]

    for c in range(1, max_cost + 1):
        for ca in range(1, c):
            cb = c - ca - 1
            if cb < 1:
                continue
            for a in by_cost.get(ca, ()):
                for b in by_cost.get(cb, ()):
                    for op in setops:
                        for m in (None, MATCHINGS[1]):
                            s = ["bin", op, a, b, m, False]
                            if derivable(s, caps, max_cost):
                                by_cost[c].append(s)
        if len(by_cost[c]) > level_beam:
            by_cost[c].sort(key=pcfg.logprob, reverse=True)
            del by_cost[c][level_beam:]

    allsh = [s for lst in by_cost.values() for s in lst]
    allsh.sort(key=pcfg.logprob, reverse=True)
    out, seen = [], set()
    for s in allsh:
        key = json.dumps(s, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= k:
            break
    return out


# ------------------------------------------------------------- assembling

def _slots(shape) -> Tuple[int, int, int]:
    """(term slots, scalar slots, matching slots) in a shape."""
    t = n = m = 0
    if isinstance(shape, list) and shape and shape[0] == "bin":
        if shape[4] is not None:
            m += 1
        for side in (shape[2], shape[3]):
            if side == T_SLOT:
                t += 1
            elif side == N_SLOT:
                n += 1
            else:
                st, sn, sm = _slots(side)
                t += st
                n += sn
                m += sm
    elif shape == T_SLOT:
        t += 1
    return t, n, m


def _fill(shape, terms: List, scalars: List, labels: List, ctr: List[int]):
    """Substitute ranked choices into a shape, left to right."""
    if shape == T_SLOT:
        v = terms[ctr[0] % len(terms)]
        ctr[0] += 1
        return v
    if shape == N_SLOT:
        # normalise exactly as ast_canon does, or an integer-valued
        # threshold serialises as 0.0 where the gold says 0 and every such
        # candidate silently misses
        v = ["num", _num(scalars[ctr[1] % len(scalars)])]
        ctr[1] += 1
        return v
    if isinstance(shape, list) and shape and shape[0] == "bin":
        m = shape[4]
        mm = None
        if m is not None:
            lab = labels[ctr[2] % len(labels)]
            ctr[2] += 1
            mm = [m[0], sorted(lab), m[2], []]
        return ["bin", shape[1],
                _fill(shape[2], terms, scalars, labels, ctr),
                _fill(shape[3], terms, scalars, labels, ctr), mm, shape[5]]
    return shape


def instantiate_term(term_shape, metric: str, window: Optional[float]):
    """Put a concrete metric (and window) into a term shape's holes."""
    def go(t):
        if not isinstance(t, list) or not t:
            return t
        if t[0] == "sel":
            return ["sel", metric, []]
        if t[0] == "range":
            return ["range", go(t[1]),
                    _num(window) if window is not None else t[2]]
        if t[0] == "num":
            return t
        if t[0] == "call":
            return ["call", t[1]] + [go(x) for x in t[2:]]
        if t[0] == "agg":
            return ["agg", t[1], go(t[2]),
                    None if t[3] is None else go(t[3]), t[4], []]
        if t[0] == "bin":
            return ["bin", t[1], go(t[2]), go(t[3]),
                    None if t[4] is None else t[4], t[5]]
        return t
    return go(term_shape)
