"""WP-C.1: inspectable deterministic score, additive across DSL axes.

    s(l, psi) = w_v S_var + w_n S_num + w_o S_op + w_w S_win + w_f S_for
                + w_a S_agg

Every component is a pure function of the visible text and one axis value, so
the score decomposes over the candidate product. That is what makes the
retained set {psi : s >= qhat} exactly computable by branch-and-bound over a
pool that is never materialised (see `retained`).

No network, no model, no hidden-store access.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.generator.generate import (  # noqa: E402
    AGG_CUES, DOWN_CUES, RATE_CUES, SUSTAIN_CUES, UP_CUES, metric_subtokens,
    text_durations, text_numbers, tokens)

# Frozen weights (fit on the development split only; see fit_weights).
DEFAULT_WEIGHTS = dict(var=0.40, num=0.18, op=0.14, win=0.12, forx=0.08, agg=0.08)
COMMON_THRESHOLDS = {0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 50.0, 80.0, 90.0, 100.0,
                     0.05, 0.1, 0.5, 0.8, 0.9, 0.95, 1000.0}


class Scorer:
    """Per-axis score tables for one deployment unit."""

    def __init__(self, visible: dict, slots: dict, weights: Dict[str, float]):
        text = visible.get("text", "")
        alert = visible.get("alert_name", "")
        low = f"{text} {alert}".lower()
        self.w = weights
        tset = set(tokens(text) + tokens(alert))
        nums = set(text_numbers(text))
        durs = set(text_durations(text))
        up = any(c in low for c in UP_CUES)
        down = any(c in low for c in DOWN_CUES)
        rate_cue = any(c in low for c in RATE_CUES)
        sustain = any(c in low for c in SUSTAIN_CUES) or bool(visible.get("has_for_field"))

        def s_var(m: str) -> float:
            sub = set(metric_subtokens(m))
            if not sub:
                return 0.0
            return len(sub & tset) / len(sub)

        def s_num(t: float) -> float:
            if nums and any(abs(t - v) <= 1e-9 or (v and abs(t - v / 100.0) <= 1e-9)
                            for v in nums):
                return 1.0
            if nums and any(v and 0.5 <= t / v <= 2.0 for v in nums if v > 0):
                return 0.55
            return 0.35 if t in COMMON_THRESHOLDS else 0.15

        def s_op(c: str) -> float:
            if up and not down:
                return 1.0 if c in (">", ">=") else 0.1
            if down and not up:
                return 1.0 if c in ("<", "<=") else 0.1
            return 0.6 if c in (">", "<") else 0.4

        def s_win(w: Optional[float]) -> float:
            if w is None:
                return 0.35 if rate_cue else 1.0
            if durs and any(abs(w - d) <= 1e-9 for d in durs):
                return 1.0
            return 0.55 if rate_cue else 0.2

        def s_for(f: float) -> float:
            if f == 0.0:
                return 0.3 if sustain else 1.0
            if durs and any(abs(f - d) <= 1e-9 for d in durs):
                return 1.0
            return 0.5 if sustain else 0.15

        def s_agg(a: str) -> float:
            if a == "none":
                return 0.5 if any(any(c in low for c in cs)
                                  for cs in AGG_CUES.values()) else 1.0
            return 1.0 if any(c in low for c in AGG_CUES[a]) else 0.15

        self.tables: List[List[Tuple[object, float]]] = [
            [(m, self.w["var"] * s_var(m)) for m in slots["metrics"]],
            [(c, self.w["op"] * s_op(c)) for c in slots["comparators"]],
            [(t, self.w["num"] * s_num(t)) for t in slots["thresholds"]],
            [(w, self.w["win"] * s_win(w)) for w in slots["windows"]],
            [(a, self.w["agg"] * s_agg(a)) for a in slots["aggregations"]],
            [(f, self.w["forx"] * s_for(f)) for f in slots["fors"]],
        ]
        # canonical axis order is (metric, comparator, threshold, window, agg, for)
        self.axis_order = (0, 1, 2, 3, 4, 5)

    def score(self, reading: tuple) -> float:
        """Score one reading given as (metric, cmp, thr, win, agg, for)."""
        m, c, t, w, a, f = reading
        vals = (m, c, round(float(t), 6), w, a, float(f))
        total = 0.0
        for axis, v in zip(self.tables, vals):
            hit = None
            for key, sc in axis:
                kk = round(float(key), 6) if isinstance(key, float) else key
                if kk == v:
                    hit = sc
                    break
            if hit is None:
                return float("-inf")
            total += hit
        return total

    # -- exact retention over the unmaterialised product ---------------------
    def _sorted_axes(self):
        return [sorted(ax, key=lambda kv: -kv[1]) for ax in self.tables]

    def retained(self, qhat: float, enumerate_cap: int = 0):
        """Exact |Uhat| (and members up to enumerate_cap) without materialising.

        Uses suffix max/min bounds: a subtree whose best completion still falls
        below qhat is pruned; a subtree whose worst completion already clears
        qhat is counted in closed form.
        """
        axes = self._sorted_axes()
        if any(not ax for ax in axes):
            return 0, []
        n = len(axes)
        suf_max = [0.0] * (n + 1)
        suf_min = [0.0] * (n + 1)
        suf_cnt = [1] * (n + 1)
        for i in range(n - 1, -1, -1):
            suf_max[i] = suf_max[i + 1] + axes[i][0][1]
            suf_min[i] = suf_min[i + 1] + axes[i][-1][1]
            suf_cnt[i] = suf_cnt[i + 1] * len(axes[i])

        count = 0
        members: List[tuple] = []

        def rec(i: int, partial: float, prefix: List[object]):
            nonlocal count
            if partial + suf_max[i] < qhat - 1e-12:
                return
            if partial + suf_min[i] >= qhat - 1e-12:
                count += suf_cnt[i]
                if enumerate_cap and len(members) < enumerate_cap:
                    _expand(i, prefix)
                return
            if i == n:
                return
            for key, sc in axes[i]:
                if partial + sc + suf_max[i + 1] < qhat - 1e-12:
                    break            # axis sorted descending: rest cannot qualify
                rec(i + 1, partial + sc, prefix + [key])

        def _expand(i: int, prefix: List[object]):
            if i == n:
                if len(members) < enumerate_cap:
                    members.append(tuple(prefix))
                return
            for key, _sc in axes[i]:
                if len(members) >= enumerate_cap:
                    return
                _expand(i + 1, prefix + [key])

        rec(0, 0.0, [])
        return count, members

    def maximal_count(self, qhat: float) -> int:
        """|Max(Uhat)| under pointwise cost dominance (plan G.4).

        For a reading (m, op, theta, w, a, f) the induced cost fires when the
        aggregated series compares to theta, held for f. Within a fixed
        (m, op, w, a) the firing sets are totally ordered by (theta, f): for
        op in {>, >=} a smaller theta fires whenever a larger one does, for
        op in {<, <=} a larger theta does, and a shorter hold f fires whenever
        a longer one does. So each (m, op, w, a) group contributes exactly one
        maximal element, and the group is represented in Uhat iff its best
        (theta, f) completion clears qhat.
        """
        mt, ct, tt, wt, at, ft = self.tables
        if not all((mt, ct, tt, wt, at, ft)):
            return 0
        best_t = max(sc for _k, sc in tt)
        best_f = max(sc for _k, sc in ft)
        tail = best_t + best_f
        n = 0
        for _m, sm in mt:
            for _c, sc_ in ct:
                for _w, sw in wt:
                    for _a, sa in at:
                        if sm + sc_ + sw + sa + tail >= qhat - 1e-12:
                            n += 1
        return n

    def max_score(self) -> float:
        if any(not ax for ax in self.tables):
            return 0.0
        return sum(max(sc for _k, sc in ax) for ax in self.tables)

    def empty(self) -> bool:
        """True when some axis is unlicensed, so the pool is empty."""
        return any(not ax for ax in self.tables)
