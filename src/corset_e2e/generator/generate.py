"""WP-B: target-independent slot extraction and complete bounded enumeration.

The candidate pool is the FULL product of the licensed axis sets

    M x C x T x W x A x F

with no top-K truncation (plan B.2, complete mode). The product is never
materialised: because the deterministic score is additive across axes
(plan C.1), the retained set {psi : s(l,psi) >= qhat} is enumerated exactly by
branch-and-bound over per-axis score tables, and pool membership of any
reading is an O(1) per-axis test. Completeness is therefore a property of the
licensed sets, not of an enumeration budget.

Inputs are (l, m, G) only: visible text, alert name, declared metadata, frozen
grammar. This module never opens the hidden store.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.dsl.schema import (  # noqa: E402
    AGGREGATIONS, COMPARATORS, FOR_S, RATE_WINDOWS_S, THRESHOLD_GRID,
    canonicalize)

# ---- frozen licensing constants (development split only) -------------------
N_METRICS = 64            # metric-axis licensing breadth
MIN_METRIC_SCORE = 0.08   # minimum lexical evidence to license a metric

UP_CUES = ("above", "exceed", "exceeds", "exceeded", "greater", "more than",
           "over ", "higher", "too high", "high ", "spike", "saturat",
           "exhaust", "grow", "increase", "excessive", "too many", "at least")
DOWN_CUES = ("below", "less than", "under ", "lower", "too low", "low ",
             "drop", "decrease", "insufficient", "running out", "fewer",
             "at most", "starv")
NEQ_CUES = ("not equal", "differs", "mismatch", "inconsist", "unexpected",
            "changed", "!=")
RATE_CUES = ("rate", "per second", "per-second", "/s", "throughput", "qps",
             "rps", "requests per", "errors per", "traffic", "bandwidth",
             "increase", "growth", "churn")
SUSTAIN_CUES = ("for at least", "sustained", "persistent", "persistently",
                "continuously", "over the last", "in the last", "past ",
                "during the last", "consecutive", "keeps", "repeatedly",
                "has been")
AGG_CUES = {"sum": ("total", "sum", "overall", "aggregate", "combined",
                    "across all", "cumulative"),
            "avg": ("average", "avg", "mean", "typical"),
            "max": ("max", "maximum", "peak", "highest", "worst"),
            "min": ("min", "minimum", "lowest", "least"),
            "count": ("count", "number of", "how many", "instances of")}

_DUR_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d|weeks?|w)\b",
    re.I)
_NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%|percent)?")
_DUR_MULT = {"s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
             "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
             "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
             "d": 86400, "day": 86400, "days": 86400,
             "w": 604800, "week": 604800, "weeks": 604800}


def tokens(s: str) -> List[str]:
    """Split camelCase and snake_case into lowercase word tokens."""
    parts = re.findall(r"[A-Z][a-z0-9]+|[A-Z]+(?![a-z])|[a-z0-9]+", s or "")
    return [p.lower() for p in parts if len(p) > 1]


def metric_subtokens(metric: str) -> List[str]:
    return [t for t in re.split(r"[_:]+", metric.lower()) if len(t) > 1]


class Catalog:
    """Frozen public metric catalog, harvested from the DEVELOPMENT split."""

    def __init__(self, metrics: Sequence[str]):
        self.metrics = sorted(set(metrics))
        self.subtok = {m: set(metric_subtokens(m)) for m in self.metrics}

    @classmethod
    def load(cls, path: str) -> "Catalog":
        return cls(json.load(open(path))["metrics"])

    def license(self, toks: Sequence[str], verbatim: Sequence[str]) -> List[str]:
        """Score every catalog metric by subtoken overlap; keep the top N."""
        tset = set(toks)
        scored = []
        for m, sub in self.subtok.items():
            if not sub:
                continue
            inter = len(sub & tset)
            if inter == 0:
                continue
            scored.append((inter / len(sub), m))
        scored.sort(key=lambda x: (-x[0], x[1]))
        out = [m for sc, m in scored[:N_METRICS] if sc >= MIN_METRIC_SCORE]
        for v in verbatim:                      # metric named literally in text
            if v not in out:
                out.append(v)
        return out


def text_durations(text: str) -> List[float]:
    out = set()
    for m in _DUR_RE.finditer(text):
        unit = m.group(2).lower().rstrip("s") or "s"
        mult = _DUR_MULT.get(unit) or _DUR_MULT.get(m.group(2).lower())
        if mult:
            out.add(float(m.group(1)) * mult)
    return sorted(out)


def text_numbers(text: str) -> List[float]:
    out = set()
    for m in _NUM_RE.finditer(text):
        v = float(m.group(1))
        out.add(v)
        if m.group(2):                          # percent -> also the fraction
            out.add(v / 100.0)
    return sorted(out)


def _nearest_grid(vals: Sequence[float], grid: Sequence[float]) -> List[float]:
    keep = set()
    gs = list(grid)
    for v in vals:
        keep.add(min(gs, key=lambda g: abs(g - v)))
    return sorted(keep)


def extract_slots(visible: dict, catalog: Catalog, mode: str = "complete") -> dict:
    """Target-independent licensing of every DSL axis from (l, m, G).

    mode="complete" (primary): the metric axis is licensed from the public
    catalog by lexical evidence -- metric names cannot be enumerated -- and
    every other axis takes its FULL frozen grid. Generation is then complete
    over the frozen schema conditional on the metric being licensed.

    mode="budgeted" (efficiency ablation): every axis is narrowed by textual
    cues, which shrinks the pool but can exclude the target.
    """
    text = visible.get("text", "")
    alert = visible.get("alert_name", "")
    low = f"{text} {alert}".lower()
    toks = tokens(text) + tokens(alert)
    verbatim = [w for w in re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){2,}\b", text)
                if w in catalog.subtok]

    metrics = catalog.license(toks, verbatim)

    up = any(c in low for c in UP_CUES)
    down = any(c in low for c in DOWN_CUES)
    neq = any(c in low for c in NEQ_CUES)
    if up and not down:
        comparators = (">", ">=")
    elif down and not up:
        comparators = ("<", "<=")
    elif neq and not (up or down):
        comparators = ("!=", "==")
    else:
        comparators = COMPARATORS

    nums = text_numbers(text)
    thresholds = sorted(set(_nearest_grid(nums, THRESHOLD_GRID)) | {0.0, 1.0}) \
        if nums else list(THRESHOLD_GRID)

    durs = text_durations(text)
    if any(c in low for c in RATE_CUES):
        windows = tuple([None] + _nearest_grid(durs, [w for w in RATE_WINDOWS_S if w])
                        ) if durs else RATE_WINDOWS_S
    else:
        windows = (None,) if not durs else tuple(
            [None] + _nearest_grid(durs, [w for w in RATE_WINDOWS_S if w]))

    aggs = ["none"]
    for a, cues in AGG_CUES.items():
        if any(c in low for c in cues):
            aggs.append(a)
    aggregations = tuple(dict.fromkeys(aggs))

    sustain = any(c in low for c in SUSTAIN_CUES) or visible.get("has_for_field")
    if sustain:
        fors = tuple(sorted(set([0.0] + _nearest_grid(durs, [f for f in FOR_S if f]))
                            )) if durs else FOR_S
    else:
        fors = (0.0,)

    if mode == "complete":
        comparators = COMPARATORS
        thresholds = THRESHOLD_GRID
        windows = RATE_WINDOWS_S
        aggregations = AGGREGATIONS
        fors = FOR_S
    elif mode != "budgeted":
        raise ValueError(f"unknown mode {mode!r}")
    return dict(mode=mode, metrics=tuple(metrics), comparators=tuple(comparators),
                thresholds=tuple(thresholds), windows=tuple(windows),
                aggregations=tuple(aggregations), fors=tuple(fors))


def pool_size(slots: dict) -> int:
    return (len(slots["metrics"]) * len(slots["comparators"])
            * len(slots["thresholds"]) * len(slots["windows"])
            * len(slots["aggregations"]) * len(slots["fors"]))


def in_pool(slots: dict, reading: tuple) -> bool:
    """O(1) per-axis membership test in the (never materialised) full product."""
    m, c, t, w, a, f = reading
    return (m in set(slots["metrics"]) and c in set(slots["comparators"])
            and round(float(t), 6) in {round(float(x), 6) for x in slots["thresholds"]}
            and w in set(slots["windows"]) and a in set(slots["aggregations"])
            and float(f) in {float(x) for x in slots["fors"]})
