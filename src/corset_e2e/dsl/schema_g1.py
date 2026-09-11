"""G1: the frozen six-tuple with a reference-valued threshold axis.

The frozen grammar's threshold ranges over a decade grid because it
assumes a rule compares a metric to a NUMBER. The corpus's dominant
construct compares it to another SERIES:

    gitlab_component_saturation:ratio{...}
        > on(component) group_left slo:max:hard:gitlab_component_saturation:ratio

G1 widens exactly that one axis: threshold in THRESHOLD_GRID union REF,
where REF is a metric name. Nothing else changes -- comparator,
aggregation, window and for_s keep their frozen grids, and a
reference-valued reading is evaluated on the same fixtures (read the
reference series instead of a constant), so c_psi, the retained set and
the face theorem apply verbatim.

This module never opens the hidden store except through the two
classification entry points, which are called AFTER generation exactly
as the frozen schema's are.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

from corset_e2e.dsl.schema import (
    AGGREGATIONS, COMPARATORS, FOR_S, IN_DSL, RATE_WINDOWS_S, THRESHOLD_GRID,
    classify_target, target_reading)

CMP = (">=", "<=", "==", "!=", ">", "<")
_VM_HEAD = re.compile(r"\bon\s*\(|\bignoring\s*\(|group_left|group_right")
_NAME = re.compile(r"[a-z_][a-z0-9_]*(?::[a-z0-9_]+)*")
_FN = re.compile(r"\b(rate|irate|increase|sum|avg|max|min|count|delta|"
                 r"histogram_quantile|quantile|abs|clamp_max|clamp_min|"
                 r"topk|bottomk|absent|changes|resets|deriv|predict_linear)\s*\(")
_WINDOW = re.compile(r"\[(\d+)([smhdw])\]")
_UNIT = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
_AGG = {"sum": "sum", "avg": "avg", "max": "max", "min": "min",
        "count": "count"}

IN_G1_REF = "IN_DSL_REF"


def _toplevel_split(s: str):
    depth = 0
    for i, ch in enumerate(s):
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif depth == 0:
            for c in CMP:
                if s.startswith(c, i):
                    return s[:i].strip(), c, s[i + len(c):].strip()
    return None


def _first_metric(expr: str) -> Optional[str]:
    """The leading metric identifier of a (possibly wrapped) expression."""
    for m in _NAME.finditer(expr):
        name = m.group(0)
        if name in ("on", "ignoring", "group_left", "group_right", "by",
                    "without", "and", "or", "unless", "bool", "offset"):
            continue
        if _FN.match(expr[m.start():]):
            continue                       # a function call, not a metric
        if len(name) >= 4 and ("_" in name or ":" in name):
            return name
    return None


def _window_of(expr: str) -> Optional[float]:
    m = _WINDOW.search(expr)
    if not m:
        return None
    return float(m.group(1)) * _UNIT[m.group(2)]


def _agg_of(expr: str) -> str:
    m = _FN.search(expr)
    if m and m.group(1) in _AGG:
        return _AGG[m.group(1)]
    return "none"


def parse_reference(hidden: dict) -> Optional[Tuple[str, str, str, Optional[float], str, float]]:
    """(metric, comparator, REFERENCE_NAME, window, aggregation, for_s)."""
    if hidden.get("parser_status") == "OK":
        return None                        # already inside the frozen grammar
    x = (hidden.get("raw_expr") or "").strip()
    sp = _toplevel_split(x)
    if sp is None:
        return None
    lhs, cmp_, rhs = sp
    if cmp_ not in COMPARATORS:
        return None
    head = rhs.split("{")[0][:48]
    if not _VM_HEAD.search(head):
        return None                        # rhs is not a vector-matched series
    ref = _first_metric(rhs)
    met = _first_metric(lhs)
    if not ref or not met:
        return None
    w = _window_of(lhs)
    if w is not None and w not in set(RATE_WINDOWS_S):
        return None
    agg = _agg_of(lhs)
    if agg not in AGGREGATIONS:
        return None
    f = float(hidden.get("for_s", 0.0))
    if f not in set(FOR_S):
        return None
    return (met, cmp_, ref, w, agg, f)


def classify_target_g1(hidden: dict) -> str:
    """IN_DSL (frozen grammar), IN_DSL_REF (new axis), else the frozen code."""
    base = classify_target(hidden)
    if base == IN_DSL:
        return IN_DSL
    return IN_G1_REF if parse_reference(hidden) else base


def target_reading_g1(hidden: dict):
    if classify_target(hidden) == IN_DSL:
        return target_reading(hidden)
    return parse_reference(hidden)


def in_pool_g1(slots: dict, reading: tuple) -> bool:
    m, c, t, w, a, f = reading
    if m not in set(slots["metrics"]) or c not in set(slots["comparators"]):
        return False
    if isinstance(t, str):                 # reference-valued threshold
        if t not in set(slots.get("references", ())):
            return False
    elif round(float(t), 6) not in {round(float(x), 6)
                                    for x in slots["thresholds"]}:
        return False
    return (w in set(slots["windows"]) and a in set(slots["aggregations"])
            and float(f) in {float(x) for x in slots["fors"]})
