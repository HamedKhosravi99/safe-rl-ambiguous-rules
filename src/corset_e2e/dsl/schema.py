"""WP-A: the frozen bounded executable DSL (v1).

A reading is the six-tuple

    (metric, comparator, threshold, rate_window_s, aggregation, for_s)

All axes except `metric` range over frozen finite sets declared here. The
`metric` axis ranges over a frozen public catalog (harvested from the
DEVELOPMENT split only) plus metric names licensed by the unit's own visible
text. Nothing in this module reads the hidden store.

Mechanical in-DSL eligibility (plan A.4) is computed AFTER generation by
comparing the hidden target against these frozen sets; a unit that falls
outside them is labelled with its reason code and abstains rather than being
dropped from the denominator.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

DSL_VERSION = "corset-e2e-dsl-v1"

# ---- frozen finite axes (designed on the development split only) -----------
COMPARATORS: Tuple[str, ...] = (">", ">=", "<", "<=", "==", "!=")
AGGREGATIONS: Tuple[str, ...] = ("none", "sum", "avg", "max", "min", "count")
RATE_WINDOWS_S: Tuple[Optional[float], ...] = (
    None, 60.0, 120.0, 180.0, 300.0, 600.0, 900.0, 1800.0, 3600.0, 7200.0,
    10800.0, 21600.0, 43200.0, 86400.0)
FOR_S: Tuple[float, ...] = (
    0.0, 60.0, 120.0, 180.0, 300.0, 600.0, 900.0, 1200.0, 1800.0, 3600.0,
    7200.0, 14400.0, 21600.0, 43200.0, 86400.0, 604800.0)

# Decade grid: 99.9% of development thresholds are multiples of one of these
# within a decade, over 0 .. 1e10.
_MANTISSAS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0, 9.0)
_DECADES = tuple(10.0 ** e for e in range(-4, 11))


def threshold_grid() -> Tuple[float, ...]:
    vals = {0.0, 1.0}
    for d in _DECADES:
        for m in _MANTISSAS:
            v = m * d
            if v <= 1e10:
                vals.add(round(v, 6))
    # fractional guards used pervasively for ratios
    vals.update({0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999})
    return tuple(sorted(vals))


THRESHOLD_GRID = threshold_grid()
_TG = set(THRESHOLD_GRID)
_RW = set(RATE_WINDOWS_S)
_FS = set(FOR_S)

# ---- eligibility reason codes (plan A.4) ----------------------------------
IN_DSL = "IN_DSL"
OOD_OPERATOR = "OUT_OF_DSL_OPERATOR"
OOD_FEATURE = "OUT_OF_DSL_FEATURE"
OOD_CONSTANT = "OUT_OF_DSL_CONSTANT"
OOD_WINDOW = "OUT_OF_DSL_WINDOW"
PARSER_FAILURE = "PARSER_FAILURE"


def canonicalize(metric: str, comparator: str, threshold: float,
                 rate_window_s, aggregation: str, for_s: float) -> tuple:
    """Single canonical representation of a reading (plan B.3)."""
    return (str(metric),
            str(comparator),
            round(float(threshold), 6),
            None if rate_window_s in (None, 0, 0.0) else float(rate_window_s),
            str(aggregation),
            float(for_s))


def type_check(r: tuple) -> bool:
    """Well-typedness against the frozen finite axes (metric axis excluded)."""
    _m, c, t, w, a, f = r
    return (c in COMPARATORS and a in AGGREGATIONS and w in _RW and f in _FS
            and isinstance(t, float))


def classify_target(hidden: dict) -> str:
    """Mechanical eligibility of the hidden target (plan A.4).

    This measures the REACH OF THE GRAMMAR only: can the frozen DSL express
    this target at all, with every parameter inside the public grids? It is
    deliberately independent of the generator's metric catalog and of slot
    licensing, so that a target the generator failed to license counts as a
    generation failure against rho_gen rather than being relabelled
    out-of-DSL and quietly removed from the denominator.
    """
    if hidden.get("parser_status") != "OK" or not hidden.get("parsed"):
        return PARSER_FAILURE
    p = hidden["parsed"]
    if p["comparator"] not in COMPARATORS or p["aggregation"] not in AGGREGATIONS:
        return OOD_OPERATOR
    if round(float(p["threshold"]), 6) not in _TG:
        return OOD_CONSTANT
    w = p["rate_window_s"]
    w = None if w in (None, 0, 0.0) else float(w)
    if w not in _RW or float(hidden.get("for_s", 0.0)) not in _FS:
        return OOD_WINDOW
    return IN_DSL


def target_reading(hidden: dict) -> Optional[tuple]:
    if hidden.get("parser_status") != "OK" or not hidden.get("parsed"):
        return None
    p = hidden["parsed"]
    return canonicalize(p["metric"], p["comparator"], p["threshold"],
                        p["rate_window_s"], p["aggregation"],
                        float(hidden.get("for_s", 0.0)))


def manifest() -> dict:
    payload = dict(
        dsl_version=DSL_VERSION, comparators=list(COMPARATORS),
        aggregations=list(AGGREGATIONS),
        rate_windows_s=[w if w is not None else "none" for w in RATE_WINDOWS_S],
        for_s=list(FOR_S), n_thresholds=len(THRESHOLD_GRID),
        threshold_min=THRESHOLD_GRID[0], threshold_max=THRESHOLD_GRID[-1])
    payload["dsl_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


if __name__ == "__main__":
    print(json.dumps(manifest(), indent=1))
    print("threshold grid size:", len(THRESHOLD_GRID))
