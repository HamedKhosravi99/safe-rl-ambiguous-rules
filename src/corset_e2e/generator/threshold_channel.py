"""Threshold candidates: a registered finite grid, plus numbers retrieved
from target-blind evidence.

`threshold_offgrid` was declared as a G2 capability, which quietly asserted
that arbitrary reals are in the grammar. No generator can enumerate the
reals, so that declaration assumed a channel that did not exist and the
slot autopsy duly showed threshold availability at 56.2%. The honest
formulation separates the two:

    candidates = registered finite grid  UNION  numbers retrieved from
                 allowed evidence

Allowed evidence is the unit's own VISIBLE text and alert name -- the
prose an operator wrote -- and nothing else. The hidden executable target
is never consulted, which is what distinguishes retrieval from reading the
answer, and is enforced here by taking the visible record as the only
argument.

Retrieval is not just literal extraction. An operator writing "more than
5% of requests" implies 0.05 and 95% implies 0.95, so each literal is
expanded by the percentage readings of itself. The expansions are fixed
and declared here rather than fitted, and every one of them is a
NUMBER THE TEXT MENTIONS read in a different unit -- not a search over
values that might make the gold appear.
"""
from __future__ import annotations

import re
from typing import Iterable, Set

# a number, optionally followed by a percent sign or a unit suffix
_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%?)")
MAX_RETRIEVED = 256


def _expand(v: float, pct: bool) -> Iterable[float]:
    """The readings of a written number that denote a threshold."""
    yield v
    if pct or 0.0 < v <= 100.0:
        yield v / 100.0            # "5%"      -> 0.05
        yield 1.0 - v / 100.0      # "99.9%"   -> 0.001 error budget
    if 0.0 < v < 1.0:
        yield 1.0 - v              # a ratio and its complement


def retrieved_thresholds(text: str, alert_name: str = "") -> Set[float]:
    """Numbers the unit's own visible prose mentions, and their readings."""
    out: Set[float] = set()
    blob = f"{alert_name} {text}"
    for m in _NUM.finditer(blob):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if abs(v) > 1e11:
            continue
        for x in _expand(v, m.group(2) == "%"):
            out.add(round(float(x), 6))
        if len(out) > MAX_RETRIEVED:
            break
    return out


def threshold_axis(grid: Set[float], visible: dict) -> Set[float]:
    """The full candidate axis served to one unit."""
    return set(grid) | retrieved_thresholds(
        visible.get("text", "") or "", visible.get("alert_name", "") or "")


def dev_scalar_closure(dev_scalars, max_abs: float = 1e4) -> Set[float]:
    """Products, quotients and complements of the scalars dev rules use.

    Burn-rate alerting does not write its thresholds; it derives them. A
    multi-window SLO rule compares against 14.4 * 0.001 or 1 - 14.4 * 0.005,
    and those products appear in no prose and on no decade grid, which is
    why text retrieval added a median of ZERO candidates per unit and
    threshold availability did not move.

    The values are still not arbitrary: they are a small closure over the
    scalars the DEVELOPMENT split already uses, under the operations SLO
    arithmetic actually applies. Depth one suffices.

    Operators also write TRUNCATED decimals -- the corpus says 0.16667
    where one sixth is 0.166666..., and an exact closure misses 626 units
    on that alone -- so each derived value is also offered at three to six
    decimal places. Both the base set and the operations are declared here
    and derived from dev; neither is a search for values that make a gold
    appear.
    """
    base = sorted({round(float(x), 6) for x in dev_scalars
                   if abs(float(x)) <= max_abs})
    out: Set[float] = set(base)
    for a in base:
        for b in base:
            cands = [a * b, 1.0 - a * b]
            if b:
                cands += [a / b, 1.0 - a / b]
            for v in cands:
                if v != v or abs(v) > 1e11:
                    continue
                for dp in (3, 4, 5, 6):
                    out.add(round(float(v), dp))
    return out
