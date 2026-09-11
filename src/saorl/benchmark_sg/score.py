"""Source-grounded benchmark, step 5: the frozen deterministic score s(l, psi).

Implements plan 3.3 exactly:

    s(l, psi) = 0.30 s_entity + 0.20 s_operator + 0.20 s_value
              + 0.15 s_temporal + 0.10 s_scope + 0.05 s_action

Every component is deterministic and shares one frozen tokenizer / unit map /
weight vector (declared as module constants below -- frozen before the test
split is touched).  No LLM, no learned weights.  s_entity is token-Jaccard
between the rule's natural-language annotation and a canonical gloss of the
candidate (the plan permits "BM25 or token-Jaccard"; we use token-Jaccard).

For an intentionally vague rule that gives no cue for a component (e.g. no
number in the text), that component TIES across all candidates at a fixed
neutral constant TIE=0.5 -- creating larger sets rather than fabricating
precision (plan 3.3).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .parse import KyvReading, KyvTarget, PromReading, PromTarget, dur_to_s

# ----- FROZEN constants (fixed before inspecting any test split) -----------
WEIGHTS = dict(entity=0.30, operator=0.20, value=0.20, temporal=0.15,
               scope=0.10, action=0.05)
TIE = 0.5

STOPWORDS = frozenset("""a an the of for to in on is are be been has have had this that
these those with without and or not no than then it its as at by from up out over more less
most least when while has been being value currently only left space using use""".split())

# operator phrase -> (direction, strictness); frozen exact mapping (plan 3.3)
_OP_GT = ("above", "greater", "exceed", "exceeds", "exceeding", "higher", "more than",
          "over", "too high", "at least", "no less", "greater than or equal")
_OP_LT = ("below", "less than", "lower", "under", "fewer", "too low", "at most",
          "no more than", "less than or equal", "running out", "almost out", "out of")
_OP_INCLUSIVE = ("at least", "at most", "no more than", "no less", "or equal")

# action / severity phrases
_ACTION_MAP = {
    "critical": "critical", "page": "critical", "urgent": "critical",
    "warning": "warning", "warn": "warning", "should": "warning",
    "info": "info", "informational": "info", "notice": "info",
    "deny": "Enforce", "block": "Enforce", "forbidden": "Enforce",
    "disallow": "Enforce", "must not": "Enforce", "prevent": "Enforce",
    "enforce": "Enforce", "reject": "Enforce",
    "audit": "Audit", "report": "Audit", "detect": "Audit",
}
_SCOPE_TOKENS = frozenset("""namespace namespaces cluster clusters instance instances pod pods
container containers node nodes device mountpoint workload deployment job service volume
system default registry image""".split())


# --------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    text = re.sub(r"[_:./-]", " ", text.lower())
    toks = re.findall(r"[a-z0-9]+", text)
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _text_numbers(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", text)]


def _text_durations_s(text: str) -> List[float]:
    out = []
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(seconds?|minutes?|hours?|days?|weeks?|ms|sec|min|hrs?|[smhdw])\b", text.lower()):
        u = unit[0] if unit[0] in "smhdw" and len(unit) <= 2 else unit
        mult = {"s": 1, "sec": 1, "second": 1, "seconds": 1, "m": 60, "min": 60,
                "minute": 60, "minutes": 60, "h": 3600, "hr": 3600, "hrs": 3600,
                "hour": 3600, "hours": 3600, "d": 86400, "day": 86400, "days": 86400,
                "w": 604800, "week": 604800, "weeks": 604800, "ms": 0.001}.get(u.rstrip("s"), None)
        if mult is None:
            mult = {"minutes": 60, "seconds": 1, "hours": 3600, "days": 86400, "weeks": 604800}.get(unit, 60)
        out.append(float(num) * mult)
    return out


def _num_match(a: float, cands: List[float]) -> bool:
    for b in cands:
        if a == b or abs(a - b) <= max(1e-9, 0.02 * abs(b)):
            return True
        if b != 0 and (abs(a - b * 100) <= 0.02 * abs(b * 100) or abs(a * 100 - b) <= 0.02 * abs(b)):
            return True
    return False


def _text_direction(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = text.lower()
    gt = any(p in t for p in _OP_GT)
    lt = any(p in t for p in _OP_LT)
    incl = "inclusive" if any(p in t for p in _OP_INCLUSIVE) else None
    if gt and not lt:
        return "gt", incl
    if lt and not gt:
        return "lt", incl
    return None, incl


# --------------------------------------------------------------------------
# Prometheus scoring
# --------------------------------------------------------------------------

def _prom_gloss_tokens(r: PromReading) -> List[str]:
    toks = tokenize(r.metric)
    for k, _, v in r.selectors:
        toks += tokenize(k) + tokenize(v)
    toks += tokenize(r.aggregation) + tokenize(r.severity)
    toks += list(k2 for k2 in r.agg_by)
    return toks


def score_prom(target: PromTarget, r: PromReading) -> Tuple[float, Dict[str, float]]:
    text = target.text
    ttoks = tokenize(text)
    c = {}
    # entity
    c["entity"] = _jaccard(ttoks, _prom_gloss_tokens(r))
    # operator (comparator direction/strictness)
    tdir, tincl = _text_direction(text)
    cdir = "gt" if r.comparator in (">", ">=") else "lt"
    cincl = "inclusive" if r.comparator in (">=", "<=") else "strict"
    if tdir is None:
        c["operator"] = TIE
    elif cdir == tdir:
        c["operator"] = 1.0 if (tincl is None or tincl == cincl) else 0.75
    else:
        c["operator"] = 0.0
    # value
    nums = _text_numbers(text)
    c["value"] = TIE if not nums else (1.0 if _num_match(r.threshold, nums) else 0.0)
    # temporal (for-duration and rate window)
    durs = _text_durations_s(text)
    cand_durs = [r.for_s] + ([r.rate_window_s] if r.rate_window_s else [])
    if not durs:
        c["temporal"] = TIE
    else:
        hit = any(_num_match(cd, durs) for cd in cand_durs if cd is not None and cd > 0)
        # a candidate for=0 matches text saying no duration; treat as tie if 0
        c["temporal"] = 1.0 if hit else (TIE if all((cd or 0) == 0 for cd in cand_durs) else 0.0)
    # scope
    tscope = [t for t in ttoks if t in _SCOPE_TOKENS]
    cscope = [t for t in _prom_gloss_tokens(r) if t in _SCOPE_TOKENS]
    c["scope"] = TIE if (not tscope or not cscope) else _jaccard(tscope, cscope)
    # action ~ severity
    tact = None
    for kw, val in _ACTION_MAP.items():
        if kw in text.lower() and val in ("critical", "warning", "info"):
            tact = val
            break
    if tact is None or not r.severity:
        c["action"] = TIE
    else:
        c["action"] = 1.0 if r.severity == tact else 0.0
    total = sum(WEIGHTS[k] * c[k] for k in WEIGHTS)
    return round(total, 6), c


# --------------------------------------------------------------------------
# Kyverno scoring
# --------------------------------------------------------------------------

def _kyv_entity_tokens(r: KyvReading) -> List[str]:
    toks = tokenize(r.family)
    for coll in (r.required_labels, r.allowed_registries, r.allowed_caps,
                 r.forbidden, r.resource_reqs, r.match_kinds):
        for x in coll:
            toks += tokenize(str(x))
    return toks


def score_kyv(target: KyvTarget, r: KyvReading) -> Tuple[float, Dict[str, float]]:
    text = target.text
    ttoks = tokenize(text)
    c = {}
    c["entity"] = _jaccard(ttoks, _kyv_entity_tokens(r))
    # operator: match.any vs match.all (rarely stated) + require/deny direction
    t = text.lower()
    if "all " in t and "any " not in t:
        c["operator"] = 1.0 if r.match_mode == "all" else 0.0
    elif "any " in t and "all " not in t:
        c["operator"] = 1.0 if r.match_mode == "any" else 0.0
    else:
        c["operator"] = TIE
    # value: overlap of the allowlist / required-set with entities named in text
    valset = list(r.required_labels) + list(r.allowed_registries) + list(r.allowed_caps) + \
             list(r.forbidden) + list(r.resource_reqs)
    if not valset:
        c["value"] = TIE
    else:
        named = sum(1 for v in valset if any(tok in ttoks for tok in tokenize(str(v))))
        c["value"] = named / len(valset)
    # temporal: no temporal axis in Kyverno -> tie
    c["temporal"] = TIE
    # scope: match kinds + exception namespaces vs text scope tokens
    cscope = [x for k in r.match_kinds for x in tokenize(k)] + \
             [x for n in r.exclude_ns for x in tokenize(n)]
    tscope = [x for x in ttoks if x in _SCOPE_TOKENS]
    c["scope"] = TIE if (not cscope or not tscope) else _jaccard(cscope, tscope)
    # action
    tact = None
    for kw, val in _ACTION_MAP.items():
        if kw in t and val in ("Enforce", "Audit"):
            tact = val
            break
    if tact is None:
        c["action"] = TIE
    else:
        c["action"] = 1.0 if r.action == tact else (0.5 if r.action == "Warn" else 0.0)
    total = sum(WEIGHTS[k] * c[k] for k in WEIGHTS)
    return round(total, 6), c


# --------------------------------------------------------------------------
def score_reading(target, reading) -> Tuple[float, Dict[str, float]]:
    if isinstance(target, PromTarget):
        return score_prom(target, reading)
    return score_kyv(target, reading)


def class_score(target, reading_class) -> float:
    """A merged class's score = max over its member readings (best gloss)."""
    return max(score_reading(target, m)[0] for m in reading_class.members)
