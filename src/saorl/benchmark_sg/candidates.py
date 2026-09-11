"""Source-grounded benchmark, step 3: the FROZEN candidate transformation library.

Given one parsed target reading, deterministically enumerate structurally valid
alternative readings through the fixed transformation families of plan 3.1
(Prometheus) and 3.2 (Kyverno).  Nothing here is learned or sampled; the grids
and substitution rules are frozen constants.  The identity reading (the parsed
rule itself) is always included as the source-grounded gold candidate.

Structurally-identical duplicates are removed here (by struct_key); candidates
that are merely *behaviourally* identical on the fixture universe are merged
later, in evaluate.py, per plan 3.1 ("candidates ... identical on the frozen
fixture universe are merged").
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence

from .parse import KyvReading, PromReading, PromTarget

# --------------------------------------------------------------------------
# Frozen grids (plan 3.1)
# --------------------------------------------------------------------------
DURATION_GRID_S = [0.0, 120.0, 300.0, 600.0, 900.0, 1800.0]     # {0,2,5,10,15,30}m
RATE_GRID_S = [60.0, 300.0, 600.0, 1800.0]                       # {1,5,10,30}m
AGG_VARIANTS = ["none", "sum", "max", "avg"]                     # per-instance/sum by/max by/avg by


def _threshold_neighbours(t: float, bank: Sequence[float]) -> List[float]:
    """Adjacent thresholds already present in the metric family, else
    multiplicative neighbours (frozen factors)."""
    below = [v for v in bank if v < t]
    above = [v for v in bank if v > t]
    out = []
    if below:
        out.append(max(below))
    if above:
        out.append(min(above))
    # frozen multiplicative adjacents guarantee >=2 structurally-distinct values
    for f in (0.9, 1.1):
        v = round(t * f, 6)
        if v != t:
            out.append(v)
    # dedup, keep order, drop the identity value
    seen, res = set(), []
    for v in out:
        if v not in seen and v != t:
            seen.add(v)
            res.append(v)
    return res[:4]


def prom_candidates(target: PromTarget, threshold_bank: Dict[str, List[float]]) -> List[PromReading]:
    r = target.reading
    root = r.metric.split("_")[0]
    bank = threshold_bank.get(root) or threshold_bank.get("__all__", [])
    cands: List[PromReading] = [replace(r, axis="identity", op_tag="identity")]

    # threshold substitutions (adjacent values)
    for v in _threshold_neighbours(r.threshold, bank):
        cands.append(replace(r, threshold=v, axis="threshold", op_tag=f"threshold={v:g}"))

    # comparator alternatives (same-direction boundary variant; type-valid)
    upper = r.comparator in (">", ">=")
    for c in ([">", ">="] if upper else ["<", "<="] if r.comparator in ("<", "<=") else []):
        if c != r.comparator:
            cands.append(replace(r, comparator=c, axis="comparator", op_tag=f"comparator={c}"))

    # duration grid
    for d in DURATION_GRID_S:
        if d != r.for_s:
            cands.append(replace(r, for_s=d, axis="duration", op_tag=f"for={d/60:g}m"))

    # rate/lookback grid (only when the target uses a range window)
    if r.rate_window_s is not None:
        for w in RATE_GRID_S:
            if w != r.rate_window_s:
                cands.append(replace(r, rate_window_s=w, axis="rate", op_tag=f"rate={w/60:g}m"))

    # aggregation variants (per-instance / sum by / max by / avg by / global sum)
    grouping = r.agg_by if r.agg_by else tuple(sorted(k for k, _, _ in r.selectors))[:2]
    for a in AGG_VARIANTS:
        if a != r.aggregation:
            gb = tuple() if a == "none" else grouping
            cands.append(replace(r, aggregation=a, agg_by=gb, axis="aggregation", op_tag=f"agg={a}"))
    if r.aggregation != "sum":
        cands.append(replace(r, aggregation="sum", agg_by=tuple(), axis="aggregation", op_tag="agg=global_sum"))

    # scope +/- one label matcher
    if r.selectors:
        dropped = tuple(s for s in r.selectors if s != r.selectors[-1])
        cands.append(replace(r, selectors=dropped, axis="scope", op_tag="scope-drop1"))
    added = tuple(sorted(r.selectors + (("cluster", "=", "prod"),)))
    cands.append(replace(r, selectors=added, axis="scope", op_tag="scope+cluster"))

    # conjunction / disjunction (both compile; distinct on the guard/secondary channels)
    cands.append(replace(r, logic="conj", axis="logic", op_tag="AND guard"))
    cands.append(replace(r, logic="disj", axis="logic", op_tag="OR secondary"))

    return _dedup(cands)


# --------------------------------------------------------------------------
# Kyverno (plan 3.2)
# --------------------------------------------------------------------------
KYV_COMMON_EXCEPTIONS = ["kube-system", "kube-node-lease", "kube-public"]
KYV_COMMON_LABEL = "app.kubernetes.io/version"
KYV_COMMON_REGISTRY = "ghcr.io"
KYV_COMMON_CAP = "NET_BIND_SERVICE"
KYV_KIND_BROADEN = ["Deployment", "StatefulSet", "DaemonSet"]
KYV_ACTIONS = ["Enforce", "Audit", "Warn"]


def kyv_candidates(reading: KyvReading) -> List[KyvReading]:
    r = reading
    cands: List[KyvReading] = [replace(r, axis="identity", op_tag="identity")]

    # match.any vs match.all
    other = "all" if r.match_mode == "any" else "any"
    cands.append(replace(r, match_mode=other, axis="match_mode", op_tag=f"match.{other}"))

    # exception set: strict subset (drop one) and strict superset (add one)
    if r.exclude_ns:
        sub = tuple(r.exclude_ns[:-1])
        cands.append(replace(r, exclude_ns=sub, axis="exception", op_tag="except-subset"))
    add_ns = next((n for n in KYV_COMMON_EXCEPTIONS if n not in r.exclude_ns), None)
    if add_ns:
        sup = tuple(sorted(set(r.exclude_ns) | {add_ns}))
        cands.append(replace(r, exclude_ns=sup, axis="exception", op_tag="except-superset"))

    # action variants
    for a in KYV_ACTIONS:
        if a != r.action:
            cands.append(replace(r, action=a, axis="action", op_tag=f"action={a}"))

    # match-kind scope: broaden (add a workload kind) / narrow (Pod only)
    add_kind = next((k for k in KYV_KIND_BROADEN if k not in r.match_kinds), None)
    if add_kind:
        cands.append(replace(r, match_kinds=tuple(sorted(set(r.match_kinds) | {add_kind})),
                             axis="scope", op_tag=f"kind+{add_kind}"))
    if set(r.match_kinds) != {"Pod"} and "Pod" in r.match_kinds:
        cands.append(replace(r, match_kinds=("Pod",), axis="scope", op_tag="kind=Pod-only"))

    # required-label-set variants
    if r.family == "required_labels" and r.required_labels:
        if len(r.required_labels) > 1:
            cands.append(replace(r, required_labels=tuple(r.required_labels[:-1]),
                                 axis="label_set", op_tag="labels-drop1"))
        cands.append(replace(r, required_labels=tuple(sorted(set(r.required_labels) | {KYV_COMMON_LABEL})),
                             axis="label_set", op_tag="labels+version"))

    # registry allowlist variants
    if r.family == "registry" and r.allowed_registries:
        if len(r.allowed_registries) > 1:
            cands.append(replace(r, allowed_registries=tuple(r.allowed_registries[:-1]),
                                 axis="allowlist", op_tag="registry-drop1"))
        cands.append(replace(r, allowed_registries=tuple(sorted(set(r.allowed_registries) | {KYV_COMMON_REGISTRY})),
                             axis="allowlist", op_tag="registry+ghcr"))

    # capability allowlist variants
    if r.family == "capabilities":
        cands.append(replace(r, allowed_caps=tuple(sorted(set(r.allowed_caps) | {KYV_COMMON_CAP})),
                             axis="allowlist", op_tag="cap+NET_BIND"))
        if r.allowed_caps:
            cands.append(replace(r, allowed_caps=tuple(r.allowed_caps[:-1]),
                                 axis="allowlist", op_tag="cap-drop1"))

    # resource threshold variants
    if r.family == "resources":
        for reqs in (("limits",), ("requests",), ("limits", "requests")):
            if tuple(sorted(reqs)) != r.resource_reqs:
                cands.append(replace(r, resource_reqs=tuple(sorted(reqs)),
                                     axis="threshold", op_tag=f"reqs={'+'.join(reqs)}"))

    return _dedup_kyv(cands)


# --------------------------------------------------------------------------
def _dedup(cands: List[PromReading]) -> List[PromReading]:
    seen, out = set(), []
    for c in cands:
        k = c.struct_key()
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _dedup_kyv(cands: List[KyvReading]) -> List[KyvReading]:
    seen, out = set(), []
    for c in cands:
        k = c.struct_key()
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out
