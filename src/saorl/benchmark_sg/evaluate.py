"""Source-grounded benchmark, step 4: deterministic executable evaluators.

Each candidate reading is executed against a deterministic *synthetic fixture
universe* built to hit the boundaries the candidate axes vary (plan 3.1/3.2
step "generate deterministic synthetic fixtures ... that test threshold
boundaries, duration boundaries, label scopes, and aggregation behavior").
This is the executable target: it is faithful for the generated candidate
family (no promtool / kyverno CLI binary is required).

For a candidate we compute a *cost/fire vector* over the fixture universe:

  * Prometheus: fire in {0,1} per synthetic time-series scenario -- does the
    alert raise (predicate holds for >= `for`).
  * Kyverno: cost in {0, 0.25, 0.5, 1.0} per synthetic resource -- 0 = allowed,
    1 = Enforce-blocked, 0.5 = Warn, 0.25 = Audit (surfaced, not blocked).

Candidates with an identical vector are *merged* into one behaviour class
(plan: "candidates identical on the frozen fixture universe are merged").
The merged pool is the input to scoring and dominance.
"""
from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .candidates import kyv_candidates, prom_candidates
from .parse import KyvReading, KyvTarget, PromReading, PromTarget

# ==========================================================================
# Merged reading class + pool
# ==========================================================================

@dataclass
class ReadingClass:
    rep: object                 # representative reading (PromReading|KyvReading)
    members: List[object]
    vector: Tuple[float, ...]   # cost/fire vector over the fixture universe
    axes: Set[str]
    is_gold: bool = False


@dataclass
class Pool:
    family: str                 # 'prometheus' | 'kyverno'
    target: object              # PromTarget | KyvTarget
    classes: List[ReadingClass]
    gold_idx: int
    n_fixtures: int
    n_raw_candidates: int


def _pick_rep(members: List[object]) -> object:
    for m in members:
        if m.axis == "identity":
            return m
    return sorted(members, key=lambda m: (m.axis, m.op_tag))[0]


def _merge(readings: Sequence[object], vectors: Sequence[Tuple[float, ...]],
           n_fixtures: int, family: str, target: object) -> Pool:
    groups: "OrderedDict[Tuple[float, ...], List[object]]" = OrderedDict()
    for rd, vec in zip(readings, vectors):
        groups.setdefault(vec, []).append(rd)
    classes: List[ReadingClass] = []
    gold_idx = 0
    for vec, members in groups.items():
        rep = _pick_rep(members)
        axes = {m.axis for m in members}
        is_gold = any(m.axis == "identity" for m in members)
        classes.append(ReadingClass(rep=rep, members=members, vector=vec,
                                    axes=axes, is_gold=is_gold))
    for i, c in enumerate(classes):
        if c.is_gold:
            gold_idx = i
            break
    return Pool(family=family, target=target, classes=classes, gold_idx=gold_idx,
                n_fixtures=n_fixtures, n_raw_candidates=len(readings))


# ==========================================================================
# Prometheus evaluator
# ==========================================================================

def _cmp(x: float, op: str, t: float) -> bool:
    if op == ">":
        return x > t
    if op == ">=":
        return x >= t
    if op == "<":
        return x < t
    if op == "<=":
        return x <= t
    if op == "==":
        return x == t
    if op == "!=":
        return x != t
    return False


def _regexify(v: str) -> str:
    return v


def _sel_match(selectors, labels: dict) -> bool:
    for k, op, v in selectors:
        if k not in labels:          # dimension not modelled by this fixture -> satisfied
            continue
        lv = str(labels[k])
        try:
            if op == "=":
                ok = lv == v
            elif op == "!=":
                ok = lv != v
            elif op == "=~":
                ok = re.fullmatch(_regexify(v), lv) is not None
            elif op == "!~":
                ok = re.fullmatch(_regexify(v), lv) is None
            else:
                ok = True
        except re.error:
            ok = True
        if not ok:
            return False
    return True


def _smooth(series: List[float], w_steps: int) -> List[float]:
    if w_steps <= 1:
        return series
    out = []
    for t in range(len(series)):
        window = series[max(0, t - w_steps + 1):t + 1]
        out.append(sum(window) / len(window))
    return out


def _groups(instances: List[dict], reading: PromReading) -> List[List[dict]]:
    if reading.aggregation == "none":
        return [[i] for i in instances]
    if not reading.agg_by:
        return [instances]
    g: "OrderedDict[tuple, List[dict]]" = OrderedDict()
    for i in instances:
        key = tuple(i["labels"].get(k) for k in reading.agg_by)
        g.setdefault(key, []).append(i)
    return list(g.values())


def _agg_series(group: List[dict], chan: str, how: str, w: int) -> List[float]:
    mats = [_smooth(inst[chan], w) for inst in group]
    H = len(mats[0])
    out = []
    for t in range(H):
        vals = [m[t] for m in mats]
        if how == "sum":
            out.append(float(sum(vals)))
        elif how == "max":
            out.append(float(max(vals)))
        elif how == "min":
            out.append(float(min(vals)))
        elif how == "avg":
            out.append(float(sum(vals) / len(vals)))
        else:  # none -> single member
            out.append(float(vals[0]))
    return out


def prom_fire(reading: PromReading, scenario: dict, step_min: float = 1.0) -> int:
    incl = [i for i in scenario["instances"] if _sel_match(reading.selectors, i["labels"])]
    if not incl:
        return 0
    w = max(1, round((reading.rate_window_s or 0) / 60.0)) if reading.rate_window_s else 1
    for_min = reading.for_s / 60.0
    for grp in _groups(incl, reading):
        prim = _agg_series(grp, "series", reading.aggregation, w)
        cond = [_cmp(prim[t], reading.comparator, reading.threshold) for t in range(len(prim))]
        if reading.logic == "conj":
            guard = _agg_series(grp, "guard", "min", 1)
            cond = [cond[t] and guard[t] > 0.5 for t in range(len(cond))]
        elif reading.logic == "disj":
            sec = _agg_series(grp, "secondary", reading.aggregation, w)
            cond = [cond[t] or _cmp(sec[t], reading.comparator, reading.threshold)
                    for t in range(len(cond))]
        # max consecutive run of True
        run = best = 0
        for c in cond:
            run = run + 1 if c else 0
            best = max(best, run)
        if best * step_min >= for_min if for_min > 0 else best >= 1:
            return 1
    return 0


def _scope_dim(selectors) -> Tuple[str, str, str]:
    """(key, matchval, otherval): matchval satisfies the last selector, otherval
    fails it.  Used so identity includes matchval-instances and excludes
    otherval-instances, while a scope-drop candidate includes both."""
    if not selectors:
        return "shard", "a", "b"
    k, op, v = selectors[-1]
    first = v.split("|")[0]
    if op in ("=", "=~"):
        return k, first, "zz_nomatch"
    if op in ("!=", "!~"):
        return k, "zz_ok", first
    return k, first, "zz_nomatch"


def _levels(T: float) -> Dict[str, float]:
    d = max(abs(T) * 0.1, 0.5)
    return dict(a2=T - 2 * d, a1=T - d, eq=T, p1=T + d, p2=T + 2 * d, d=d)


def _const_inst(labels: dict, lvl: float, H: int, sec_base: float,
                guard_val: bool = True, sec_lvl: Optional[float] = None) -> dict:
    return dict(labels=labels,
                series=[lvl] * H,
                guard=[1.0 if guard_val else 0.0] * H,
                secondary=[(sec_lvl if sec_lvl is not None else sec_base)] * H)


def _pulse_inst(labels: dict, lvl: float, k: int, base: float, H: int, sec_base: float) -> dict:
    return dict(labels=labels,
                series=[lvl] * min(k, H) + [base] * max(0, H - k),
                guard=[1.0] * H,
                secondary=[sec_base] * H)


def prom_fixtures(target: PromTarget, neighbours: Sequence[float]) -> List[dict]:
    r = target.reading
    T = r.threshold
    L = _levels(T)
    d = L["d"]
    H = 40
    upper = r.comparator in (">", ">=")
    thrs = sorted(set([T] + list(neighbours)))
    # QUIET baseline must lie beyond every candidate threshold AND every signal
    # level, so no candidate fires on it -- otherwise a threshold below the
    # baseline would fire on the baseline everywhere and spuriously dominate.
    if upper:
        base = min(thrs + [T * 0.5, L["a1"]]) - 2 * d
    else:
        base = max(thrs + [T * 1.5, L["p1"]]) + 2 * d
    sec_base = base
    # scope dimension = the selector candidates add/drop (last selector).
    # matchval must SATISFY the identity selector (so identity includes the hot
    # instance); other must FAIL it (so identity excludes it, drop1 includes it).
    sk, matchval, other = _scope_dim(r.selectors)
    lab_match = {sk: matchval, "cluster": "prod"}
    lab_dev = {sk: matchval, "cluster": "dev"}
    lab_other = {sk: other, "cluster": "prod"}

    scen: List[dict] = []

    def add(sid, instances):
        scen.append(dict(id=sid, instances=instances))

    # 1. constant sustained at each candidate threshold and just above/below it
    #    (separates threshold variants AND the >/>= boundary), plus a1/p1/p2.
    levels = sorted(set(thrs + [x + 0.3 * d for x in thrs] + [x - 0.3 * d for x in thrs]
                        + [L["a1"], L["p1"], L["p2"]]))
    for i, lvl in enumerate(levels):
        add(f"const_{i}", [_const_inst(dict(lab_match), lvl, H, sec_base)])
    # 2. duration sweeps (pulse of length k at p1)
    for k in (1, 3, 6, 12, 20):
        add(f"pulse_{k}", [_pulse_inst(dict(lab_match), L["p1"], k, base, H, sec_base)])
    # 3. spike vs sustained tradeoff
    add("brief_extreme", [_pulse_inst(dict(lab_match), L["p2"] + 2 * L["d"], 4, base, H, sec_base)])
    add("sustained_moderate", [_pulse_inst(dict(lab_match), L["a1"], 25, base, H, sec_base)])
    # 4. multi-instance aggregation
    add("one_hot", [_const_inst(dict(lab_match), L["p1"], H, sec_base),
                    _const_inst(dict(lab_dev), base, H, sec_base),
                    _const_inst(dict(lab_other), base, H, sec_base)])
    spread = (T * 0.5) if (upper and T > 0) else L["a1"]
    add("spread_sum", [_const_inst(dict(lab_match), spread, H, sec_base),
                       _const_inst(dict(lab_dev), spread, H, sec_base),
                       _const_inst(dict(lab_other), spread, H, sec_base)])
    # 5. scope separation
    add("scope_other_hot", [_const_inst(dict(lab_other), L["p1"], H, sec_base),
                            _const_inst(dict(lab_match), base, H, sec_base)])
    add("scope_cluster_dev", [_const_inst(dict(lab_dev), L["p1"], H, sec_base)])
    # 6. logic separation
    add("guard_off", [_const_inst(dict(lab_match), L["p1"], H, sec_base, guard_val=False)])
    add("secondary_hot", [_const_inst(dict(lab_match), base, H, sec_base, sec_lvl=L["p1"])])
    return scen


def build_prom_pool(target: PromTarget, threshold_bank: Dict[str, List[float]]) -> Pool:
    cands = prom_candidates(target, threshold_bank)
    root = target.reading.metric.split("_")[0]
    bank = threshold_bank.get(root) or threshold_bank.get("__all__", [])
    from .candidates import _threshold_neighbours
    neigh = _threshold_neighbours(target.reading.threshold, bank)
    fixtures = prom_fixtures(target, neigh)
    vectors = [tuple(prom_fire(c, s) for s in fixtures) for c in cands]
    return _merge(cands, [tuple(float(x) for x in v) for v in vectors],
                  len(fixtures), "prometheus", target)


# ==========================================================================
# Kyverno evaluator
# ==========================================================================

_ACTION_WEIGHT = {"Enforce": 1.0, "Warn": 0.5, "Audit": 0.25}


def _kyv_violates(reading: KyvReading, r: dict) -> bool:
    fam = reading.family
    if fam == "required_labels":
        return any(k not in r.get("labels", {}) for k in reading.required_labels)
    if fam == "registry":
        reg = r.get("registry", "")
        return not any(reg == a or reg.startswith(a) for a in reading.allowed_registries)
    if fam == "capabilities":
        return any(c not in reading.allowed_caps for c in r.get("caps", []))
    if fam == "resources":
        have = set(r.get("resources", set()))
        return any(req not in have for req in reading.resource_reqs)
    if fam == "privileged":
        return bool(r.get("privileged"))
    if fam == "host_path":
        return bool(r.get("hostPath"))
    if fam == "host_namespaces":
        return bool(r.get("hostNetwork"))
    if fam == "host_ports":
        return bool(r.get("hostPort"))
    if fam == "read_only_fs":
        return bool(r.get("writableRootFs"))
    if fam == "run_as_non_root":
        return bool(r.get("runAsRoot"))
    if fam == "privilege_escalation":
        return bool(r.get("privEscalation"))
    if fam == "seccomp":
        return bool(r.get("seccompUnconfined"))
    if fam == "node_port":
        return bool(r.get("nodePort"))
    return bool(r.get("violates_generic"))


def kyv_cost(reading: KyvReading, r: dict) -> float:
    if r["kind"] not in reading.match_kinds:
        return 0.0
    if r["namespace"] in reading.exclude_ns:
        return 0.0
    if not _kyv_violates(reading, r):
        return 0.0
    return _ACTION_WEIGHT.get(reading.action, 1.0)


def _base_resource(reading: KyvReading, kind: str, ns: str) -> dict:
    return dict(
        kind=kind, namespace=ns,
        labels={k: "x" for k in reading.required_labels} | {"app.kubernetes.io/version": "1"},
        registry=(reading.allowed_registries[0] if reading.allowed_registries else "registry.k8s.io"),
        caps=[], privileged=False, hostPath=False, hostNetwork=False, hostPort=False,
        runAsRoot=False, writableRootFs=False, privEscalation=False,
        seccompUnconfined=False, nodePort=False, resources={"limits", "requests"},
        violates_generic=False)


def _kyv_profiles(reading: KyvReading) -> List[Tuple[str, dict]]:
    """(name, override-dict) list toggling this family's predicate and its set/threshold axes."""
    fam = reading.family
    prof: List[Tuple[str, dict]] = [("comply", {})]
    if fam == "required_labels":
        if reading.required_labels:
            prof.append(("miss_first", {"labels": {k: "x" for k in reading.required_labels[1:]}}))
            prof.append(("miss_last", {"labels": {k: "x" for k in reading.required_labels[:-1]}}))
        prof.append(("miss_version", {"labels": {k: "x" for k in reading.required_labels}}))
    elif fam == "registry":
        prof.append(("disallowed", {"registry": "evil.example.io"}))
        if reading.allowed_registries:
            prof.append(("from_last", {"registry": reading.allowed_registries[-1]}))
        prof.append(("from_ghcr", {"registry": "ghcr.io"}))
    elif fam == "capabilities":
        prof.append(("cap_sysadmin", {"caps": ["SYS_ADMIN"]}))
        prof.append(("cap_netbind", {"caps": ["NET_BIND_SERVICE"]}))
        if reading.allowed_caps:
            prof.append(("cap_last", {"caps": [reading.allowed_caps[-1]]}))
    elif fam == "resources":
        prof.append(("no_limits", {"resources": {"requests"}}))
        prof.append(("no_requests", {"resources": {"limits"}}))
        prof.append(("neither", {"resources": set()}))
    else:
        # boolean / generic families: one violating profile
        field = {"privileged": "privileged", "host_path": "hostPath",
                 "host_namespaces": "hostNetwork", "host_ports": "hostPort",
                 "read_only_fs": "writableRootFs", "run_as_non_root": "runAsRoot",
                 "privilege_escalation": "privEscalation", "seccomp": "seccompUnconfined",
                 "node_port": "nodePort"}.get(fam, "violates_generic")
        prof.append(("violate", {field: True}))
    return prof


def kyv_fixtures(reading: KyvReading) -> List[dict]:
    prim_kind = reading.match_kinds[0] if reading.match_kinds else "Pod"
    kinds = [prim_kind]
    for k in ("Deployment", "Job"):
        if k not in kinds:
            kinds.append(k)
    nss = ["app", "prod"]
    for n in reading.exclude_ns[:1]:
        if n not in nss:
            nss.append(n)
    for n in ("kube-system",):
        if n not in nss:
            nss.append(n)
    profiles = _kyv_profiles(reading)
    fixtures: List[dict] = []
    seen = set()

    def add(kind, ns, name, ov):
        r = _base_resource(reading, kind, ns)
        r.update(ov)
        key = (kind, ns, name)
        if key in seen:
            return
        seen.add(key)
        fixtures.append(r)

    # profiles x namespaces at the primary kind (separates predicate + exception set)
    for name, ov in profiles:
        for ns in nss:
            add(prim_kind, ns, name, ov)
    # a violating profile across kinds (separates kind scope)
    viol = next((p for p in profiles if p[0] != "comply"), profiles[0])
    for kind in kinds[1:]:
        add(kind, "app", "kindscope_" + viol[0], viol[1])
    return fixtures


def build_kyv_pool(target: KyvTarget) -> Pool:
    cands = kyv_candidates(target.reading)
    fixtures = kyv_fixtures(target.reading)
    vectors = [tuple(kyv_cost(c, f) for f in fixtures) for c in cands]
    return _merge(cands, [tuple(round(float(x), 6) for x in v) for v in vectors],
                  len(fixtures), "kyverno", target)


# ==========================================================================
def summarize_pool(p: Pool) -> dict:
    return dict(target=getattr(p.target, "name", getattr(p.target, "policy_name", "?")),
                family=p.family, n_raw=p.n_raw_candidates, n_classes=len(p.classes),
                n_fixtures=p.n_fixtures, gold_idx=p.gold_idx,
                axes=sorted({a for c in p.classes for a in c.axes}))
