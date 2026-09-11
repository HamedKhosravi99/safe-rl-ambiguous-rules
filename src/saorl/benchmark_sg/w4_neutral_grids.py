"""W4 (REGISTRATION_W4): target-independent fixture grids.

The frozen universe (`prom_fixtures`) seeds signal levels and the threshold
grid from psi^src, so fixture points cluster near the gold's thresholds --
precisely where candidate readings separate.  Both the dominance rate and the
free-class screen rate inherit this.  This study re-evaluates the SAME
candidate pools on target-blind universes and reports the paired per-pool
comparison.

REGISTRATION (frozen before any vector is computed; the module writes
`w4_registration.json` first and refuses to run if a registration with
different content already exists):

Arms (all target-blind in signal placement):
  A  a-priori lattice (PRIMARY): per-unit fixed level grids --
       percent [0,100]: 1,5,10,25,50,75,90,95,99
       ratio   [0,1]:   the same /100
       seconds:         0.01,0.1,0.5,1,5,10,60,300,1800
       bytes:           1e3,1e6,1e7,1e8,1e9,1e10,1e11,1e12,5e12
       count (default): 0.01,0.1,0.5,1,5,10,100,1000,10000
     unit inferred from registered metric-name keywords only;
     durations on the fixed operational ladder 1m/5m/15m/1h (the frozen
     evaluator is minute-resolution, so the 0s/30s rungs of the conventional
     ladder coincide with 1m and are represented by it); horizon 80 min;
     quiet baseline 0 for upper comparators, 2x top-of-grid for lower;
     scope/logic/aggregation skeleton scenarios as in the frozen universe
     with hot level = grid[7], spread level = grid[4].
  B  repo-empirical: per metric root, numeric literals harvested from the
     pinned repos' NON-RULE files (dashboards *.json, recording-rule files;
     any file containing an `alert:` entry is excluded); levels = empirical
     quantiles 0.1/0.3/0.5/0.7/0.9 of the harvested set when >= 4 distinct
     values exist (else the arm is unavailable for that pool); durations and
     skeleton as in arm A.
  C  pool-derived, gold excluded: the frozen constructor's shape with the
     anchor threshold replaced by the median of the NON-GOLD candidate
     thresholds and the threshold grid by the non-gold candidate thresholds.
     Isolates leakage-through-the-target from leakage-through-the-pool.
Kyverno analogues: arm A = fixed a-priori resource catalogue (kinds, label
sets, registries, capability sets, resource-block subsets, boolean toggles,
namespaces {app, prod, kube-system, staging}); arm C = union of the
candidate-derived profiles over NON-GOLD candidates; arm B is not defined
(no harvested non-rule value source for admission policies).

Endpoints, computed per family x arm on rebuilt pools (same candidates,
new universe, behaviour classes re-merged):
  1. non-dominance rate (frac_genuinely_non_dominated)
  2. collapse rate
  3. free-class screen fire counts at d in {0.01, 0.02, 0.05, 0.10} (E4 rule)
  4. nu_Pi mean/median + indicator-saturation flag
  5. axis-stratified table (the registered _STRATA)
  6. paired per-pool (gold-centred verdict, arm verdict): McNemar-style
     disagreement counts and the direction of every flip
  7. compiled-screen sensitivity: fraction of pools that are genuinely
     non-dominated under the arm among those non-dominated under the frozen
     universe (the compiled 1/88 screen's precondition)

DECISION RULE (pre-declared): if the arm-A non-dominance rate lies within
+-0.10 of the frozen-universe rate in both families, the headline stands as
written; if outside, the abstract quotes the neutral rate and the
gold-centred rate becomes a sensitivity.

Run:  PYTHONPATH=. python3 -m saorl.benchmark_sg.w4_neutral_grids
Writes results/e2e/w4_registration.json then results/e2e/w4_neutral_grids.json
"""
from __future__ import annotations

import json
import os
import re
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import dominance
from .e4_screen import screen_fires
from .evaluate import (Pool, _merge, prom_fire, kyv_cost, _const_inst,
                       _pulse_inst, _scope_dim)
from .parse import parse_prometheus, parse_kyverno, prom_threshold_bank

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = os.path.join(ROOT, "data", "rule_corpora")
REG_OUT = os.path.join(ROOT, "results/e2e", "w4_registration.json")
OUT = os.path.join(ROOT, "results/e2e", "w4_neutral_grids.json")

BUDGETS = (0.01, 0.02, 0.05, 0.10)

REGISTRATION = {
    "id": "REGISTRATION_W4",
    "primary_arm": "A",
    "primary_endpoint": "non-dominance rate per family, arm A vs frozen universe",
    "decision_rule": ("within +-0.10 in both families -> headline stands; "
                      "outside -> abstract quotes the neutral rate, "
                      "gold-centred becomes a sensitivity"),
    "level_grids": {
        "percent": [1, 5, 10, 25, 50, 75, 90, 95, 99],
        "ratio": [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
        "seconds": [0.01, 0.1, 0.5, 1, 5, 10, 60, 300, 1800],
        "bytes": [1e3, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 5e12],
        "count": [0.01, 0.1, 0.5, 1, 5, 10, 100, 1000, 10000],
    },
    "unit_keywords": {
        "percent": ["percent", "utilization", "utilisation", "cpu_usage"],
        "ratio": ["ratio", "fraction", "availability"],
        "seconds": ["seconds", "latency", "duration", "_time"],
        "bytes": ["bytes", "memory_", "_size"],
    },
    "duration_ladder_min": [1, 5, 15, 60],
    "horizon_min": 80,
    "hot_quantile_index": 7, "spread_quantile_index": 4,
    "armB_quantiles": [0.1, 0.3, 0.5, 0.7, 0.9], "armB_min_distinct": 4,
    "screen_budgets": list(BUDGETS),
    "kyverno_catalogue": {
        "bases": [["Pod", "app"], ["Pod", "kube-system"], ["Deployment", "prod"],
                  ["Job", "staging"], ["StatefulSet", "app"]],
        "label_sets": [[], ["app.kubernetes.io/name"],
                       ["app.kubernetes.io/name", "team"],
                       ["app.kubernetes.io/name", "team", "owner",
                        "app.kubernetes.io/version"]],
        "registries": ["registry.k8s.io", "docker.io", "ghcr.io", "evil.example.io"],
        "cap_sets": [[], ["NET_BIND_SERVICE"], ["SYS_ADMIN"], ["CHOWN", "NET_ADMIN"]],
        "resource_sets": [["limits", "requests"], ["limits"], ["requests"], []],
        "booleans": ["privileged", "hostPath", "hostNetwork", "hostPort",
                     "runAsRoot", "writableRootFs", "privEscalation",
                     "seccompUnconfined", "nodePort"],
    },
}


# ---------------------------------------------------------------------------
# Registration handling: publish the rule, then the result, in that order
# ---------------------------------------------------------------------------

def write_registration() -> None:
    os.makedirs(os.path.dirname(REG_OUT), exist_ok=True)
    if os.path.exists(REG_OUT):
        prev = json.load(open(REG_OUT))
        if prev != REGISTRATION:
            raise SystemExit("w4_registration.json exists with DIFFERENT content; "
                             "refusing to silently re-register")
    else:
        json.dump(REGISTRATION, open(REG_OUT, "w"), indent=1)
    print("[w4] registration frozen at", REG_OUT)


# ---------------------------------------------------------------------------
# Prometheus arm A: a-priori lattice
# ---------------------------------------------------------------------------

def infer_unit(metric: str) -> str:
    m = metric.lower()
    for unit, keys in REGISTRATION["unit_keywords"].items():
        if any(k in m for k in keys):
            return unit
    return "count"


def prom_fixtures_armA(target) -> List[dict]:
    r = target.reading
    grid = [float(x) for x in REGISTRATION["level_grids"][infer_unit(r.metric)]]
    H = REGISTRATION["horizon_min"]
    upper = r.comparator in (">", ">=")
    base = 0.0 if upper else 2.0 * grid[-1]
    sec_base = base
    hot = grid[REGISTRATION["hot_quantile_index"]] if upper else grid[1]
    spread = grid[REGISTRATION["spread_quantile_index"]]
    sk, matchval, other = _scope_dim(r.selectors)
    lab_match = {sk: matchval, "cluster": "prod"}
    lab_dev = {sk: matchval, "cluster": "dev"}
    lab_other = {sk: other, "cluster": "prod"}
    scen: List[dict] = []

    def add(sid, instances):
        scen.append(dict(id=sid, instances=instances))

    for i, lvl in enumerate(grid):
        add(f"const_{i}", [_const_inst(dict(lab_match), lvl, H, sec_base)])
    for k in REGISTRATION["duration_ladder_min"]:
        for i, lvl in enumerate(grid):
            add(f"pulse_{k}_{i}", [_pulse_inst(dict(lab_match), lvl, k, base, H, sec_base)])
    add("one_hot", [_const_inst(dict(lab_match), hot, H, sec_base),
                    _const_inst(dict(lab_dev), base, H, sec_base),
                    _const_inst(dict(lab_other), base, H, sec_base)])
    add("spread_sum", [_const_inst(dict(lab_match), spread, H, sec_base),
                       _const_inst(dict(lab_dev), spread, H, sec_base),
                       _const_inst(dict(lab_other), spread, H, sec_base)])
    add("scope_other_hot", [_const_inst(dict(lab_other), hot, H, sec_base),
                            _const_inst(dict(lab_match), base, H, sec_base)])
    add("scope_cluster_dev", [_const_inst(dict(lab_dev), hot, H, sec_base)])
    add("guard_off", [_const_inst(dict(lab_match), hot, H, sec_base, guard_val=False)])
    add("secondary_hot", [_const_inst(dict(lab_match), base, H, sec_base, sec_lvl=hot)])
    return scen


# ---------------------------------------------------------------------------
# Prometheus arm B: repo-empirical harvest (non-rule files only)
# ---------------------------------------------------------------------------

_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?(?:e[+-]?\d+)?)(?![\w.])", re.I)


def harvest_literals(roots: List[str]) -> Dict[str, List[float]]:
    """Numbers co-occurring on a line with a token of the metric root, from
    non-rule files under the pinned repos (files containing `alert:` are
    excluded, so alert expressions never contribute)."""
    hits: Dict[str, set] = {r: set() for r in roots}
    pats = {r: re.compile(r"\b" + re.escape(r) + r"[a-z0-9_:]*", re.I) for r in roots}
    for dirpath, dirnames, filenames in os.walk(SRC):
        dirnames[:] = [d for d in dirnames if d not in (".git",)]
        for fn in filenames:
            if not fn.endswith((".json", ".yml", ".yaml", ".libsonnet", ".jsonnet")):
                continue
            path = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(path) > 3_000_000:
                    continue
                text = open(path, errors="ignore").read()
            except OSError:
                continue
            if re.search(r"^\s*-?\s*alert\s*:", text, re.M):
                continue                       # rule file: excluded
            for line in text.splitlines():
                for root, pat in pats.items():
                    if pat.search(line):
                        for m in _NUM.finditer(line):
                            v = float(m.group(1))
                            if 0 < v < 1e13:
                                hits[root].add(v)
    return {r: sorted(v) for r, v in hits.items()}


def prom_fixtures_armB(target, lits: List[float]) -> Optional[List[dict]]:
    if len(set(lits)) < REGISTRATION["armB_min_distinct"]:
        return None
    qs = REGISTRATION["armB_quantiles"]
    arr = np.array(sorted(lits), dtype=float)
    grid = sorted(set(float(np.quantile(arr, q)) for q in qs))
    if len(grid) < 3:
        return None
    r = target.reading
    H = REGISTRATION["horizon_min"]
    upper = r.comparator in (">", ">=")
    base = 0.0 if upper else 2.0 * grid[-1]
    hot = grid[-1] if upper else grid[0]
    spread = grid[len(grid) // 2]
    sk, matchval, other = _scope_dim(r.selectors)
    lab_match = {sk: matchval, "cluster": "prod"}
    lab_dev = {sk: matchval, "cluster": "dev"}
    lab_other = {sk: other, "cluster": "prod"}
    scen: List[dict] = []

    def add(sid, instances):
        scen.append(dict(id=sid, instances=instances))

    for i, lvl in enumerate(grid):
        add(f"const_{i}", [_const_inst(dict(lab_match), lvl, H, base)])
    for k in REGISTRATION["duration_ladder_min"]:
        for i, lvl in enumerate(grid):
            add(f"pulse_{k}_{i}", [_pulse_inst(dict(lab_match), lvl, k, base, H, base)])
    add("one_hot", [_const_inst(dict(lab_match), hot, H, base),
                    _const_inst(dict(lab_dev), base, H, base),
                    _const_inst(dict(lab_other), base, H, base)])
    add("spread_sum", [_const_inst(dict(lab_match), spread, H, base),
                       _const_inst(dict(lab_dev), spread, H, base),
                       _const_inst(dict(lab_other), spread, H, base)])
    add("scope_other_hot", [_const_inst(dict(lab_other), hot, H, base),
                            _const_inst(dict(lab_match), base, H, base)])
    add("scope_cluster_dev", [_const_inst(dict(lab_dev), hot, H, base)])
    add("guard_off", [_const_inst(dict(lab_match), hot, H, base, guard_val=False)])
    add("secondary_hot", [_const_inst(dict(lab_match), base, H, base, sec_lvl=hot)])
    return scen


# ---------------------------------------------------------------------------
# Prometheus arm C: pool-derived, gold excluded
# ---------------------------------------------------------------------------

def prom_fixtures_armC(target, cand_thresholds: List[float]) -> Optional[List[dict]]:
    r = target.reading
    thrs = sorted(set(round(t, 9) for t in cand_thresholds
                      if abs(t - r.threshold) > 1e-12))
    if len(thrs) < 2:
        return None
    T = float(median(thrs))
    d = max(abs(T) * 0.1, 0.5)
    H = 40
    upper = r.comparator in (">", ">=")
    if upper:
        base = min(thrs + [T * 0.5, T - d]) - 2 * d
    else:
        base = max(thrs + [T * 1.5, T + d]) + 2 * d
    sec_base = base
    sk, matchval, other = _scope_dim(r.selectors)
    lab_match = {sk: matchval, "cluster": "prod"}
    lab_dev = {sk: matchval, "cluster": "dev"}
    lab_other = {sk: other, "cluster": "prod"}
    scen: List[dict] = []

    def add(sid, instances):
        scen.append(dict(id=sid, instances=instances))

    levels = sorted(set(thrs + [x + 0.3 * d for x in thrs] + [x - 0.3 * d for x in thrs]
                        + [T - d, T + d, T + 2 * d]))
    for i, lvl in enumerate(levels):
        add(f"const_{i}", [_const_inst(dict(lab_match), lvl, H, sec_base)])
    for k in (1, 3, 6, 12, 20):
        add(f"pulse_{k}", [_pulse_inst(dict(lab_match), T + d, k, base, H, sec_base)])
    add("brief_extreme", [_pulse_inst(dict(lab_match), T + 4 * d, 4, base, H, sec_base)])
    add("sustained_moderate", [_pulse_inst(dict(lab_match), T - d, 25, base, H, sec_base)])
    add("one_hot", [_const_inst(dict(lab_match), T + d, H, sec_base),
                    _const_inst(dict(lab_dev), base, H, sec_base),
                    _const_inst(dict(lab_other), base, H, sec_base)])
    spread = (T * 0.5) if (upper and T > 0) else T - d
    add("spread_sum", [_const_inst(dict(lab_match), spread, H, sec_base),
                       _const_inst(dict(lab_dev), spread, H, sec_base),
                       _const_inst(dict(lab_other), spread, H, sec_base)])
    add("scope_other_hot", [_const_inst(dict(lab_other), T + d, H, sec_base),
                            _const_inst(dict(lab_match), base, H, sec_base)])
    add("scope_cluster_dev", [_const_inst(dict(lab_dev), T + d, H, sec_base)])
    add("guard_off", [_const_inst(dict(lab_match), T + d, H, sec_base, guard_val=False)])
    add("secondary_hot", [_const_inst(dict(lab_match), base, H, sec_base, sec_lvl=T + d)])
    return scen


# ---------------------------------------------------------------------------
# Kyverno arms
# ---------------------------------------------------------------------------

def _neutral_resource(kind: str, ns: str, labels: List[str], registry: str,
                      caps: List[str], resources: List[str], **flags) -> dict:
    r = dict(kind=kind, namespace=ns,
             labels={k: "x" for k in labels},
             registry=registry, caps=list(caps),
             privileged=False, hostPath=False, hostNetwork=False, hostPort=False,
             runAsRoot=False, writableRootFs=False, privEscalation=False,
             seccompUnconfined=False, nodePort=False,
             resources=set(resources), violates_generic=False)
    r.update(flags)
    return r


def kyv_fixtures_armA() -> List[dict]:
    cat = REGISTRATION["kyverno_catalogue"]
    L3 = cat["label_sets"][3]
    fixtures: List[dict] = []
    # base compliant + generic violation per (kind, ns)
    for kind, ns in cat["bases"]:
        fixtures.append(_neutral_resource(kind, ns, L3, cat["registries"][0],
                                          [], cat["resource_sets"][0]))
        fixtures.append(_neutral_resource(kind, ns, L3, cat["registries"][0],
                                          [], cat["resource_sets"][0],
                                          violates_generic=True))
    # single-axis mutations on the (Pod, app) base
    for ls in cat["label_sets"]:
        fixtures.append(_neutral_resource("Pod", "app", ls, cat["registries"][0],
                                          [], cat["resource_sets"][0]))
    for reg in cat["registries"][1:]:
        fixtures.append(_neutral_resource("Pod", "app", L3, reg,
                                          [], cat["resource_sets"][0]))
    for caps in cat["cap_sets"][1:]:
        fixtures.append(_neutral_resource("Pod", "app", L3, cat["registries"][0],
                                          caps, cat["resource_sets"][0]))
    for rs in cat["resource_sets"][1:]:
        fixtures.append(_neutral_resource("Pod", "app", L3, cat["registries"][0],
                                          [], rs))
    for flag in cat["booleans"]:
        fixtures.append(_neutral_resource("Pod", "app", L3, cat["registries"][0],
                                          [], cat["resource_sets"][0], **{flag: True}))
    return fixtures


def kyv_fixtures_armC(cands, gold) -> List[dict]:
    """Union of candidate-derived fixtures over non-gold candidates."""
    from .evaluate import kyv_fixtures
    seen = set()
    out: List[dict] = []
    for c in cands:
        if c.struct_key() == gold.struct_key():
            continue
        for f in kyv_fixtures(c):
            key = json.dumps({k: sorted(v) if isinstance(v, set) else v
                              for k, v in f.items()}, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                out.append(f)
    return out


# ---------------------------------------------------------------------------
# Pool rebuild + endpoints
# ---------------------------------------------------------------------------

def rebuild_prom(target, cands, fixtures) -> Pool:
    vectors = [tuple(float(prom_fire(c, s)) for s in fixtures) for c in cands]
    return _merge(cands, vectors, len(fixtures), "prometheus", target)


def rebuild_kyv(target, cands, fixtures) -> Pool:
    vectors = [tuple(round(float(kyv_cost(c, f)), 6) for f in fixtures) for c in cands]
    return _merge(cands, vectors, len(fixtures), "kyverno", target)


def pool_endpoints(pool: Pool) -> dict:
    vecs = [c.vector for c in pool.classes]
    dom = dominance.pool_dominance(vecs)
    return dict(n_classes=len(pool.classes),
                non_dominated=bool(dom["genuinely_non_dominated"]),
                collapses=bool(dom["collapses"]),
                nu_pi=float(dom["nu_pi"]),
                width=int(dom["antichain_width"]),
                fires={str(d): bool(screen_fires(vecs, d)) for d in BUDGETS})


def family_summary(rows: List[dict], frozen_nd: List[bool]) -> dict:
    nd = [r["non_dominated"] for r in rows]
    nus = np.array([r["nu_pi"] for r in rows])
    flips_gain = sum(1 for f, a in zip(frozen_nd, nd) if (not f) and a)
    flips_loss = sum(1 for f, a in zip(frozen_nd, nd) if f and (not a))
    both = sum(1 for f, a in zip(frozen_nd, nd) if f and a)
    neither = sum(1 for f, a in zip(frozen_nd, nd) if (not f) and (not a))
    kept = [a for f, a in zip(frozen_nd, nd) if f]
    return dict(
        n_pools=len(rows),
        non_dominance_rate=float(np.mean(nd)),
        collapse_rate=float(np.mean([r["collapses"] for r in rows])),
        nu_pi_mean=float(nus.mean()), nu_pi_median=float(np.median(nus)),
        nu_pi_saturated=bool(np.all(np.isin(nus[nus > 0], [1.0]))) if (nus > 0).any() else None,
        screen_fire_counts={str(d): int(sum(r["fires"][str(d)] for r in rows))
                            for d in BUDGETS},
        paired=dict(both_nd=both, neither_nd=neither,
                    flip_to_nd=flips_gain, flip_to_dominated=flips_loss),
        compiled_precondition_survives=(float(np.mean(kept)) if kept else None),
        mean_classes=float(np.mean([r["n_classes"] for r in rows])),
    )


def strata_table(pools: List[Pool], family: str) -> dict:
    from .run_benchmark import _STRATA
    out = {}
    for name, axes in _STRATA[family]:
        nds, ns = [], []
        for p in pools:
            keep = {"identity"} | axes
            sv = [p.classes[i].vector for i in range(len(p.classes))
                  if frozenset(p.classes[i].axes) & keep]
            sv = list(dict.fromkeys(sv))
            if len(sv) < 2:
                continue
            d = dominance.pool_dominance(sv)
            nds.append(d["genuinely_non_dominated"])
            ns.append(len(sv))
        if ns:
            out[name] = dict(n_pools=len(ns),
                             frac_non_dominated=round(float(np.mean(nds)), 3))
    return out


def main() -> None:
    write_registration()
    from .candidates import prom_candidates, kyv_candidates, _threshold_neighbours
    from . import run_benchmark as rb

    pt, _ = parse_prometheus()
    kt, _ = parse_kyverno()
    bank = prom_threshold_bank(pt)

    # frozen-universe reference verdicts (the published pools)
    frozen_prom = [rb._record(rb.build_prom_pool(t, bank), t, rb._prom_skeleton(t)) for t in pt]
    frozen_kyv = [rb._record(rb.build_kyv_pool(t), t, rb._kyv_skeleton(t)) for t in kt]
    frozen_nd_prom = [bool(r.dom["genuinely_non_dominated"]) for r in frozen_prom]
    frozen_nd_kyv = [bool(r.dom["genuinely_non_dominated"]) for r in frozen_kyv]
    frozen_rates = dict(
        prometheus=float(np.mean(frozen_nd_prom)),
        kyverno=float(np.mean(frozen_nd_kyv)))
    print(f"[w4] frozen-universe non-dominance: prom {frozen_rates['prometheus']:.3f} "
          f"kyv {frozen_rates['kyverno']:.3f}")

    roots = sorted({t.reading.metric.split("_")[0] for t in pt})
    print(f"[w4] harvesting non-rule literals for {len(roots)} metric roots ...")
    harvest = harvest_literals(roots)
    n_cov = sum(1 for r in roots if len(set(harvest[r])) >= REGISTRATION["armB_min_distinct"])
    print(f"[w4] arm B coverage: {n_cov}/{len(roots)} roots")

    out = {"registration": "REGISTRATION_W4 (see w4_registration.json)",
           "frozen_rates": frozen_rates, "arms": {}}

    # ---- Prometheus arms (aligned with frozen verdicts pool-by-pool)
    for arm in ("A", "B", "C"):
        pools, rows, frozen_sub = [], [], []
        for i, t in enumerate(pt):
            cands = prom_candidates(t, bank)
            if arm == "A":
                fx = prom_fixtures_armA(t)
            elif arm == "B":
                root = t.reading.metric.split("_")[0]
                fx = prom_fixtures_armB(t, harvest.get(root, []))
            else:
                fx = prom_fixtures_armC(t, [c.threshold for c in cands])
            if fx is None:
                continue
            p = rebuild_prom(t, cands, fx)
            pools.append(p)
            rows.append(pool_endpoints(p))
            frozen_sub.append(frozen_nd_prom[i])
        avail = len(rows)
        summ = family_summary(rows, frozen_sub)
        summ["n_available"] = avail
        summ["strata"] = strata_table(pools, "prometheus")
        out["arms"].setdefault(arm, {})["prometheus"] = summ
        print(f"[w4 prom arm {arm}] n={avail} nd={summ['non_dominance_rate']:.3f} "
              f"collapse={summ['collapse_rate']:.3f} fires@0.05="
              f"{summ['screen_fire_counts']['0.05']}/{avail} "
              f"flips +{summ['paired']['flip_to_nd']}/-{summ['paired']['flip_to_dominated']}")

    # ---- Kyverno arms (A and C), aligned with frozen verdicts pool-by-pool
    for arm in ("A", "C"):
        pools, rows, frozen_sub = [], [], []
        fxA = kyv_fixtures_armA() if arm == "A" else None
        for i, t in enumerate(kt):
            cands = kyv_candidates(t.reading)
            fx = fxA if arm == "A" else kyv_fixtures_armC(cands, t.reading)
            if not fx:
                continue
            p = rebuild_kyv(t, cands, fx)
            pools.append(p)
            rows.append(pool_endpoints(p))
            frozen_sub.append(frozen_nd_kyv[i])
        summ = family_summary(rows, frozen_sub)
        summ["n_available"] = len(rows)
        summ["strata"] = strata_table(pools, "kyverno")
        out["arms"].setdefault(arm, {})["kyverno"] = summ
        print(f"[w4 kyv arm {arm}] n={len(rows)} nd={summ['non_dominance_rate']:.3f} "
              f"collapse={summ['collapse_rate']:.3f} fires@0.05="
              f"{summ['screen_fire_counts']['0.05']}/{len(rows)} "
              f"flips +{summ['paired']['flip_to_nd']}/-{summ['paired']['flip_to_dominated']}")

    # decision rule evaluation (arm A primary)
    dA = out["arms"]["A"]
    within = (abs(dA["prometheus"]["non_dominance_rate"] - frozen_rates["prometheus"]) <= 0.10
              and abs(dA["kyverno"]["non_dominance_rate"] - frozen_rates["kyverno"]) <= 0.10)
    out["decision"] = dict(
        rule="+-0.10 both families, arm A",
        prom_delta=float(dA["prometheus"]["non_dominance_rate"] - frozen_rates["prometheus"]),
        kyv_delta=float(dA["kyverno"]["non_dominance_rate"] - frozen_rates["kyverno"]),
        headline_stands=bool(within))
    print(f"[w4] decision: headline_stands={within} "
          f"(prom {out['decision']['prom_delta']:+.3f}, kyv {out['decision']['kyv_delta']:+.3f})")
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
