"""P1 control suite: source-grounded semi-synthetic control instances.

Executes the frozen protocol of paper/Plans/EVIDENCE_FREEZE_MANIFEST.md
(committed 91074b3, before any outcome below was computed):

  * universe: the 88 pinned Prometheus pools;
  * eligibility (outcome-blind, from parse trees + pointwise dominance):
      (i)  >= 2 maximal readings;
      (ii) all maximal readings share (metric, selectors, aggregation),
           comparator '>'/'>=', and differ only in (threshold, for_s);
      (iii) 2 <= |Max| <= 4 with strictly ordered thresholds;
      (iv) at least one crossing pair (theta_i < theta_j, for_i > for_j);
  * diversity: <= 3 instances per source family, deterministic order,
    first <= 20 eligible;
  * compilation: fixed load-chain template (control_mdp's CPU instance is
    the L = 3 special case), gamma = 0.97, R_cont = 1, R_int = 0.3;
  * solved objects: exact occupancy LPs (HiGHS) -- unconstrained,
    singleton oracles, robust over Max, all pairwise, nu_Pi, and the
    pinned oracle-selected singleton (full-set evaluation);
  * primary endpoint at d* = 0.01; grid {0.005, 0.01, 0.02, 0.05}.

Run: SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.control_suite
Writes results/conformal/benchmark_sg/control_suite.json.
"""
from __future__ import annotations

import json
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

from .control_mdp import GAMMA, R_CONT, R_INT  # frozen constants (0.97, 1.0, 0.3)
from .dominance import pool_dominance
from .evaluate import build_prom_pool
from .parse import parse_prometheus, prom_threshold_bank

_ROOT = Path(__file__).resolve().parents[3]
_OUT = _ROOT / "results/conformal" / "benchmark_sg" / "control_suite.json"

BUDGETS = (0.005, 0.01, 0.02, 0.05)
D_STAR = 0.01
MAX_INSTANCES = 20
PER_FAMILY_CAP = 3
DUR_CAP = 8
TOL = 1e-9


# ---------------------------------------------------------------------------
# eligibility (manifest criteria; no policy is run here)
# ---------------------------------------------------------------------------

def _maximal_readings(pool) -> Tuple[List[object], List[Tuple[float, ...]]]:
    reps = [c.rep for c in pool.classes]
    vecs = [tuple(c.vector) for c in pool.classes]
    keep_r, keep_v = [], []
    for i, vi in enumerate(vecs):
        dominated = any(
            j != i
            and all(a <= b for a, b in zip(vi, vecs[j]))
            and any(b > a for a, b in zip(vi, vecs[j]))
            for j in range(len(vecs))
        )
        if not dominated:
            keep_r.append(reps[i])
            keep_v.append(vi)
    return keep_r, keep_v


def _eligible(pool) -> Tuple[bool, str, List[object]]:
    # Amendment A1: restrict to the gold reading's threshold/duration
    # subfamily (same metric/selectors/aggregation, upper comparator),
    # then take maximal elements of THAT subfamily on the fixture universe.
    gold = pool.classes[pool.gold_idx].rep
    if gold.comparator not in (">", ">="):
        return False, "non-upper comparator", []
    fam = [(c.rep, tuple(c.vector)) for c in pool.classes
           if (c.rep.metric, c.rep.selectors, c.rep.aggregation, c.rep.agg_by)
           == (gold.metric, gold.selectors, gold.aggregation, gold.agg_by)
           and c.rep.comparator in (">", ">=")]
    if len(fam) < 2:
        return False, "singleton subfamily", []
    mx = []
    for i, (ri, vi) in enumerate(fam):
        dominated = any(
            j != i
            and all(a <= b for a, b in zip(vi, fam[j][1]))
            and any(b > a for a, b in zip(vi, fam[j][1]))
            for j in range(len(fam)))
        if not dominated:
            mx.append(ri)
    if len(mx) < 2:
        return False, "singleton antichain", []
    # A2(a): same-theta dedup by min for_s (pointwise dominance in the
    # compiled MDP at equal threshold)
    by_theta = {}
    for m in mx:
        th = float(m.threshold)
        if th not in by_theta or float(m.for_s) < float(by_theta[th].for_s):
            by_theta[th] = m
    mx = list(by_theta.values())
    keyed = sorted({(float(m.threshold), float(m.for_s)) for m in mx})
    if not (2 <= len(keyed) <= 4):
        return False, f"|Max|={len(keyed)} outside [2,4]", []
    crossing = any(
        (t1 < t2 and f1 > f2) or (t2 < t1 and f2 > f1)
        for (t1, f1), (t2, f2) in combinations(keyed, 2)
    )
    if not crossing:
        return False, "no threshold/duration crossing", []
    mx_sorted = sorted(
        {(float(m.threshold), float(m.for_s)): m for m in mx}.items())
    return True, "", [m for _k, m in mx_sorted]


# ---------------------------------------------------------------------------
# fixed compiler: readings -> load-chain MDP (manifest formula)
# ---------------------------------------------------------------------------

def _load_matrix(L: int) -> np.ndarray:
    """Fixed template: interior persistence 0.70; top level persistence
    0.40 with spike-decay 0.55 to level 0; remaining mass split toward
    adjacent levels proportionally (L = 3 reproduces the CPU instance's
    qualitative structure: sustained interiors, spiky top)."""
    P = np.zeros((L, L))
    for i in range(L):
        if i == L - 1:
            P[i, i] = 0.40
            P[i, 0] += 0.55
            rest = 0.05
        else:
            P[i, i] = 0.70 if i > 0 else 0.55
            rest = 1.0 - P[i, i]
        others = [j for j in range(L) if j != i and not (i == L - 1 and j == 0)]
        if i == L - 1:
            others = [j for j in range(1, L - 1)]
            if not others:
                P[i, 0] += rest
                continue
        w = np.array([1.0 / (1 + abs(i - j)) for j in others])
        w = w / w.sum() * rest
        for j, wj in zip(others, w):
            P[i, j] += wj
    P = P / P.sum(axis=1, keepdims=True)
    return P


def compile_instance(readings: Sequence[object]) -> dict:
    keyed = sorted({(float(m.threshold), float(m.for_s)) for m in readings})
    thetas = [k[0] for k in keyed]
    fors = [k[1] for k in keyed]
    # A4: fixed physical unit (60 s/step); A3 ordinal rank as tie-breaker
    STEP_S = 60.0
    caps = [max(1, min(DUR_CAP, int(round(f / STEP_S)))) for f in fors]
    uniq_f = sorted(set(fors))
    if len(set(caps)) < len(set(fors)):
        seen = {}
        for i, f in enumerate(fors):
            c = caps[i]
            for g, cg in seen.items():
                if f != g and c == cg:
                    c = max(1, min(DUR_CAP, 1 + uniq_f.index(f)))
            caps[i] = c
            seen[f] = caps[i]
    K = len(keyed)
    uniq = sorted(set(thetas))
    lvl_of = [1 + uniq.index(t) for t in thetas]   # A2(b)
    L = len(uniq) + 1               # level 0 below all thresholds
    loadP = _load_matrix(L)
    # states: ('safe',) + (load, c_1..c_K) with c_k <= caps[k]
    ranges = [range(c + 1) for c in caps]
    S: List[tuple] = [("safe",)]
    for load in range(L):
        for cs in product(*ranges):
            S.append((load,) + cs)
    idx = {s: i for i, s in enumerate(S)}
    nS, nA = len(S), 2
    P = np.zeros((nS, nA, nS))
    r = np.zeros((nS, nA))
    C = np.zeros((K, nS, nA))
    for s in S:
        i = idx[s]
        if s[0] == "safe":
            P[i, :, i] = 1.0
            r[i, :] = R_INT
            continue
        load, cs = s[0], s[1:]
        for load2 in range(L):
            cs2 = tuple(
                min(caps[k], cs[k] + 1) if load2 >= lvl_of[k] else 0
                for k in range(K))
            P[i, 0, idx[(load2,) + cs2]] += loadP[load, load2]
        r[i, 0] = R_CONT
        for k in range(K):
            C[k, i, 0] = 1.0 if cs[k] >= caps[k] else 0.0
        P[i, 1, idx[("safe",)]] = 1.0
        r[i, 1] = R_INT
    mu0 = np.zeros(nS)
    # fresh top-level spike and fresh level-1 ramp, equally weighted
    top = (L - 1,) + tuple(min(caps[k], 1) if (L - 1) >= lvl_of[k] else 0
                           for k in range(K))
    ramp = (1,) + tuple(min(caps[k], 1) if 1 >= lvl_of[k] else 0
                        for k in range(K))
    mu0[idx[top]] = 0.5
    mu0[idx[ramp]] += 0.5
    return dict(S=S, nS=nS, nA=nA, P=P, r=r, C=C, mu0=mu0,
                thetas=thetas, fors=fors, caps=caps, L=L)


# ---------------------------------------------------------------------------
# exact occupancy LPs
# ---------------------------------------------------------------------------

def _flow(m: dict):
    nS, nA = m["nS"], m["nA"]
    A_eq = np.zeros((nS, nS * nA))
    for sp in range(nS):
        A_eq[sp, sp * nA:(sp + 1) * nA] += 1.0
    A_eq -= GAMMA * np.transpose(m["P"], (2, 0, 1)).reshape(nS, nS * nA)
    return A_eq, (1.0 - GAMMA) * m["mu0"]


def solve(m: dict, budgets: Dict[int, float]) -> dict:
    A_eq, b_eq = _flow(m)
    r = m["r"].reshape(-1)
    A_ub = np.array([m["C"][k].reshape(-1) for k in sorted(budgets)]) \
        if budgets else None
    b_ub = np.array([budgets[k] for k in sorted(budgets)]) if budgets else None
    res = linprog(-r, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, None), method="highs")
    if res.status != 0:
        return dict(feasible=False)
    x = res.x
    J = [float(m["C"][k].reshape(-1) @ x) for k in range(m["C"].shape[0])]
    return dict(feasible=True, ret=float(r @ x), J=J)


def analyze_instance(m: dict, d: float) -> dict:
    K = m["C"].shape[0]
    unc = solve(m, {})
    singles = [solve(m, {k: d}) for k in range(K)]
    robust = solve(m, {k: d for k in range(K)})
    pairs = {f"{a}+{b}": solve(m, {a: d, b: d})["ret"]
             for a, b in combinations(range(K), 2)} if K > 2 else {}
    # nu_Pi at this budget's policy class is budget-free by definition; use
    # the standard sup over occupancies: min_k sup_mu [max_j J_j - J_k]
    nu = None
    A_eq, b_eq = _flow(m)
    vals = []
    for k in range(K):
        best = -np.inf
        for j in range(K):
            if j == k:
                continue
            c = (m["C"][j] - m["C"][k]).reshape(-1)
            res = linprog(-c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None),
                          method="highs")
            if res.status == 0:
                best = max(best, float(c @ res.x))
        vals.append(best)
    nu = float(min(vals)) if vals else 0.0
    # pinned oracle-selected singleton: full-set evaluation of each
    # singleton-constrained policy
    oracle = None
    feas_rets = []
    for k, s in enumerate(singles):
        if not s["feasible"]:
            continue
        worst = max(s["J"])
        entry = dict(k=k, ret=s["ret"], worst=worst,
                     fullset_feasible=bool(worst <= d + TOL))
        feas_rets.append(entry)
    feas = [e for e in feas_rets if e["fullset_feasible"]]
    if feas:
        oracle = max(feas, key=lambda e: e["ret"])
    elif feas_rets:
        oracle = min(feas_rets, key=lambda e: e["worst"])
    gap = (robust["ret"] - oracle["ret"]) if (oracle and oracle["fullset_feasible"]) else None
    return dict(
        unconstrained=unc["ret"],
        singleton_rets=[s["ret"] if s["feasible"] else None for s in singles],
        singleton_worst=[max(s["J"]) if s["feasible"] else None for s in singles],
        robust_ret=robust["ret"] if robust["feasible"] else None,
        robust_feasible=robust["feasible"],
        pairwise=pairs,
        nu_pi=nu,
        oracle_singleton=oracle,
        no_singleton_fullset_feasible=bool(not feas),
        robust_minus_oracle_gap=(None if gap is None else float(gap)),
    )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main() -> None:
    targets, _stats = parse_prometheus()
    bank = prom_threshold_bank(targets)
    per_family: Dict[str, int] = {}
    selected, skipped = [], {}
    for t in sorted(targets, key=lambda t: (t.file, t.group, t.name)):
        pool = build_prom_pool(t, bank)
        ok, why, mx = _eligible(pool)
        if not ok:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        fam = t.file
        if per_family.get(fam, 0) >= PER_FAMILY_CAP:
            skipped["family cap"] = skipped.get("family cap", 0) + 1
            continue
        per_family[fam] = per_family.get(fam, 0) + 1
        selected.append((t, mx))
        if len(selected) >= MAX_INSTANCES:
            break

    out = dict(config=dict(
        manifest="paper/Plans/EVIDENCE_FREEZE_MANIFEST.md@91074b3",
        gamma=GAMMA, r_cont=R_CONT, r_int=R_INT, budgets=list(BUDGETS),
        d_star=D_STAR, max_instances=MAX_INSTANCES,
        per_family_cap=PER_FAMILY_CAP, dur_cap=DUR_CAP),
        n_universe=len(targets), skipped=skipped,
        n_selected=len(selected), instances=[])

    gaps, nus, no_feas = [], [], 0
    for t, mx in selected:
        m = compile_instance(mx)
        inst = dict(uid=t.name, family=t.file, group=t.group,
                    geometry=None,
                    metric=mx[0].metric,
                    readings=[dict(theta=float(r.threshold),
                                   for_s=float(r.for_s)) for r in mx],
                    caps=m["caps"], n_states=m["nS"],
                    budgets={})
        inst["geometry"] = f"L{m['L']}-caps{'-'.join(map(str,m['caps']))}"
        for d in BUDGETS:
            inst["budgets"][str(d)] = analyze_instance(m, d)
        prim = inst["budgets"][str(D_STAR)]
        if prim["no_singleton_fullset_feasible"]:
            no_feas += 1
        if prim["robust_minus_oracle_gap"] is not None:
            gaps.append(prim["robust_minus_oracle_gap"])
        nus.append(prim["nu_pi"])
        out["instances"].append(inst)

    n = len(selected)
    g = np.array([x for x in gaps])
    geoms = sorted({i["geometry"] for i in out["instances"]})
    out["distinct_geometries"] = geoms
    out["primary_endpoint"] = dict(
        n_distinct_geometries=len(geoms),
        d_star=D_STAR, n_instances=n,
        frac_no_singleton_fullset_feasible=(no_feas / n if n else None),
        n_with_feasible_oracle=len(gaps),
        oracle_gap_positive_frac=(float((g > TOL).mean()) if len(g) else None),
        oracle_gap_mean=(float(g.mean()) if len(g) else None),
        oracle_gap_median=(float(np.median(g)) if len(g) else None),
        nu_pi_mean=float(np.mean(nus)) if nus else None,
        nu_pi_positive_frac=float(np.mean([v > TOL for v in nus])) if nus else None,
    )
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(out, indent=1))
    print("wrote", _OUT)
    print("selected", n, "instances; skipped:", skipped)
    print("primary:", json.dumps(out["primary_endpoint"], indent=1))


if __name__ == "__main__":
    main()
