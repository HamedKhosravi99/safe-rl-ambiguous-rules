"""Class ladder: measuring the middle between the two published classes.

POST-HOC ENDPOINT (2026-09-01), not part of the frozen manifest.  The paper
reports the screen under two decision classes: the compiled monitoring
controller (fires 1/60) and the free per-fixture class (55/60 on the shared
subset; 2/44 admission).  Its own text concedes "the classes between the
extremes are unmeasured."  This module measures them.

THE LADDER.  A controller is indexed by the set A of reading-predicates it
can evaluate at decision time.  Every fixture f then looks to it like its
fingerprint  g_A(f) = (1[psi fires on f])_{psi in A},  and any policy the
controller can implement must accept or reject fingerprint groups
wholesale.  The class value under constraint set B is the exact optimum of

    V_{B,A}(d) = max  sum_g n_g y_g
                 s.t. sum_g (m_{psi,g} - d n_g) y_g <= 0   for all psi in B,
                      y_g in {0,1},

with n_g the group size and m_{psi,g} the number of psi-firing fixtures in
group g -- the free-class program of policy_class_budget.py aggregated to
the partition g_A.  A = emptyset is the blind controller (act on all
fixtures or none); |A| grows toward the free class, which is the identity
partition.  Refining the partition can only enlarge the feasible set, so
V is monotone in A for every B; the SCREEN
    fires(A, d)  iff  min_psi V_{psi,A}(d) - V_{U,A}(d) > 0
is not monotone, and where it turns on is the object of study: the
capability frontier of the rule -- the observation power at which the
unresolved wording starts to change the best safe decision.

ANCHORS.  The free rung is recomputed with free_values_multi and asserted
equal, row by row, to the archived results/e2e/policy_class_budget.json, so
the ladder's top end IS the published number.  The compiled controller is
carried alongside from the same archive (no re-solve).

Run:  PYTHONPATH=. python3 -m saorl.benchmark_sg.class_ladder
Writes results/e2e/class_ladder.json
"""
from __future__ import annotations

import itertools
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .evaluate import build_prom_pool, build_kyv_pool
from .parse import parse_prometheus, parse_kyverno, prom_threshold_bank
from .policy_class_budget import free_values_multi

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARCHIVE = os.path.join(ROOT, "results/e2e", "policy_class_budget.json")
OUT = os.path.join(ROOT, "results/e2e", "class_ladder.json")

BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10)
OPERATING = 0.05
# k-rungs evaluated at every budget; the exhaustive subset sweep at each k
# runs at the operating budget (cost control, disclosed in the JSON).
GRID_KS = (0, 1)
SUBSET_CAP = 60          # per (pool, k); deterministic lexicographic order


def _grouped(vecs: np.ndarray, A: Sequence[int]):
    """Partition fixtures by fingerprint over A -> (n_g, m[psi, g])."""
    K, F = vecs.shape
    if len(A) == 0:
        keys = np.zeros(F, dtype=int)
    else:
        sub = vecs[list(A)] > 0.5
        keys = np.zeros(F, dtype=int)
        for row in sub:                       # binary encode the fingerprint
            keys = keys * 2 + row.astype(int)
    uniq, inv = np.unique(keys, return_inverse=True)
    G = len(uniq)
    n_g = np.bincount(inv, minlength=G).astype(float)
    m = np.zeros((K, G))
    for k in range(K):
        m[k] = np.bincount(inv, weights=vecs[k], minlength=G)
    return n_g, m


def _value(n_g: np.ndarray, m: np.ndarray, rows: Sequence[int], d: float) -> int:
    """Exact V over {0,1}^G for constraint rows `rows` at budget d."""
    G = len(n_g)
    W = m[list(rows)] - d * n_g[None, :]
    res = milp(c=-n_g,
               constraints=LinearConstraint(W, -np.inf, 0.0),
               integrality=np.ones(G), bounds=Bounds(0, 1))
    if not res.success:
        return 0
    return int(round(-res.fun))


def _screen(n_g, m, K, d) -> Tuple[int, int, bool]:
    """(V_U, min_psi V_psi, fires) in the class the partition defines."""
    V_U = _value(n_g, m, list(range(K)), d)
    V_min = min(_value(n_g, m, [k], d) for k in range(K))
    return V_U, V_min, bool(V_min - V_U > 0)


def main() -> None:
    with open(ARCHIVE, encoding="utf8") as fh:
        arch = json.load(fh)
    free_rows = {(r["rule_id"], r["family"], round(r["budget"], 6)): r
                 for r in arch["rows"] if r["policy_class"] == "free"}
    compiled_rows = {(r["rule_id"], round(r["budget"], 6)): r
                     for r in arch["rows"]
                     if r["policy_class"] == "monitoring_compiled"}

    pt, _ = parse_prometheus()
    kt, _ = parse_kyverno()
    bank = prom_threshold_bank(pt)

    out_rows: List[dict] = []
    n_anchor = 0
    for fam, targets, builder in (
            ("prometheus", pt, lambda t: build_prom_pool(t, bank)),
            ("kyverno", kt, build_kyv_pool)):
        for ti, t in enumerate(targets):
            pool = builder(t)
            uid = f'{getattr(t, "name", getattr(t, "policy_name", "?"))}#{ti}'
            vecs = np.asarray([np.asarray(c.vector, float) for c in pool.classes])
            K, F = vecs.shape

            # ---- anchor: the free rung must equal the archive, every budget
            vals = free_values_multi([tuple(v) for v in vecs], BUDGETS)
            for d in BUDGETS:
                ar = free_rows.get((uid, fam, round(d, 6)))
                V_full, V_k = vals[d]
                assert ar is not None, f"no archived free row for {uid} d={d}"
                gap = min(v - V_full for v in V_k)
                assert ar["V_U"] == V_full and ar["screen_fires"] == (gap > 0), (
                    f"free anchor mismatch {uid} d={d}: archive "
                    f"({ar['V_U']},{ar['screen_fires']}) vs ({V_full},{gap > 0})")
                n_anchor += 1

            # ---- the ladder
            ks = sorted(set(list(GRID_KS) + list(range(min(K, 3) + 1)) + [K]))
            ks = [k for k in ks if k <= K]
            for k in ks:
                subsets = list(itertools.combinations(range(K), k))
                truncated = len(subsets) > SUBSET_CAP
                subsets = subsets[:SUBSET_CAP]
                buds = BUDGETS if (k in GRID_KS or k == K) else (OPERATING,)
                for A in subsets:
                    n_g, m = _grouped(vecs, A)
                    for d in buds:
                        V_U, V_min, fires = _screen(n_g, m, K, d)
                        out_rows.append(dict(
                            rule_id=uid, family=fam, k=k, A=list(A),
                            n_groups=int(len(n_g)), budget=d,
                            V_U=V_U, V_min_single=V_min,
                            screen_fires=fires, truncated=truncated))
            # free rung, restated as a ladder row (identity partition value
            # taken from the anchored recomputation, not re-solved per fixture)
            for d in BUDGETS:
                V_full, V_k = vals[d]
                out_rows.append(dict(
                    rule_id=uid, family=fam, k="free", A=None,
                    n_groups=F, budget=d, V_U=V_full,
                    V_min_single=min(V_k),
                    screen_fires=bool(min(V_k) - V_full > 0), truncated=False))
        print(f"[{fam}] pools done")

    # ---- monotonicity audit: refining the partition never shrinks V_U
    by_pool: Dict[Tuple[str, float], List[dict]] = {}
    for r in out_rows:
        if r["k"] == "free":
            continue
        by_pool.setdefault((r["rule_id"], r["budget"]), []).append(r)
    for (_uid, _d), rs in by_pool.items():
        for a, b in itertools.combinations(rs, 2):
            if set(a["A"]) <= set(b["A"]):
                assert a["V_U"] <= b["V_U"], (a, b)

    # ---- aggregates at the operating budget
    def agg(fam: str) -> dict:
        pools = sorted({r["rule_id"] for r in out_rows if r["family"] == fam})
        per_k: Dict[str, dict] = {}
        kvals = sorted({r["k"] for r in out_rows
                        if r["family"] == fam and isinstance(r["k"], int)})
        for k in list(kvals) + ["free"]:
            some = every = n_pools = 0
            for uid in pools:
                rs = [r for r in out_rows
                      if r["rule_id"] == uid and r["family"] == fam
                      and r["k"] == k and r["budget"] == OPERATING]
                if not rs:
                    continue
                n_pools += 1
                fires = [r["screen_fires"] for r in rs]
                some += any(fires)
                every += all(fires)
            per_k[str(k)] = dict(n_pools=n_pools, fires_some=some,
                                 fires_all=every)
        # capability frontier: least k at which SOME size-k observation set
        # makes set protection necessary (None = not even free)
        frontier: Dict[str, object] = {}
        for uid in pools:
            f = None
            for k in kvals:
                rs = [r for r in out_rows
                      if r["rule_id"] == uid and r["family"] == fam
                      and r["k"] == k and r["budget"] == OPERATING]
                if rs and any(r["screen_fires"] for r in rs):
                    f = k
                    break
            if f is None:
                fr = [r for r in out_rows
                      if r["rule_id"] == uid and r["family"] == fam
                      and r["k"] == "free" and r["budget"] == OPERATING]
                f = "free" if (fr and fr[0]["screen_fires"]) else None
            frontier[uid] = f
        return dict(per_k=per_k, frontier=frontier)

    res = dict(
        registration=("post-hoc class-ladder endpoint, 2026-09-01; free rung "
                      "anchored row-by-row to policy_class_budget.json "
                      f"({n_anchor} rows asserted equal); compiled controller "
                      "carried from the same archive, not re-solved"),
        budgets=list(BUDGETS), operating=OPERATING,
        subset_cap=SUBSET_CAP, grid_ks=list(GRID_KS),
        aggregate={fam: agg(fam) for fam in ("prometheus", "kyverno")},
        compiled_anchor={
            uid_d[0]: dict(fires=r["screen_fires"], status=r["status"])
            for uid_d, r in compiled_rows.items()
            if round(uid_d[1], 6) == round(OPERATING, 6)},
        rows=out_rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {OUT}  ({len(out_rows)} rows, {n_anchor} anchor checks)")
    for fam in ("prometheus", "kyverno"):
        print(fam, json.dumps(res["aggregate"][fam]["per_k"]))


if __name__ == "__main__":
    main()
