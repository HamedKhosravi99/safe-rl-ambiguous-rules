"""D2: pre-registered expected-cost deployment certificate.

Executes the D2 section of paper/Plans/EVIDENCE_FREEZE_MANIFEST.md
(committed 91074b3 before any outcome): for each shipped policy (seed),

    E[W] <= B_dom * p_bar,

where W = per-episode worst fixed-psi-normalized retained cost,
p_bar = exact one-sided Clopper-Pearson upper bound at level 1-delta_ev/2
(delta_ev = 0.05) on the episode-violation rate from the ARCHIVED per-seed
counts (k of n=300; results/conformal/{risk50,budget50}), and

    B_dom = max_k  (max-cost DP value of reading k) / Z_k,

with the DP maximizing over actions and over successors of positive
transition probability (almost-sure bound), horizon = the domain's H, and
Z_k the fixed per-reading normalizer of exact_lp.fixed_psi_Z.

Justification (registered): Z_k(tau) = 1{sum_t c_k > 0} in
saorl/offline.py, so {violation} = {W > 0} exactly and
E[W] = E[W 1{W>0}] <= B Pr(viol).

Run: SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.expected_cost_certificate
Writes results/conformal/lp/expected_cost_certificate.json.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

import numpy as np
from scipy.stats import beta

from .alt_defense import GREEDY_ACTION
from .exact_lp import (_ROOT, build_budget, build_gridworld, build_real,
                       build_synthetic, fixed_psi_Z)

_OUT = _ROOT / "results/conformal" / "lp" / "expected_cost_certificate.json"
DELTA_EV = 0.05
LEVEL = 1.0 - DELTA_EV / 2.0      # registered split: delta_ev/2 to this bound
D_BUDGET = 0.05


def cp_upper(k: int, n: int, level: float = LEVEL) -> float:
    if k >= n:
        return 1.0
    return float(beta.ppf(level, k + 1, n - k))


def max_cost_dp(m, reading: str) -> float:
    """Almost-sure per-episode max of sum_t c_reading over any policy and
    any positive-probability transition realization (finite horizon)."""
    c = m.cost[reading]                       # S x A per-step cost
    S, A, H = m.S, m.A, m.horizon
    succ: List[List[np.ndarray]] = []
    for a in range(A):
        Ta = m.T[a].tocsr()
        succ.append([Ta.indices[Ta.indptr[s]:Ta.indptr[s + 1]]
                     for s in range(S)])
    V = np.zeros(S)
    for _t in range(H):
        V_next = V
        V = np.full(S, -np.inf)
        for s in range(S):
            best = -np.inf
            for a in range(A):
                ns = succ[a][s]
                cont = float(V_next[ns].max()) if ns.size else 0.0
                best = max(best, float(c[s, a]) + cont)
            V[s] = best
        # states with no successors under any action: treat as terminal 0
        V[np.isneginf(V)] = 0.0
    start = np.nonzero(m.p0 > 0)[0]
    return float(V[start].max()) if start.size else float(V.max())


def _newest(dirname: str):
    fs = glob.glob(os.path.join(_ROOT, "results/conformal", dirname,
                                "experiments_*.json"))
    return json.load(open(max(fs, key=os.path.getmtime))) if fs else None


def study(domain: str) -> dict:
    from .experiments import _domain_specs
    specs = _domain_specs([domain], ["fqi"])
    _data, env, U, _ret, _pool = specs[domain].build(0)
    builder = dict(synthetic=build_synthetic, real=build_real,
                   gridworld=build_gridworld, budget=build_budget)[domain]
    m = builder(env, list(U))
    names = [c.name for c in U]
    a_greedy = m.action_names.index(GREEDY_ACTION[domain])
    Z = fixed_psi_Z(m, names, a_greedy)
    per_reading = {}
    B = 0.0
    for k in names:
        raw = max_cost_dp(m, k)
        z = float(Z[k]) if float(Z[k]) > 0 else 1.0
        per_reading[k] = dict(raw_max=raw, Z=float(Z[k]),
                              B_k=raw / z)
        B = max(B, raw / z)

    blob = _newest("budget50" if domain == "budget" else "risk50")
    n_ep = int(blob["config"].get("n_eval_episodes",
               blob["config"].get("n_episodes", 300)))
    out = dict(B=B, per_reading=per_reading, n_episodes=n_ep,
               level=LEVEL, budget=D_BUDGET, arms={})
    for model, label in (("fqi:saorl", "corset"), ("fqi:single", "single")):
        recs = [r for r in blob["records"]
                if r["domain"] == domain and r["model"] == model]
        if not recs:
            continue
        bounds = []
        for r in recs:
            k = int(round(float(r["chance"]) * n_ep))
            bounds.append(B * cp_upper(k, n_ep))
        bnd = np.array(bounds)
        out["arms"][label] = dict(
            n_seeds=len(recs),
            median_expected_cost_bound=float(np.median(bnd)),
            best=float(bnd.min()), worst=float(bnd.max()),
            n_certified_at_budget=int((bnd <= D_BUDGET).sum()),
            median_certifies=bool(np.median(bnd) <= D_BUDGET))
    return out


def main() -> None:
    out = dict(config=dict(
        manifest="paper/Plans/EVIDENCE_FREEZE_MANIFEST.md@91074b3",
        delta_ev=DELTA_EV, level=LEVEL, budget=D_BUDGET,
        route="E[W] <= B * CP_{1-delta_ev/2}(k, n); B by max-cost DP / fixed_psi_Z"),
        domains={})
    for d in ("synthetic", "real", "gridworld", "budget"):
        out["domains"][d] = study(d)
        s = out["domains"][d]
        c = s["arms"].get("corset", {})
        print(f"{d:>10s}: B={s['B']:.3f}  corset median bound="
              f"{c.get('median_expected_cost_bound', float('nan')):.4f} "
              f"(certified at d={D_BUDGET}: "
              f"{c.get('n_certified_at_budget','-')}/{c.get('n_seeds','-')})")
    n_median = sum(1 for d in out["domains"].values()
                   if d["arms"].get("corset", {}).get("median_certifies"))
    out["decision_D2"] = dict(
        median_certifies_in_domains=n_median,
        rule="thm:deploy stays in main iff median CORSET seed certifies in >= 2 of 4",
        keep_in_main=bool(n_median >= 2))
    _OUT.write_text(json.dumps(out, indent=1))
    print("wrote", _OUT)
    print("D2 decision:", out["decision_D2"])


if __name__ == "__main__":
    main()
