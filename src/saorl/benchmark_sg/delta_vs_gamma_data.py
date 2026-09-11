"""V30: how much data does each certificate need?

The paper proves two finite-data statements about the same instance:

  Gamma route (Theorem 3): certify that EVERY eps-optimal policy under psi is
    full-set safe. Needs a certified eps-face -- one LP per competing reading
    over a confidence-inflated polytope.

  Delta route (Theorem 4): certify that SOME eps-optimal policy under psi is
    full-set safe. Needs only simultaneous value bounds:
        Vbar_psi - Vlow_U <= eps.

Both are run here with the SAME concentration machinery (per-pair Weissman L1
radius, centered simulation lemma, union over state-action pairs and visit
counts) so the comparison is apples to apples and any difference is due to
what is being certified, not to how the confidence set is built.

Registration: results/e2e/REGISTRATION_V28.md (V30 addendum).
Run: PYTHONPATH=. python3 -m saorl.benchmark_sg.delta_vs_gamma_data
Writes results/e2e/delta_vs_gamma_data.json
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

import numpy as np

from .control_mdp import GAMMA
from .control_suite import _eligible, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "paper", "theory_extension"))
from finite_data_certificate import (  # noqa: E402
    DiscountedCMDP, V_set, V_single, certificate, sample_trajectories)

SEL = os.path.join(ROOT, "results/e2e", "safe_face_select.json")
OUT = os.path.join(ROOT, "results/e2e", "delta_vs_gamma_data.json")

BUDGET = 0.005
EPS = 0.01
DELTA = 0.05
N_GRID = (10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8)
SEEDS = (0, 1, 2)
N_MIN = 5


def delta_certificate(m, Phat, counts, k, d, eps, delta, n_min=1, Nmax=None):
    """Certify Delta_psi^eps <= 0 from simultaneous value bounds.

    Uses the same per-pair Weissman radius and centered simulation lemma as
    `certificate`, then compares an UPPER bound on V_psi with a LOWER bound on
    V_U.  Relaxing the estimated budget by beta_c makes the estimated feasible
    set contain the true one (upper bound); tightening it by beta_c makes the
    estimated feasible set contained in the true one (lower bound).
    """
    import math
    S, A, g = m.S, m.A, m.gamma
    support = counts >= max(1, n_min)
    if not support.any():
        return {"pass": False, "gap_hi": None, "beta": None, "covered": 0.0}
    npairs = S * A
    delta_pair = delta / npairs if Nmax is None else delta / (npairs * Nmax)
    errs = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            if support[s, a]:
                n = int(counts[s, a])
                errs[s, a] = math.sqrt(
                    2.0 * (S * math.log(2.0) + math.log(1.0 / delta_pair)) / n)
    beta = (g / (2.0 * (1.0 - g))) * float(errs[support].max())
    r_max = float(m.r.max())
    c_max = float(max(c.max() for c in m.costs))
    beta_r, beta_c = r_max * beta, c_max * beta
    V_hi = V_single(m, k, d + beta_c, Phat, support)      # upper bound on V_psi
    V_lo = V_set(m, d - beta_c, Phat, support)            # lower bound on V_U
    if V_hi is None or V_lo is None:
        return {"pass": False, "gap_hi": None, "beta": beta,
                "covered": float(support.mean())}
    gap_hi = (V_hi + beta_r) - (V_lo - beta_r)
    return {"pass": bool(gap_hi <= eps + 1e-12), "gap_hi": float(gap_hi),
            "beta": beta, "covered": float(support.mean())}


def main() -> None:
    with open(SEL, encoding="utf8") as fh:
        sel = json.load(fh)
    mid = [r for r in sel["rows"]
           if r["regime"] == "optimizer_resolvable"
           and abs(r["budget"] - BUDGET) < 1e-12]

    pt, _ = parse_prometheus()
    bank = prom_threshold_bank(pt)
    compiled = {}
    for ti, t in enumerate(pt):
        ok, _w, readings = _eligible(build_prom_pool(t, bank))
        if ok:
            compiled[f'{getattr(t, "name", "?")}#{ti}'] = compile_instance(readings)

    rows: List[dict] = []
    for row in mid:
        cm = compiled[row["rule_id"]]
        m = DiscountedCMDP(cm["P"], cm["mu0"], cm["r"],
                           [cm["C"][k] for k in range(cm["C"].shape[0])], GAMMA)
        k = row["anchor"]
        behav = np.full((m.S, m.A), 1.0 / m.A)
        for seed in SEEDS:
            rng = np.random.default_rng(1000 + seed)
            first_delta: Optional[int] = None
            first_gamma: Optional[int] = None
            per_n = {}
            for n in N_GRID:
                Phat, counts = sample_trajectories(m, behav, n, rng)
                dc = delta_certificate(m, Phat, counts, k, BUDGET, EPS, DELTA,
                                       n_min=N_MIN, Nmax=n)
                gc = certificate(m, Phat, counts, k, BUDGET, EPS, DELTA,
                                 bound="weissman", n_min=N_MIN, Nmax=n)
                per_n[str(n)] = dict(delta=dc["pass"], gamma=gc["pass"],
                                     beta=dc["beta"], covered=dc["covered"])
                if dc["pass"] and first_delta is None:
                    first_delta = n
                if gc["pass"] and first_gamma is None:
                    first_gamma = n
            rows.append(dict(rule_id=row["rule_id"], seed=seed, anchor=k,
                             n_delta=first_delta, n_gamma=first_gamma,
                             per_n=per_n))
        print(f"  {row['rule_id']:38s} "
              f"delta@{rows[-1]['n_delta']} gamma@{rows[-1]['n_gamma']}")

    cert_d = [r for r in rows if r["n_delta"] is not None]
    cert_g = [r for r in rows if r["n_gamma"] is not None]
    both = [r for r in rows
            if r["n_delta"] is not None and r["n_gamma"] is not None]
    agg = dict(
        n_runs=len(rows),
        delta_certified=len(cert_d),
        gamma_certified=len(cert_g),
        delta_n_med=(float(np.median([r["n_delta"] for r in cert_d]))
                     if cert_d else None),
        gamma_n_med=(float(np.median([r["n_gamma"] for r in cert_g]))
                     if cert_g else None),
        both_certified=len(both),
        delta_strictly_cheaper=sum(1 for r in both
                                   if r["n_delta"] < r["n_gamma"]),
        delta_only=sum(1 for r in rows if r["n_delta"] is not None
                       and r["n_gamma"] is None),
        gamma_only=sum(1 for r in rows if r["n_gamma"] is not None
                       and r["n_delta"] is None))
    res = dict(
        registration="V30 (REGISTRATION_V28.md addendum): smallest logged-"
                     "transition count at which the value-gap (Delta) and the "
                     "face (Gamma) certificates fire, same confidence "
                     f"machinery, d={BUDGET}, eps={EPS}, delta={DELTA}, "
                     f"n_min={N_MIN}, uniform behaviour, seeds {list(SEEDS)}",
        budget=BUDGET, eps=EPS, delta=DELTA, n_grid=list(N_GRID),
        aggregate=agg, rows=rows)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {OUT}")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
