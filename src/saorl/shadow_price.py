"""T1/T2 deep-theory study (paper/corset_deep_theory_plan.md).

T2 (shadow-price identification of the price of ambiguity):
    PoA(U) <= sum_psi lambda*_psi (J_{c_psi}(pi^{psi*}) - b_psi)
with lambda* the optimal duals of the robust fixed-psi occupancy LP and
pi^{psi*} the price-realizing singleton oracle.  The tightness gap
(bound - PoA) equals g(lambda*) - L(mu*, lambda*), i.e. how far the oracle
occupancy is from maximizing the Lagrangian at lambda* -- computed exactly.

T1 (exact minimax value of the semantic game): a uniformly-valid algorithm
facing observationally indistinguishable worlds is a mixture over policies;
its exact minimax regret is the value of the subset LP
    V*(delta) = max { sum_T x_T V_T : x >= 0, sum x_T = 1,
                      sum_{T not containing w} x_T <= delta  for all w }
over constraint subsets T of the retained set, R*(delta) = max_w V_w - V*.
Compared against the paper's current Thm-4 lower bound
PoA - M*delta*Delta_R (which is the feasible point x_{[M]}=1-M*delta,
x_{[M]\\{w}}=delta).

Everything runs in the set-independent fixed_psi normalization (plan 9.2),
the geometry of the paper's certified prices.

Run: SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.shadow_price \
        [--domains synthetic,real,gridworld,budget] [--budget 0.05]
Writes results/conformal/lp/shadow_price.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import linprog

from .exact_lp import (_PRICE_REPORT, _ROOT, GREEDY_ACTION, IPM_THRESHOLD,
                       build_budget, build_gridworld, build_real,
                       build_synthetic, dp_optimal, fixed_psi_Z,
                       solve_occupancy_cg, solve_occupancy_lp)

_OUT = _ROOT / "results/conformal" / "lp" / "shadow_price.json"

DELTAS = [0.0, 0.01, 0.05, 0.10, 0.15]


def _solve(m, honored, d, names, Z):
    n_lp_vars = m.horizon * m.S * m.A
    solver = solve_occupancy_cg if n_lp_vars > IPM_THRESHOLD else solve_occupancy_lp
    return solver(m, honored, d, report_readings=names,
                  normalization="fixed_psi", Z=Z)


def mixture_lp(V_by_subset: Dict[frozenset, float], names: List[str],
               delta: float) -> dict:
    """Exact T1 subset LP.  Variables: one weight per subset T (incl. empty).
    max sum x_T V_T  s.t.  sum_{T not ni w} x_T <= delta (each w), sum x = 1."""
    subsets = list(V_by_subset.keys())
    c = np.array([-V_by_subset[T] for T in subsets])
    A_ub = np.array([[1.0 if w not in T else 0.0 for T in subsets]
                     for w in names])
    b_ub = np.full(len(names), delta)
    A_eq = np.ones((1, len(subsets)))
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"mixture LP failed: {res.message}")
    x = np.maximum(res.x, 0.0)
    support = {"+".join(sorted(T)) or "(none)": float(x[i])
               for i, T in enumerate(subsets) if x[i] > 1e-12}
    return dict(V_star=float(-res.fun), support=support)


def study(domain: str, d: float) -> dict:
    from .experiments import _domain_specs
    specs = _domain_specs([domain], ["fqi"])
    _data, env, U, _ret, _pool = specs[domain].build(0)
    price = json.load(open(_PRICE_REPORT))
    assert [c.name for c in U] == price["domains"][domain]["rows"][0]["u_names"]
    builder = dict(synthetic=build_synthetic, real=build_real,
                   gridworld=build_gridworld, budget=build_budget)[domain]
    m = builder(env, U)
    names = [c.name for c in U]
    M = len(names)
    Z = fixed_psi_Z(m, names, m.action_names.index(GREEDY_ACTION[domain]))
    b = {k: d * float(Z[k]) for k in names}

    print(f"[{domain}] S={m.S} A={m.A} H={m.horizon} M={M}; "
          f"solving {2 ** M - 1} constrained programs + unconstrained ...")

    # ---- all subset values (T1) + singleton solutions (T2) -----------------
    V: Dict[frozenset, float] = {frozenset(): dp_optimal(m)}
    singles: Dict[str, object] = {}
    for r in range(1, M + 1):
        for T in itertools.combinations(names, r):
            sol = _solve(m, list(T), d, names, Z)
            V[frozenset(T)] = sol.value
            if r == 1:
                singles[T[0]] = sol
            if r == M:
                robust = sol

    # ---- T2: shadow-price bound at the price-realizing oracle --------------
    psi_star = max(names, key=lambda k: V[frozenset([k])])
    lam = robust.diagnostics["cost_duals"]
    oracle = singles[psi_star]
    E = {k: oracle.on_policy[k]["E_episodic_cost"] for k in names}
    excess = {k: E[k] - b[k] for k in names}
    bound_signed = sum(lam[k] * excess[k] for k in names)
    bound_plus = sum(lam[k] * max(excess[k], 0.0) for k in names)
    poa = V[frozenset([psi_star])] - V[frozenset(names)]
    lp_gap = robust.diagnostics["relative_duality_gap"]
    t2 = dict(
        psi_star=psi_star,
        V_psi_star=V[frozenset([psi_star])],
        V_robust=V[frozenset(names)],
        poa_exact=poa,
        duals=lam,
        budgets=b,
        oracle_episodic_costs=E,
        oracle_excess=excess,
        bound_signed=bound_signed,
        bound_plus=bound_plus,
        tightness_gap=bound_signed - poa,   # = g(lam*) - L(mu*, lam*) >= -tol
        robust_lp_relative_duality_gap=lp_gap,
        holds=bool(bound_signed >= poa - 1e-6 * (1 + abs(poa))),
    )

    # ---- T1: exact minimax value vs the paper's Thm-4 bound ----------------
    delta_R = float((m.r.max() - m.r.min()) * m.horizon)   # gamma = 1
    max_Vw = max(V[frozenset([k])] for k in names)
    rows = []
    for delta in DELTAS:
        mix = mixture_lp(V, names, delta)
        r_star = max_Vw - mix["V_star"]
        old_lb = max(0.0, poa_full(V, names) - M * delta * delta_R)
        rows.append(dict(delta=delta, V_star=mix["V_star"],
                         R_star=r_star, old_thm4_lower_bound=old_lb,
                         slack_of_old_bound=r_star - old_lb,
                         optimal_mixture=mix["support"]))
    t1 = dict(delta_R=delta_R, M=M,
              V_by_subset={"+".join(sorted(T)) or "(none)": v
                           for T, v in V.items()},
              minimax=rows)
    return dict(U=names, budget=d, normalization="fixed_psi",
                solver=robust.diagnostics["method"], t2_shadow_price=t2,
                t1_exact_minimax=t1)


def poa_full(V: Dict[frozenset, float], names: List[str]) -> float:
    return max(V[frozenset([k])] for k in names) - V[frozenset(names)]


def main() -> None:
    assert os.environ.get("SAORL_CONFORMAL") == "1", "run with SAORL_CONFORMAL=1"
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="synthetic,real,gridworld,budget")
    ap.add_argument("--budget", type=float, default=0.05)
    args = ap.parse_args()

    report: dict = dict(config=dict(
        budget=args.budget, deltas=DELTAS, normalization="fixed_psi",
        note="T2 shadow-price identification + T1 exact minimax game value; "
             "see paper/corset_deep_theory_plan.md"), domains={})
    for dom in args.domains.split(","):
        t0 = time.time()
        out = study(dom, args.budget)
        out["elapsed_s"] = time.time() - t0
        report["domains"][dom] = out
        t2, t1 = out["t2_shadow_price"], out["t1_exact_minimax"]
        print(f"[{dom}] PoA={t2['poa_exact']:.3f}  "
              f"dual-bound={t2['bound_signed']:.3f} "
              f"(plus={t2['bound_plus']:.3f})  "
              f"gap={t2['tightness_gap']:.4f}  holds={t2['holds']}")
        for r in t1["minimax"]:
            print(f"        delta={r['delta']:>5}: R*={r['R_star']:.3f}  "
                  f"old-LB={r['old_thm4_lower_bound']:.3f}  "
                  f"slack={r['slack_of_old_bound']:.3f}  "
                  f"mix={r['optimal_mixture']}")
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(_OUT, "w"), indent=1)
    print(f"wrote {_OUT}")


if __name__ == "__main__":
    main()
