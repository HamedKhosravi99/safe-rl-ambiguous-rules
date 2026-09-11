"""Alternative conservative representations of ambiguity — exact LP study.

Reviewer question: the single-translation kill test is structurally
favorable to CORSET (a policy optimized for one reading is evaluated
against all retained readings).  The fair fight is among *reasonably
conservative* representations.  This study compares, at their exact
tabular optima (no learner confounds; fixed-psi normalization, plan 9.2):

  corset          robust program over the conformal set U-hat (reference)
  best_single     the best retained singleton, chosen with ORACLE knowledge
                  of the deployment metric (upper-envelopes every
                  "most conservative member" rule)
  pruned_pool     dominance-pruned RAW POOL, robust over the maximal
                  elements -- conservatism without calibration
  union_pool      one cost = pointwise max over the RAW POOL (union/max
                  aggregation, maximal conservatism without calibration)
  posterior       Bayesian soft defense: one cost = plausibility-posterior
                  mixture over the raw pool, E_p[cost] budgeted
  top_p           credal/HPD-style construction: smallest plausibility-mass
                  >= 1-delta_sem prefix of the raw pool, robust over it

plus two return-matched scalars per domain (the "matched by return, not by
training procedure" comparison, exact by bisection over the LP frontier):

  t_at_corset     min uniform budget multiplier t such that some policy has
                  J_r >= V_corset and worst retained cost <= t*d  (t~1 <=>
                  CORSET sits exactly on the feasibility frontier)
  t_at_single     same at the most-plausible singleton's return level
                  (t >> 1 <=> ANY policy with that return must violate --
                  the kill-test failure is forced by the return level, not
                  by the baseline's training procedure)

Note kept as a paper remark, not a table row: the "matched-size" Bayesian
HPD set (top-|U| by score) IS the conformal set -- conformal sets are
score-monotone, so at equal size the two coincide; calibration's entire
content is the threshold/size choice.

Run: SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.alt_defense \
        [--domains synthetic,real,gridworld,budget] [--budget 0.05]
Writes results/conformal/lp/alt_defense.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .exact_lp import (_PRICE_REPORT, _ROOT, GREEDY_ACTION, IPM_THRESHOLD,
                       build_budget, build_gridworld, build_real,
                       build_synthetic, fixed_psi_Z, solve_occupancy_cg,
                       solve_occupancy_lp)

_OUT = _ROOT / "results/conformal" / "lp" / "alt_defense.json"
DSEM = 0.10          # matches the deployed conformal level


def _solve(m, honored, d, names, Z):
    n_lp_vars = m.horizon * m.S * m.A
    solver = solve_occupancy_cg if n_lp_vars > IPM_THRESHOLD else solve_occupancy_lp
    return solver(m, honored, d, report_readings=names,
                  normalization="fixed_psi", Z=Z)


def _worst_retained(sol, retained: Sequence[str], Z: Dict[str, float]) -> float:
    """max over retained readings of E[episodic cost]/Z_psi (the fixed-psi
    active-normalized analogue; compare against the budget d)."""
    vals = []
    for k in retained:
        Ec = sol.on_policy[k]["E_episodic_cost"]
        vals.append(Ec / Z[k] if Z[k] > 0 else (0.0 if Ec <= 1e-12 else float("inf")))
    return float(max(vals))


def _add_reading(m, name: str, cost: np.ndarray, fire: np.ndarray):
    """Return a shallow copy of the model with one synthetic reading added."""
    m2 = copy.copy(m)
    m2.cost = dict(m.cost)
    m2.fire = dict(m.fire)
    m2.cost[name] = cost
    m2.fire[name] = fire
    return m2


def _bisect_t(m, retained, names, Z, d, v_target, lo=0.0, hi=None,
              iters=11, slack=1e-6) -> Optional[float]:
    """Smallest uniform budget multiplier t with V(t*d) >= v_target."""
    if hi is None:
        hi = 1.0
        for _ in range(12):                      # find a feasible upper end
            v = _solve(m, list(retained), hi * d, names, Z).value
            if v >= v_target - slack * (1 + abs(v_target)):
                break
            hi *= 2.0
        else:
            return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v = _solve(m, list(retained), mid * d, names, Z).value
        if v >= v_target - slack * (1 + abs(v_target)):
            hi = mid
        else:
            lo = mid
    return float(hi)


def _budget_pool():
    """Reconstruct the budget domain's FULL cap pool ($20..$80) with its
    cached plausibilities -- exactly the raw set the deployed construction
    filtered (experiments.py _budget_spec)."""
    from .budget import RULE_SETS_B
    from .judge import attach_plausibility, load_llm_cache
    cache = str(Path(__file__).with_name("plausibility_cache_budget.json"))
    raw = [c for c, _ in RULE_SETS_B["budget_ambiguous"]["items"]]
    judge = load_llm_cache("budget_ambiguous", cache)
    return attach_plausibility(raw, judge)


def study(domain: str, d: float) -> dict:
    from .experiments import _domain_specs
    specs = _domain_specs([domain], ["fqi"])
    _data, env, U, _ret, raw_pool = specs[domain].build(0)
    price = json.load(open(_PRICE_REPORT))
    assert [c.name for c in U] == price["domains"][domain]["rows"][0]["u_names"]

    if raw_pool is None and domain == "budget":
        raw_pool = _budget_pool()               # rejected caps ARE the point here
    pool_cands = list(raw_pool) if raw_pool is not None else list(U)

    builder = dict(synthetic=build_synthetic, real=build_real,
                   gridworld=build_gridworld, budget=build_budget)[domain]
    m = builder(env, pool_cands)                # model carries the whole pool
    retained = [c.name for c in U]
    pool = [c.name for c in pool_cands]
    plaus = {c.name: c.plaus_mean() for c in pool_cands}
    a_greedy = m.action_names.index(GREEDY_ACTION[domain])
    Z = fixed_psi_Z(m, pool, a_greedy)

    print(f"[{domain}] S={m.S} A={m.A} H={m.horizon} pool={len(pool)} "
          f"retained={len(retained)}", flush=True)

    rows: Dict[str, dict] = {}

    def record(key, sol, honored, note=""):
        rows[key] = dict(
            honored=list(honored), V=sol.value,
            worst_retained=_worst_retained(sol, retained, Z), note=note)
        print(f"  {key:12s} V={sol.value:9.3f}  "
              f"worst-retained={rows[key]['worst_retained']:.4f} "
              f"(d={d})  honored={honored}", flush=True)

    # -- reference -----------------------------------------------------------
    record("corset", _solve(m, retained, d, pool, Z), retained)

    # -- best retained singleton (oracle member choice) ----------------------
    best = None
    for k in retained:
        sol = _solve(m, [k], d, pool, Z)
        w = _worst_retained(sol, retained, Z)
        if best is None or w < best[1]:
            best = (k, w, sol)
    record("best_single", best[2], [best[0]],
           note="oracle choice: min worst-retained over retained singletons")

    # -- dominance-pruned raw pool (no calibration) ---------------------------
    maximal = [i for i in pool
               if not any(j != i and np.all(m.cost[i] <= m.cost[j])
                          and np.any(m.cost[j] != m.cost[i]) for j in pool)]
    record("pruned_pool", _solve(m, maximal, d, pool, Z), maximal)

    # -- union / max-cost over the raw pool -----------------------------------
    c_union = np.maximum.reduce([m.cost[k] for k in pool])
    f_union = np.maximum.reduce([m.fire[k] for k in pool])
    mu = _add_reading(m, "UNION", c_union, f_union)
    Zu = dict(Z); Zu.update(fixed_psi_Z(mu, ["UNION"], a_greedy))
    record("union_pool", _solve(mu, ["UNION"], d, pool, Zu), ["UNION"])

    # -- Bayesian posterior soft defense --------------------------------------
    tot = sum(plaus.values()) or 1.0
    p = {k: plaus[k] / tot for k in pool}
    c_post = sum(p[k] * m.cost[k] for k in pool)
    mp = _add_reading(m, "POST", c_post, f_union)
    Zp = dict(Z); Zp.update(fixed_psi_Z(mp, ["POST"], a_greedy))
    record("posterior", _solve(mp, ["POST"], d, pool, Zp), ["POST"],
           note=f"p propto plausibility: { {k: round(p[k],3) for k in pool} }")

    # -- top-p credal construction (mass >= 1-dsem) ----------------------------
    order = sorted(pool, key=lambda k: -plaus[k])
    acc, chosen = 0.0, []
    for k in order:
        chosen.append(k); acc += p[k]
        if acc >= 1 - DSEM:
            break
    record("top_p", _solve(m, chosen, d, pool, Z), chosen,
           note=f"smallest plausibility-mass >= {1-DSEM:g} prefix")

    # -- return-matched multipliers -------------------------------------------
    v_corset = rows["corset"]["V"]
    most_pl = max(retained, key=lambda k: plaus[k])
    v_single = _solve(m, [most_pl], d, pool, Z).value
    t_c = _bisect_t(m, retained, pool, Z, d, v_corset, hi=1.0)
    t_s = _bisect_t(m, retained, pool, Z, d, v_single)
    print(f"  t_at_corset={t_c}  t_at_single={t_s} (most plausible={most_pl})",
          flush=True)

    return dict(U=retained, pool=pool, plausibility=plaus, budget=d,
                normalization="fixed_psi", Z={k: Z[k] for k in pool},
                representations=rows,
                return_matched=dict(
                    v_corset=v_corset, t_at_corset=t_c,
                    most_plausible=most_pl, v_single=v_single,
                    t_at_single=t_s))


def main() -> None:
    assert os.environ.get("SAORL_CONFORMAL") == "1", "run with SAORL_CONFORMAL=1"
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="synthetic,real,gridworld,budget")
    ap.add_argument("--budget", type=float, default=0.05)
    args = ap.parse_args()
    report: dict = dict(config=dict(
        budget=args.budget, dsem=DSEM, normalization="fixed_psi",
        note="alternative conservative representations at exact LP optima; "
             "see reviewer item 4 fix"), domains={})
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    for dom in args.domains.split(","):
        t0 = time.time()
        out = study(dom, args.budget)
        out["elapsed_s"] = time.time() - t0
        report["domains"][dom] = out
        # write after EVERY domain: a killed/preempted run keeps its finished
        # legs, and progress is observable from the file's mtime
        json.dump(report, open(_OUT, "w"), indent=1)
        print(f"  [{dom}] done in {out['elapsed_s']:.0f}s; "
              f"wrote partial {_OUT.name}", flush=True)
    print(f"wrote {_OUT}", flush=True)


if __name__ == "__main__":
    main()
