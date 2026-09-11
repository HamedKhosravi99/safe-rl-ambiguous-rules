"""V13: exact policy-sufficiency (optimal-face) study on the compiled suite.

For each eligible compiled monitoring instance and budget, and each
reading psi: value sufficiency asks whether SOME psi-optimal occupancy is
feasible for every retained reading (the published screen); POLICY
sufficiency asks whether EVERY psi-optimal occupancy is (the review's
F_psi = F_U, our Gate hierarchy's (B)). The face test is one LP per
competing reading:

    W_psi = max_phi  max { c_phi'x : flow(x), c_psi'x <= d,
                           r'x >= V_psi(d) - tol },

and psi is policy-sufficient iff W_psi <= d (+tol). Archived screen
verdicts are re-derived and asserted en route, as in surrogate_price.py.

Also runs the registered tie-break stress: 100 seeded random secondary
objectives optimized over the face; policy-sufficient readings must be
U-safe on all of them (asserted), merely-value-sufficient readings are
reported wherever they land.

Run: PYTHONPATH=. python3 -m saorl.benchmark_sg.policy_sufficiency
Writes results/e2e/policy_sufficiency.json
"""
from __future__ import annotations

import json
import os
from typing import List

import numpy as np
from scipy.optimize import linprog

from .control_suite import _eligible, _flow, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARCHIVE = os.path.join(ROOT, "results/e2e", "policy_class_budget.json")
OUT = os.path.join(ROOT, "results/e2e", "policy_sufficiency.json")

BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10)
OPERATING = 0.05
TOL = 1e-7
FACE_TOL = 1e-7
N_TIEBREAK = 100
SEED = 20260901


def _lp(m, obj, rows, rhs, extra_row=None, extra_rhs=None, maximize=True):
    A_eq, b_eq = _flow(m)
    A_ub = list(rows)
    b_ub = list(rhs)
    if extra_row is not None:
        A_ub.append(extra_row)
        b_ub.append(extra_rhs)
    res = linprog(-obj if maximize else obj,
                  A_ub=np.array(A_ub) if A_ub else None,
                  b_ub=np.array(b_ub) if b_ub else None,
                  A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if res.status != 0:
        return None, None
    return float(obj @ res.x), res.x


def main() -> None:
    with open(ARCHIVE, encoding="utf8") as fh:
        arch = json.load(fh)
    verd = {(r["rule_id"], round(r["budget"], 6)): r
            for r in arch["rows"] if r["policy_class"] == "monitoring_compiled"
            and r["status"] == "compiled"}

    pt, _ = parse_prometheus()
    bank = prom_threshold_bank(pt)
    rng = np.random.default_rng(SEED)
    rows_out: List[dict] = []
    for ti, t in enumerate(pt):
        pool = build_prom_pool(t, bank)
        ok, _why, readings = _eligible(pool)
        if not ok:
            continue
        uid = f'{getattr(t, "name", "?")}#{ti}'
        m = compile_instance(readings)
        r = m["r"].reshape(-1)
        K = m["C"].shape[0]
        crows = [m["C"][k].reshape(-1) for k in range(K)]
        for d in BUDGETS:
            av = verd.get((uid, round(d, 6)))
            if av is None:
                continue
            V_U, _ = _lp(m, r, crows, [d] * K)
            if V_U is None:
                continue
            V = []
            for k in range(K):
                v, _ = _lp(m, r, [crows[k]], [d])
                V.append(v)
            gap = min(v - V_U for v in V if v is not None)
            assert av["screen_fires"] == (gap > TOL), (uid, d)
            value_suff = [k for k in range(K)
                          if V[k] is not None and V[k] - V_U <= TOL]
            psi_rows = []
            for k in value_suff:
                W = -np.inf
                for kk in range(K):
                    if kk == k:
                        continue
                    w, _ = _lp(m, crows[kk], [crows[k], -r],
                               [d, -(V[k] - FACE_TOL)])
                    if w is None:
                        W = np.inf
                        break
                    W = max(W, w)
                if K == 1:
                    W = 0.0
                psi_rows.append(dict(reading=k, V=V[k],
                                     W=(None if np.isinf(W) else float(W)),
                                     policy_sufficient=bool(W <= d + 1e-6)))
            rows_out.append(dict(
                rule_id=uid, budget=d, K=K, V_U=V_U,
                screen_fires=bool(av["screen_fires"]),
                value_sufficient_exists=bool(value_suff),
                policy_sufficient_exists=any(pr["policy_sufficient"]
                                             for pr in psi_rows),
                readings=psi_rows))

            # registered tie-break stress at the operating budget, plus a
            # POST-HOC labeled repeat at d=0.005 where value and policy
            # sufficiency separate (registered stress covered OPERATING)
            if d in (OPERATING, 0.005) and value_suff:
                for k in value_suff:
                    safe = 0
                    for _s in range(N_TIEBREAK):
                        sigma = rng.standard_normal(r.shape[0])
                        _v, x = _lp(m, sigma, [crows[k], -r],
                                    [d, -(V[k] - FACE_TOL)])
                        if x is None:
                            continue
                        worst = max(float(c @ x) for c in crows)
                        safe += worst <= d + 1e-6
                    frac = safe / N_TIEBREAK
                    pr = next(p for p in rows_out[-1]["readings"]
                              if p["reading"] == k)
                    pr["tiebreak_safe_frac"] = frac
                    if pr["policy_sufficient"]:
                        assert frac == 1.0, (uid, k, frac,
                                             "theorem violated: policy-"
                                             "sufficient face with unsafe "
                                             "tie-break")

    op = [r for r in rows_out if r["budget"] == OPERATING]
    clear = [r for r in op if not r["screen_fires"]]
    pol = [r for r in clear if r["policy_sufficient_exists"]]
    overs = [pr["W"] - r["budget"] for r in op for pr in r["readings"]
             if pr["W"] is not None and not pr["policy_sufficient"]]
    ties = [pr.get("tiebreak_safe_frac") for r in op for pr in r["readings"]
            if pr.get("tiebreak_safe_frac") is not None
            and not pr["policy_sufficient"]]
    agg = dict(
        n_instances=len(op),
        n_value_clear=len(clear),
        n_policy_sufficient_exists=len(pol),
        n_value_clear_policy_fail=len(clear) - len(pol),
        overshoot_med=(float(np.median(overs)) if overs else None),
        overshoot_max=(max(overs) if overs else None),
        tiebreak_value_only_med=(float(np.median(ties)) if ties else None),
        tiebreak_value_only_min=(min(ties) if ties else None))
    res = dict(
        registration="V13 (REGISTRATION_V13.md): exact optimal-face policy "
                     "sufficiency vs the published value screen; archived "
                     "verdicts asserted; tie-break stress seeded "
                     f"{SEED}, N={N_TIEBREAK}",
        budgets=list(BUDGETS), operating=OPERATING,
        aggregate_at_operating=agg, rows=rows_out)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {OUT} ({len(rows_out)} rows)")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
