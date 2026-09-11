"""The exact operational price of always-robustify under a single-cost learner.

POST-HOC ENDPOINT (2026-09-01).  Reviewer question: does ARROW measurably
beat BOTH baselines end to end, not just where the screen fires?  In the
exact setting the return of always-robustify equals V_U by definition, so
no value gap can exist THERE.  But the paper's own Section 3 records where
the gap lives: a learner restricted to ONE cost signal cannot optimize the
robust program directly and must use the pointwise-max surrogate
max_psi c_psi, whose feasible set is contained in the robust one
(Appendix proof-surrogate).  That containment has an exact price:

    V_surr(d) = max { r'x : x in X, (max_psi c_psi)'x <= d }   <=   V_U(d),

and the gap V_U - V_surr is what a deployment that never decides pays,
per instance, in the same LP units as everything else.  ARROW at a
cleared screen hands the learner ONE true cost and attains V_U exactly;
so wherever the screen clears and V_surr < V_U, ARROW strictly beats
single-cost always-robustify on return at zero safety cost -- an
end-to-end exact statement, no learner in the loop.  (Multi-cost learners
optimize (2) directly and tie; the comparison is about the single-signal
operating mode most published safe-RL learners actually expose, which is
the paper's stated motivation for the surrogate.)

Caveat stated where it belongs: max_psi c_psi here is the pointwise max
of the compiled cost matrices, the object Section 3 names.

Population: every eligible compiled monitoring instance (the same 28 the
screen study solves), full budget grid.  Verdicts are read from the
policy_class_budget archive and asserted to match re-derived gaps.

Run: PYTHONPATH=. python3 -m saorl.benchmark_sg.surrogate_price
Writes results/e2e/surrogate_price.json
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
from scipy.optimize import linprog

from .control_suite import _eligible, _flow, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARCHIVE = os.path.join(ROOT, "results/e2e", "policy_class_budget.json")
OUT = os.path.join(ROOT, "results/e2e", "surrogate_price.json")

BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10)
OPERATING = 0.05
TOL = 1e-7


def _solve(m: dict, rows: List[np.ndarray], d: float):
    A_eq, b_eq = _flow(m)
    r = m["r"].reshape(-1)
    A_ub = np.array(rows) if rows else None
    b_ub = np.array([d] * len(rows)) if rows else None
    res = linprog(-r, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, None), method="highs")
    return None if res.status != 0 else float(r @ res.x)


def main() -> None:
    with open(ARCHIVE, encoding="utf8") as fh:
        arch = json.load(fh)
    verd = {(r["rule_id"], round(r["budget"], 6)): r
            for r in arch["rows"] if r["policy_class"] == "monitoring_compiled"
            and r["status"] == "compiled"}

    pt, _ = parse_prometheus()
    bank = prom_threshold_bank(pt)
    rows_out: List[dict] = []
    for ti, t in enumerate(pt):
        pool = build_prom_pool(t, bank)
        ok, _why, readings = _eligible(pool)
        if not ok:
            continue
        uid = f'{getattr(t, "name", "?")}#{ti}'
        m = compile_instance(readings)
        K = m["C"].shape[0]
        crows = [m["C"][k].reshape(-1) for k in range(K)]
        surr = np.max(m["C"], axis=0).reshape(-1)
        for d in BUDGETS:
            V_U = _solve(m, crows, d)
            V_surr = _solve(m, [surr], d)
            V_singles = [_solve(m, [c], d) for c in crows]
            av = verd.get((uid, round(d, 6)))
            if V_U is None or av is None:
                continue
            # cross-check the archived verdict against the re-derived gap
            gap = min(v - V_U for v in V_singles if v is not None)
            assert av["screen_fires"] == (gap > TOL), (uid, d)
            rows_out.append(dict(
                rule_id=uid, budget=d, K=K,
                V_U=V_U, V_surr=(V_surr if V_surr is not None else None),
                V_best_single=max(v for v in V_singles if v is not None),
                surrogate_gap=(None if V_surr is None else float(V_U - V_surr)),
                surrogate_gap_rel=(None if (V_surr is None or V_U <= 0)
                                   else float((V_U - V_surr) / V_U)),
                screen_fires=bool(av["screen_fires"])))
    # aggregates at the operating budget
    op = [r for r in rows_out if r["budget"] == OPERATING]
    clear = [r for r in op if not r["screen_fires"]]
    pos = [r for r in clear if r["surrogate_gap"] is not None
           and r["surrogate_gap"] > TOL]
    rel = sorted(r["surrogate_gap_rel"] for r in pos)
    agg = dict(
        n_instances=len(op), n_clear=len(clear),
        n_clear_positive_gap=len(pos),
        rel_gap_min=(rel[0] if rel else None),
        rel_gap_med=(float(np.median(rel)) if rel else None),
        rel_gap_max=(rel[-1] if rel else None),
        infeasible_surrogate=sum(1 for r in op if r["V_surr"] is None))
    res = dict(
        registration=("post-hoc exact surrogate-price endpoint: V under the "
                      "pointwise-max single-cost surrogate vs V_U on every "
                      "eligible compiled instance; archived screen verdicts "
                      "asserted against re-derived gaps"),
        budgets=list(BUDGETS), operating=OPERATING,
        aggregate_at_operating=agg, rows=rows_out)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {OUT} ({len(rows_out)} rows)")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
