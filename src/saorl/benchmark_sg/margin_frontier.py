"""V31: margin-aware Safe-Face -- the exact reward/robustness frontier.

V29 found the learned selector unsafe on a third of runs because the exact
target has Delta^0 = 0: the budget binds, so the safe-face optimum spends
exactly d and estimation error crosses it. The extension asks for a strict
buffer instead,

    Delta_psi^eps(d) <= -kappa,

and the claim under test is

    Delta_psi^eps(d) <= -kappa   <=>   V_psi(d) - V_U(d - kappa) <= eps,

so that eps*(kappa) = V_psi(d) - V_U(d - kappa) is the least reward tolerance
that buys safety buffer kappa. This module verifies that equivalence exactly,
by solving Delta directly from the tightened epigraph LP and comparing, and
traces the frontier over the registered kappa grid.

Registration: results/e2e/REGISTRATION_V28.md (V31 addendum).
Run: PYTHONPATH=. python3 -m saorl.benchmark_sg.margin_frontier
Writes results/e2e/margin_frontier.json
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

from .control_suite import _eligible, _flow, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SEL = os.path.join(ROOT, "results/e2e", "safe_face_select.json")
OUT = os.path.join(ROOT, "results/e2e", "margin_frontier.json")

BUDGET = 0.005
KAPPA_FRAC = (0.0, 0.1, 0.2, 0.3, 0.5)      # registered grid, kappa / d
EPS = 0.01
TOL = 1e-9
FACE_TOL = 1e-7


def _lp(m, obj, rows, rhs, maximize=True):
    A_eq, b_eq = _flow(m)
    res = linprog(-obj if maximize else obj,
                  A_ub=np.array(rows) if rows else None,
                  b_ub=np.array(rhs) if rows else None,
                  A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if res.status != 0:
        return None, None
    return float(obj @ res.x), res.x


def delta_kappa(m, r, crows, k, d, V_k, eps) -> Tuple[Optional[float],
                                                      Optional[np.ndarray]]:
    """min_x max_phi (c_phi'x - d) over the eps-face of reading k.

    The optimum is Delta_psi^eps(d); it is <= -kappa exactly when a policy on
    the face keeps every retained cost at or below d - kappa.
    """
    A_eq, b_eq = _flow(m)
    n = A_eq.shape[1]
    A_eq2 = np.hstack([A_eq, np.zeros((A_eq.shape[0], 1))])
    rows = [np.concatenate([crows[k], [0.0]]),
            np.concatenate([-r, [0.0]])]
    rhs = [d, -(V_k - eps - FACE_TOL)]
    for c in crows:
        rows.append(np.concatenate([c, [-1.0]]))
        rhs.append(d)
    obj = np.zeros(n + 1)
    obj[-1] = 1.0
    res = linprog(obj, A_ub=np.array(rows), b_ub=np.array(rhs),
                  A_eq=A_eq2, b_eq=b_eq,
                  bounds=[(0, None)] * n + [(None, None)], method="highs")
    if res.status != 0:
        return None, None
    return float(res.x[-1]), res.x[:n]


def main() -> None:
    with open(SEL, encoding="utf8") as fh:
        sel = json.load(fh)
    mid = [r for r in sel["rows"]
           if r["regime"] == "optimizer_resolvable"
           and abs(r["budget"] - BUDGET) < 1e-12]
    print(f"{len(mid)} optimizer-resolvable instances at d={BUDGET}")

    pt, _ = parse_prometheus()
    bank = prom_threshold_bank(pt)
    compiled = {}
    for ti, t in enumerate(pt):
        ok, _w, readings = _eligible(build_prom_pool(t, bank))
        if ok:
            compiled[f'{getattr(t, "name", "?")}#{ti}'] = compile_instance(readings)

    rows_out: List[dict] = []
    n_checked = 0
    for row in mid:
        m = compiled[row["rule_id"]]
        r = m["r"].reshape(-1)
        K = m["C"].shape[0]
        crows = [m["C"][k].reshape(-1) for k in range(K)]
        a = row["anchor"]
        V_a = row["V"][a]
        d = BUDGET

        # Delta at the true budget, once: the equivalence is a statement about
        # this single number against the tightened full-set values.
        Delta, _x = delta_kappa(m, r, crows, a, d, V_a, EPS)

        per_kappa = {}
        for kf in KAPPA_FRAC:
            kap = kf * d
            # right-hand side of the claim
            V_U_tight, x_t = _lp(m, r, crows, [d - kap] * K)
            eps_star = None if V_U_tight is None else V_a - V_U_tight
            rhs_holds = (V_U_tight is not None and eps_star <= EPS + TOL)
            # left-hand side, solved directly
            lhs_holds = (Delta is not None and Delta <= -kap + 1e-9)
            assert lhs_holds == rhs_holds, (
                row["rule_id"], kf, Delta, eps_star,
                "margin-aware equivalence failed")
            n_checked += 1
            sc = None
            if x_t is not None:
                cmax = max(float(c @ x_t) for c in crows)
                sc = dict(J_r=float(r @ x_t), C_max=cmax,
                          margin=float(d - cmax),
                          ret_frac=float((r @ x_t) / row["V_U"]),
                          safe=bool(cmax <= d + 1e-6))
            per_kappa[str(kf)] = dict(
                kappa=kap, V_U_tight=V_U_tight, eps_star=eps_star,
                feasible=bool(V_U_tight is not None),
                holds=bool(rhs_holds), selector=sc)
        rows_out.append(dict(rule_id=row["rule_id"], anchor=a, budget=d,
                             V_anchor=V_a, V_U=row["V_U"], Delta=Delta,
                             per_kappa=per_kappa))

    agg = {}
    for kf in KAPPA_FRAC:
        rs = [x["per_kappa"][str(kf)] for x in rows_out]
        ok = [x for x in rs if x["feasible"]]
        es = [x["eps_star"] for x in ok]
        rf = [x["selector"]["ret_frac"] for x in ok if x["selector"]]
        mg = [x["selector"]["margin"] / BUDGET for x in ok if x["selector"]]
        agg[str(kf)] = dict(
            kappa=kf * BUDGET,
            n_feasible=len(ok), n_holds=sum(1 for x in rs if x["holds"]),
            eps_star_med=(float(np.median(es)) if es else None),
            eps_star_max=(max(es) if es else None),
            ret_frac_of_VU_med=(float(np.median(rf)) if rf else None),
            margin_over_d_med=(float(np.median(mg)) if mg else None))

    res = dict(
        registration="V31 (REGISTRATION_V28.md addendum): margin-aware "
                     "safe-face. Verifies Delta <= -kappa iff "
                     "V_psi(d) - V_U(d-kappa) <= eps by solving both sides, "
                     f"on the {len(mid)} optimizer-resolvable instances at "
                     f"d={BUDGET}, eps={EPS}, kappa/d in {list(KAPPA_FRAC)}",
        budget=BUDGET, eps=EPS, kappa_frac=list(KAPPA_FRAC),
        n_equivalence_checks=n_checked, aggregate=agg, rows=rows_out)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {OUT}")
    print(f"equivalence asserted at {n_checked} (instance, kappa) points, "
          f"all agreed")
    print(f"{'kappa/d':>8} {'feasible':>9} {'eps*(med)':>11} "
          f"{'eps*(max)':>11} {'return/V_U':>11} {'margin/d':>9}")
    for kf in KAPPA_FRAC:
        v = agg[str(kf)]
        f = lambda x, w=11, p=4: (f"{x:>{w}.{p}f}" if x is not None
                                  else " " * (w - 2) + "--")
        print(f"{kf:>8} {v['n_feasible']:>9} {f(v['eps_star_med'])} "
              f"{f(v['eps_star_max'])} {f(v['ret_frac_of_VU_med'])} "
              f"{f(v['margin_over_d_med'], 9, 3)}")


if __name__ == "__main__":
    main()
