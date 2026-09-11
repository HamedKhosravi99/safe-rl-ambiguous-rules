"""P0-C + P0-D: the exact non-nested singleton-versus-set analysis, tie-safe.

P0-C (primary non-nested consequence experiment).  On every compiled
source-grounded instance, at every budget:

  1. solve V_U exactly (robust: every retained reading <= d);
  2. solve every V_psi exactly (singleton-constrained optimum);
  3. record an exact singleton-optimal occupancy for each psi;
  4. evaluate every singleton optimum against every retained reading;
  5. hidden_gap = max_phi J_{c_phi} - J_{c_honored};
  6. stratify by whether the screen fires at that budget;
  7. report exact return loss, worst-cost/budget ratio, violation rate.

No learner is involved: every policy here is an exact occupancy-LP optimum,
so nothing in this table mixes optimizer error into the phenomenon.

TIE SAFETY (the plan's explicit requirement: "never let arbitrary LP
tie-breaking decide a scientific claim").  A singleton optimum is generally
non-unique, and its worst retained cost is not a function of the value
alone.  For each (instance, budget, psi) we therefore report three
quantities over the singleton-optimal FACE
      F_psi = {x feasible : <c_psi, x> <= d, <r, x> >= V_psi - tol}:
  * returned : worst retained cost of the occupancy the solver happened to
               return (this is what a naive study would report);
  * best     : min over F_psi of max_phi <c_phi, x>   (one epigraph LP)
               -- the most favourable tie-break for the singleton;
  * worst    : max over F_psi of max_phi <c_phi, x>   (K LPs)
               -- the least favourable tie-break.
The paper's consequence claim is made with `best`, i.e. against the
singleton's most favourable tie-break, so it cannot be an artifact.

P0-D (screen definition check).  The main paper defines the operating-point
screen by VALUE EQUALITY,
      screen fires  iff  min_psi [ V_psi(d) - V_U(d) ] > 0.
The shipped control-suite predicate instead asks whether a REPRESENTATIVE
singleton optimum returned by the solver is full-set feasible.  These are
equivalent in exact arithmetic:

  (=>) If V_psi = V_U then the robust optimum itself is psi-optimal and
       full-set feasible, so some singleton-optimal policy clears the set.
  (<=) If some psi-optimal policy is full-set feasible then V_psi <= V_U,
       and V_psi >= V_U always, so V_psi = V_U.

but the shipped form is evaluated at ONE optimizer, so a tie-break can flip
its label while the value form cannot.  This module recomputes both labels
on every (instance, budget) and reports disagreements, plus the numerical
value gap on screen-clear cells (the tolerance actually needed).

Run:  PYTHONPATH=. python3 -m saorl.benchmark_sg.exact_nonnested
Writes results/e2e/exact_nonnested.json
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
from scipy.optimize import linprog

from .control_suite import compile_instance, GAMMA, _flow

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SUITE = os.path.join(ROOT, "results/conformal", "benchmark_sg", "control_suite.json")
OUT = os.path.join(ROOT, "results/e2e", "exact_nonnested.json")

BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10)
OPERATING = 0.05
TOL = 1e-7               # value-equality tolerance for the screen
FACE_SLACK = 1e-9        # return slack defining the optimal face


# ---------------------------------------------------------------------------
# exact occupancy programs
# ---------------------------------------------------------------------------

def _solve(m: dict, cost_rows: List[np.ndarray], budgets: List[float],
           objective: np.ndarray, sense: str = "max",
           extra_ub: List[tuple] = ()) -> dict:
    """Occupancy LP: optimize <objective, x> subject to flow, the given cost
    budgets, and any extra (row, rhs) upper bounds.  Returns value, x, and a
    duality-gap diagnostic."""
    A_eq, b_eq = _flow(m)
    rows = [c.reshape(-1) for c in cost_rows] + [np.asarray(r).reshape(-1)
                                                 for r, _ in extra_ub]
    rhs = list(budgets) + [v for _, v in extra_ub]
    A_ub = np.array(rows) if rows else None
    b_ub = np.array(rhs) if rows else None
    c = -objective.reshape(-1) if sense == "max" else objective.reshape(-1)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, None), method="highs")
    if res.status != 0:
        return dict(feasible=False, value=None, x=None, gap=None)
    val = float(-res.fun if sense == "max" else res.fun)
    # duality gap: c'x - (b_eq'y + b_ub'lambda) with scipy's marginal signs
    gap = None
    try:
        y = np.asarray(res.eqlin.marginals)
        dual = float(b_eq @ y)
        if A_ub is not None:
            lam = np.asarray(res.ineqlin.marginals)
            dual += float(b_ub @ lam)
        gap = abs(float(res.fun) - dual)
    except Exception:
        pass
    return dict(feasible=True, value=val, x=res.x, gap=gap)


def _J(m: dict, k: int, x: np.ndarray) -> float:
    return float(m["C"][k].reshape(-1) @ x)


def analyse(m: dict, d: float) -> dict:
    """All P0-C/P0-D quantities for one compiled instance at one budget."""
    K = m["C"].shape[0]
    r = m["r"]
    costs = [m["C"][k] for k in range(K)]

    robust = _solve(m, costs, [d] * K, r)
    if not robust["feasible"]:
        return dict(feasible=False, budget=d)
    V_U = robust["value"]
    robust_worst = max(_J(m, k, robust["x"]) for k in range(K))

    singles = []
    for k in range(K):
        s = _solve(m, [costs[k]], [d], r)
        if not s["feasible"]:
            singles.append(dict(k=k, feasible=False))
            continue
        V_k = s["value"]
        x = s["x"]
        J_all = [_J(m, j, x) for j in range(K)]
        returned_worst = max(J_all)

        # optimal face: <r,x> >= V_k - slack, honoured constraint kept
        face_extra = [(-r, -(V_k - FACE_SLACK))]

        # best tie-break: min over the face of max_phi <c_phi, x>  (epigraph)
        # variables [x, t]; minimize t s.t. <c_phi,x> - t <= 0 for all phi
        A_eq, b_eq = _flow(m)
        n = A_eq.shape[1]
        A_eq2 = np.hstack([A_eq, np.zeros((A_eq.shape[0], 1))])
        rows, rhs = [], []
        for j in range(K):
            rows.append(np.append(costs[j].reshape(-1), -1.0)); rhs.append(0.0)
        rows.append(np.append(costs[k].reshape(-1), 0.0)); rhs.append(d)
        rows.append(np.append(-r.reshape(-1), 0.0)); rhs.append(-(V_k - FACE_SLACK))
        obj = np.zeros(n + 1); obj[-1] = 1.0
        res = linprog(obj, A_ub=np.array(rows), b_ub=np.array(rhs),
                      A_eq=A_eq2, b_eq=b_eq,
                      bounds=[(0, None)] * n + [(None, None)], method="highs")
        best_worst = float(res.fun) if res.status == 0 else None

        # worst tie-break: max over the face of max_phi <c_phi,x> = max_phi max_x
        worst_worst = None
        vals = []
        for j in range(K):
            w = _solve(m, [costs[k]], [d], costs[j], sense="max",
                       extra_ub=face_extra)
            if w["feasible"]:
                vals.append(w["value"])
        if vals:
            worst_worst = max(vals)

        singles.append(dict(
            k=k, feasible=True, V=V_k, honored=J_all[k], J_all=J_all,
            returned_worst=returned_worst,
            best_worst=best_worst, worst_worst=worst_worst,
            hidden_gap_returned=returned_worst - J_all[k],
            hidden_gap_best=(None if best_worst is None
                             else best_worst - J_all[k]),
            gap=s["gap"]))

    feas = [s for s in singles if s.get("feasible")]
    V_best = max(s["V"] for s in feas)
    screen_gap = min(s["V"] - V_U for s in feas)
    fires_value = bool(screen_gap > TOL)
    # P0-D registration: the optimal-FACE form of the equivalence, and the
    # identity of the value-attaining singleton with the face-feasible one.
    face_clear = any(s["best_worst"] is not None and s["best_worst"] <= d + TOL
                     for s in feas)
    argmin_V = {s["k"] for s in feas if abs(s["V"] - V_U) <= TOL}
    face_set = {s["k"] for s in feas
                if s["best_worst"] is not None and s["best_worst"] <= d + TOL}
    tiesafe_ok = bool(((not fires_value) == face_clear)
                      and (fires_value or face_set == argmin_V))

    # consequence quantities, computed against the singleton's MOST FAVOURABLE
    # tie-break (best_worst) so the claim is tie-safe
    viol_best = [s for s in feas
                 if s["best_worst"] is not None and s["best_worst"] > d + TOL]
    any_viol = bool(viol_best)
    max_best_ratio = max((s["best_worst"] / d) for s in feas
                         if s["best_worst"] is not None)

    return dict(
        feasible=True, budget=d, K=K,
        V_U=V_U, V_best_single=V_best,
        poa=V_best - V_U, rpoa=(V_best - V_U) / V_best if V_best > 0 else 0.0,
        robust_worst=robust_worst, robust_worst_ratio=robust_worst / d,
        screen_gap=screen_gap, fires_value=fires_value,
        face_clear=face_clear, tiesafe_ok=tiesafe_ok,
        n_singletons=len(feas),
        n_singletons_violating_best=len(viol_best),
        any_singleton_violates_best=any_viol,
        max_best_worst_ratio=max_best_ratio,
        median_hidden_gap_best=float(np.median(
            [s["hidden_gap_best"] for s in feas
             if s["hidden_gap_best"] is not None])),
        lp_gap_max=max([g for g in [robust["gap"]] + [s.get("gap") for s in feas]
                        if g is not None], default=None),
        singletons=singles,
    )


def main() -> None:
    suite = json.load(open(SUITE))
    rows = []
    for inst in suite["instances"]:
        class R:
            pass
        readings = []
        for rd in inst["readings"]:
            rr = R(); rr.threshold, rr.for_s = rd["theta"], rd["for_s"]
            readings.append(rr)
        m = compile_instance(readings)
        per_budget = {}
        for d in BUDGETS:
            a = analyse(m, d)
            # P0-D: compare against the shipped representative-optimizer label
            shipped = inst["budgets"].get(f"{d}") or inst["budgets"].get(str(d))
            a["fires_shipped"] = (None if shipped is None
                                  else bool(shipped["no_singleton_fullset_feasible"]))
            a["label_agrees"] = (None if a["fires_shipped"] is None
                                 else a["fires_value"] == a["fires_shipped"])
            per_budget[str(d)] = a
        rows.append(dict(uid=inst["uid"], geometry=inst["geometry"],
                         n_states=m["nS"], K=m["C"].shape[0],
                         budgets=per_budget))
        op = per_budget[str(OPERATING)]
        print(f"  {inst['uid']:38s} d={OPERATING} fires={op['fires_value']} "
              f"(shipped {op['fires_shipped']}) gap={op['screen_gap']:.2e} "
              f"viol_best={op['n_singletons_violating_best']}/{op['n_singletons']} "
              f"maxratio={op['max_best_worst_ratio']:.2f}", flush=True)

    # ---- aggregate the P0-C headline quantities at the operating budget
    def agg(d: float) -> dict:
        cells = [r["budgets"][str(d)] for r in rows if r["budgets"][str(d)]["feasible"]]
        sing_total = sum(c["n_singletons"] for c in cells)
        sing_viol = sum(c["n_singletons_violating_best"] for c in cells)
        inst_viol = sum(1 for c in cells if c["any_singleton_violates_best"])
        fire = [c for c in cells if c["fires_value"]]
        clear = [c for c in cells if not c["fires_value"]]
        return dict(
            n_instances=len(cells),
            singleton_violation_frac=sing_viol / sing_total if sing_total else 0.0,
            n_singletons=sing_total, n_singletons_violating=sing_viol,
            instance_violation_frac=inst_viol / len(cells) if cells else 0.0,
            median_hidden_gap=float(np.median([c["median_hidden_gap_best"] for c in cells])),
            median_singleton_worst_ratio=float(np.median(
                [c["max_best_worst_ratio"] for c in cells])),
            median_robust_worst_ratio=float(np.median(
                [c["robust_worst_ratio"] for c in cells])),
            median_rpoa=float(np.median([c["rpoa"] for c in cells])),
            n_fire=len(fire), n_clear=len(clear),
            fire_violation_frac=(sum(1 for c in fire if c["any_singleton_violates_best"])
                                 / len(fire) if fire else None),
            clear_violation_frac=(sum(1 for c in clear if c["any_singleton_violates_best"])
                                  / len(clear) if clear else None),
            max_clear_value_gap=(max((c["screen_gap"] for c in clear), default=0.0)),
            label_disagreements=sum(1 for c in cells if c["label_agrees"] is False),
            max_lp_gap=max((c["lp_gap_max"] for c in cells if c["lp_gap_max"] is not None),
                           default=None),
            tiesafe_ok_all=bool(all(c["tiesafe_ok"] for c in cells)),
            n_tiesafe_checked=len(cells),
            median_worst_ratio_fire=(float(np.median(
                [c["max_best_worst_ratio"] for c in fire])) if fire else None),
            median_worst_ratio_clear=(float(np.median(
                [c["max_best_worst_ratio"] for c in clear])) if clear else None),
            median_rpoa_fire=(float(np.median([c["rpoa"] for c in fire]))
                              if fire else None),
            median_rpoa_clear=(float(np.median([c["rpoa"] for c in clear]))
                               if clear else None),
        )

    out = dict(
        registration="P0-C exact non-nested consequence + P0-D screen check",
        budgets=list(BUDGETS), operating=OPERATING, tol=TOL,
        aggregate={str(d): agg(d) for d in BUDGETS},
        instances=rows,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    import csv as _csv
    csv_path = OUT.replace(".json", ".csv")
    with open(csv_path, "w", newline="") as fh:
        wtr = _csv.writer(fh)
        wtr.writerow(["uid", "geometry", "n_states", "K", "budget", "V_U",
                      "V_best_single", "poa", "rpoa", "screen_gap",
                      "fires_value", "fires_shipped", "label_agrees",
                      "tiesafe_ok", "n_singletons", "n_singletons_violating",
                      "median_hidden_gap_best", "max_best_worst_ratio",
                      "robust_worst_ratio", "lp_gap_max"])
        for r in rows:
            for b, c in r["budgets"].items():
                if not c.get("feasible"):
                    continue
                wtr.writerow([r["uid"], r["geometry"], r["n_states"], r["K"], b,
                              c["V_U"], c["V_best_single"], c["poa"], c["rpoa"],
                              c["screen_gap"], c["fires_value"],
                              c["fires_shipped"], c["label_agrees"],
                              c["tiesafe_ok"], c["n_singletons"],
                              c["n_singletons_violating_best"],
                              c["median_hidden_gap_best"],
                              c["max_best_worst_ratio"],
                              c["robust_worst_ratio"], c["lp_gap_max"]])
    print("wrote", csv_path)
    a = out["aggregate"][str(OPERATING)]
    print(f"\n[P0-C @ d={OPERATING}] instances={a['n_instances']} "
          f"singleton-violation={a['singleton_violation_frac']:.3f} "
          f"({a['n_singletons_violating']}/{a['n_singletons']}); "
          f"instance-violation={a['instance_violation_frac']:.3f}")
    print(f"           median hidden gap={a['median_hidden_gap']:.4f}; "
          f"singleton worst/d={a['median_singleton_worst_ratio']:.2f} vs "
          f"robust {a['median_robust_worst_ratio']:.2f}")
    print(f"[P0-D] label disagreements value-vs-shipped: {a['label_disagreements']}; "
          f"max value gap on screen-clear cells={a['max_clear_value_gap']:.2e}; "
          f"max LP duality gap={a['max_lp_gap']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
