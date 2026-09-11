"""W7 bridge: registered-endpoint scoring + gate-attainability check.

Scores the registered gates EXACTLY as written, then checks whether the
Stage-0 certificate clause (per-seed episode-level Clopper-Pearson upper
bound <= d) is attainable at all by evaluating the SAME certificate on the
exact LP-optimal single-reading policy (the ground-truth best case).  If
the exact optimum fails the clause, the gate is unattainable as written
(an expected-cost budget does not bound the episode-cost tail), and the
registered primary is vacuous by gating rather than informative; the
exact-cost endpoints are then reported descriptively, labeled as such.

Writes results/e2e/w7_bridge_analysis.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import linprog

from .control_suite import compile_instance, GAMMA, _flow
from .w7_bridge_protocol import REGISTRATION, occupancy_fixed_policy
from .w7_bridge_run import episode_cert, exact_eval

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SUITE = os.path.join(ROOT, "results/conformal", "benchmark_sg", "control_suite.json")
BRIDGE = os.path.join(ROOT, "results/e2e", "w7_bridge.json")
OUT = os.path.join(ROOT, "results/e2e", "w7_bridge_analysis.json")
D = REGISTRATION["budget_d"]


def exact_policy(m, cost_sa):
    """Deterministic policy from the exact occupancy LP for one constraint."""
    A_eq, b_eq = _flow(m)
    r = m["r"].reshape(-1)
    res = linprog(-r, A_ub=cost_sa.reshape(1, -1), b_ub=[D],
                  A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    x = res.x.reshape(m["nS"], m["nA"])
    pol = x.argmax(axis=1)
    return pol


def main() -> None:
    suite = json.load(open(SUITE))
    bridge = json.load(open(BRIDGE))
    by_uid = {c["uid"]: c for c in bridge["rows"]}
    rows = []
    for inst in suite["instances"]:
        class R:
            pass
        readings = []
        for rd in inst["readings"]:
            rr = R(); rr.threshold, rr.for_s = rd["theta"], rd["for_s"]
            readings.append(rr)
        m = compile_instance(readings)
        pol_or = exact_policy(m, m["C"][0])                # exact psi1-optimal
        ret_or, J_or = exact_eval(m, pol_or)
        cp_or = [episode_cert(m, pol_or, s) for s in REGISTRATION["seeds"]]
        cell = by_uid[inst["uid"]]
        # registered Stage 0 as written (CP on worst-reading episodes <= d);
        # note the runner's stored cp is the worst-reading certificate
        s0 = sum(1 for s in cell["seeds"] if s["single"]["cp"] <= D)
        rows.append(dict(
            uid=inst["uid"],
            stage0_pass_seeds=s0,
            exact_single_own_J=J_or[0],
            exact_single_cp=cp_or,
            exact_gate_passes=bool(sum(1 for c in cp_or if c <= D) >= 4),
            learned_single_own_ok=sum(1 for s in cell["seeds"]
                                      if s["single"]["own_cost"] <= D),
            learned_set_worst_ok=sum(1 for s in cell["seeds"]
                                     if s["set"]["worst_cost"] <= D),
            hidden_gap_median=float(np.median([s["single"]["hidden_gap"]
                                               for s in cell["seeds"]])),
            set_worst_median=float(np.median([s["set"]["worst_cost"]
                                              for s in cell["seeds"]])),
        ))
    n = len(rows)
    admitted = sum(1 for r in rows if r["stage0_pass_seeds"] >= 4)
    exact_pass = sum(1 for r in rows if r["exact_gate_passes"])
    own_ok = sum(r["learned_single_own_ok"] for r in rows)
    set_ok = sum(r["learned_set_worst_ok"] for r in rows)
    out = dict(
        registration="REGISTRATION_W7 analysis",
        n_instances=n,
        stage0_admitted_as_registered=admitted,
        exact_optimum_passes_gate=exact_pass,
        gate_attainable=bool(exact_pass > 0),
        descriptive=dict(
            single_own_exact_ok=f"{own_ok}/{5*n}",
            set_worst_exact_ok=f"{set_ok}/{5*n}",
            hidden_gap_median=float(np.median([r["hidden_gap_median"] for r in rows])),
            set_worst_median=float(np.median([r["set_worst_median"] for r in rows])),
        ),
        rows=rows,
    )
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"stage0 admitted (as registered): {admitted}/{n}")
    print(f"exact LP optimum passes the same gate: {exact_pass}/{n} -> attainable={out['gate_attainable']}")
    print(f"descriptive: single own-ok {own_ok}/{5*n}; set worst-ok {set_ok}/{5*n}; "
          f"gap median {out['descriptive']['hidden_gap_median']:.4f} -> set {out['descriptive']['set_worst_median']:.4f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
