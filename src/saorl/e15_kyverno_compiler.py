"""E15 (REGISTRATION_V10): compile Kyverno rules into admission control models.

Five reviews object that the paper compares a COMPILED monitoring rate
(0.011) against a FREE-class admission rate (0.204). This builds the
missing compiler.

The model, licensed by E13 (no Kyverno reading carries a temporal field,
so nothing couples one decision to the next) and E14 (the two candidate
cost normalizations agree exactly on this family):

  states   one per fixture, the arriving request type, plus terminal
  arrival  uniform over the fixture universe
  actions  admit / reject
  reward   1 on admit, 0 on reject
  cost_k   v_k[i] on admit, 0 on reject
  horizon  1

Solved as an exact occupancy LP, the same object the monitoring side uses.

The sufficiency predicate is deliberately tie-free. E10 showed that
reading a solver's returned optimizer measures the solver when subset
values tie, so nothing here inspects an argmax: reading k suffices iff
the value achievable under psi_k alone is still achievable under every
reading at once.

Run:  PYTHONPATH=. python3 -m saorl.e15_kyverno_compiler
"""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import linprog

from saorl.benchmark_sg import run_benchmark as rb
from saorl.benchmark_sg.e4_screen import BUDGETS, screen_fires
from saorl.benchmark_sg.parse import parse_kyverno

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/e2e" / "e15_kyverno_compiled.json"
TOL = 1e-9


def compile_admission_mdp(vectors: Sequence[Sequence[float]]) -> dict:
    """The admission MDP for one pool. n request states, uniform arrival."""
    n = len(vectors[0])
    return dict(n=n, mu0=np.full(n, 1.0 / n),
                C=[np.asarray(v, dtype=float) for v in vectors])


def solve(m: dict, honored: Sequence[int], d: float):
    """max E[admits] over the occupancy polytope s.t. E[cost_k] <= d, k honored.

    Occupancy variable x_i = mass admitted in request state i, so
    0 <= x_i <= mu0_i and the objective is sum_i x_i."""
    n, mu0 = m["n"], m["mu0"]
    A_ub = [m["C"][k] for k in honored]
    b_ub = [d] * len(honored)
    res = linprog(-np.ones(n),
                  A_ub=(np.array(A_ub) if A_ub else None),
                  b_ub=(np.array(b_ub) if b_ub else None),
                  bounds=[(0.0, float(mu0[i])) for i in range(n)],
                  method="highs")
    return None if res.status != 0 else float(-res.fun)


def compiled_screen_fires(vectors, d: float) -> bool:
    M = len(vectors)
    if M < 2:
        return False
    m = compile_admission_mdp(vectors)
    V_all = solve(m, range(M), d)
    if V_all is None:
        return False
    for k in range(M):
        V_k = solve(m, [k], d)
        if V_k is None:
            continue
        if V_all >= V_k - TOL:      # reading k suffices
            return False
    return True


def main() -> None:
    kt, _ = parse_kyverno()
    recs = [rb._record(rb.build_kyv_pool(t), t, rb._kyv_skeleton(t))
            for t in kt]
    out = {"registration": "REGISTRATION_V10 E15", "n_pools": len(recs),
           "budgets": list(BUDGETS), "compiled": {}, "free": {}, "match": {}}
    for d in BUDGETS:
        c = sum(1 for r in recs if compiled_screen_fires(r.vectors, d))
        f = sum(1 for r in recs if screen_fires(r.vectors, d))
        out["compiled"][f"d={d}"] = round(c / len(recs), 4)
        out["free"][f"d={d}"] = round(f / len(recs), 4)
        out["match"][f"d={d}"] = (c == f)
        print(f"d={d:<5} compiled {c}/{len(recs)} = {c/len(recs):.4f}   "
              f"free {f}/{len(recs)} = {f/len(recs):.4f}   "
              f"match={c == f}")
    out["branch"] = ("1: compiled equals free -- the free class was the exact "
                     "compiled class for admission"
                     if all(out["match"].values()) else
                     "2: compiled differs -- the free class was a proxy")
    os.makedirs(OUT.parent, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nregistered branch ->", out["branch"])
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
