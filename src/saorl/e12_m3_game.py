"""E12 (REGISTRATION_V10): the exact game on real three-reading instances.

E10 solved the game on the 17 published control instances and found every
cell degenerate. E11 showed that is what M=2 predicts. The published
suite is M=2 everywhere because its outcome-blind selection caps
instances per source family and requires a crossing pair, not because
real rules carry two readings: over the eligible pinned Prometheus pools,
28 have three distinct (threshold, duration) pairs among the maximal
readings of the gold's subfamily, of which 27 are distinct instances
(two targets compile to an identical MDP).

This compiles those and solves the same game. M=3 gives 8 subsets and a
support bound of M+1=4.

Degeneracy test is E10's corrected one. The degenerate mixture is always
feasible, so the question is whether it is STRICTLY suboptimal; reading
the support the solver returns measures the solver.

Run:  PYTHONPATH=. python3 -m saorl.e12_m3_game
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from saorl.benchmark_sg import run_benchmark as rb
from saorl.benchmark_sg.control_mdp import solve_constrained
from saorl.benchmark_sg.control_suite import compile_instance
from saorl.benchmark_sg.parse import parse_prometheus, prom_threshold_bank

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/e2e" / "e12_m3_game.json"
BUDGETS = (0.005, 0.01, 0.02, 0.05)
DELTAS = (0.0, 0.01, 0.05, 0.10, 0.15)
TOL = 1e-9


def maximal_subfamily(pool):
    """Maximal readings of the gold's threshold/duration subfamily (A1)."""
    gold = pool.classes[pool.gold_idx].rep
    if gold.comparator not in (">", ">="):
        return []
    fam = [(c.rep, tuple(c.vector)) for c in pool.classes
           if (c.rep.metric, c.rep.selectors, c.rep.aggregation, c.rep.agg_by)
           == (gold.metric, gold.selectors, gold.aggregation, gold.agg_by)
           and c.rep.comparator in (">", ">=")]
    if len(fam) < 2:
        return []
    return [fam[i][0] for i, (_, vi) in enumerate(fam)
            if not any(j != i
                       and all(a <= b for a, b in zip(vi, fam[j][1]))
                       and any(b > a for a, b in zip(vi, fam[j][1]))
                       for j in range(len(fam)))]


def subset_values(m: dict, d: float, M: int):
    m = dict(m, C=m["C"])
    V = {}
    for k in range(M + 1):
        for T in itertools.combinations(range(M), k):
            A_ub = [m["C"][w].reshape(-1) for w in T]
            b_ub = [d] * len(T)
            nS, nA = m["nS"], m["nA"]
            from saorl.benchmark_sg.control_mdp import _flow_matrices
            A_eq, b_eq = _flow_matrices(m)
            res = linprog(-m["r"].reshape(-1),
                          A_ub=(np.array(A_ub) if A_ub else None),
                          b_ub=(np.array(b_ub) if b_ub else None),
                          A_eq=A_eq, b_eq=b_eq, bounds=(0, None),
                          method="highs")
            if res.status != 0:
                return {}
            V[frozenset(T)] = float(-res.fun)
    return V


def game(V, M, delta):
    keys = list(V.keys())
    A_ub = np.array([[1.0 if w not in T else 0.0 for T in keys]
                     for w in range(M)])
    res = linprog(np.array([-V[T] for T in keys]), A_ub=A_ub,
                  b_ub=np.full(M, delta), A_eq=np.ones((1, len(keys))),
                  b_eq=[1.0], bounds=(0, None), method="highs")
    if res.status != 0:
        return None, None
    x = np.maximum(res.x, 0.0)
    sup = {"+".join(str(w) for w in sorted(T)) or "(none)": float(x[i])
           for i, T in enumerate(keys) if x[i] > 1e-12}
    return float(-res.fun), sup


def main() -> None:
    pt, _ = parse_prometheus()
    bank = prom_threshold_bank(pt)
    rows, strict, maxsup, n_inst = [], [], 0, 0
    seen = set()

    for t in pt:
        mx = maximal_subfamily(rb.build_prom_pool(t, bank))
        pairs = sorted({(float(r.threshold), float(r.for_s)) for r in mx})
        if len(pairs) != 3:
            continue
        # Two pinned targets can compile to the same MDP (identical name and
        # identical (theta, duration) set). Counting both double-counts the
        # cells and overstates the denominator, so keep the first.
        uid = getattr(t, "name", str(t))
        if uid in seen:
            continue
        seen.add(uid)
        n_inst += 1
        M = 3

        class _R:
            def __init__(s, th, f):
                s.threshold, s.for_s = th, f
        m = compile_instance([_R(a, b) for a, b in pairs])
        for d in BUDGETS:
            V = subset_values(m, d, M)
            if not V:
                continue
            for delta in DELTAS:
                vstar, sup = game(V, M, delta)
                if vstar is None:
                    continue
                v_deg = (delta * V[frozenset()]
                         + (1 - delta) * V[frozenset(range(M))])
                is_strict = vstar > v_deg + TOL
                maxsup = max(maxsup, len(sup))
                rec = dict(uid=getattr(t, "name", str(t)), d=d, delta=delta,
                           V_star=vstar, V_degenerate=v_deg,
                           gap=vstar - v_deg, support=sup,
                           strictly_nondegenerate=is_strict)
                rows.append(rec)
                if is_strict:
                    strict.append(rec)

    out = dict(registration="REGISTRATION_V10 E12", M=3,
               n_instances=n_inst, cells=len(rows),
               strictly_nondegenerate=len(strict),
               frac=round(len(strict) / len(rows), 4) if rows else None,
               max_support=maxsup, support_bound=4,
               branch=("1: the mixture binds on a real instance" if strict
                       else "2: degenerate at M=3 as well"),
               examples=sorted(strict, key=lambda r: -r["gap"])[:10],
               rows=rows)
    os.makedirs(OUT.parent, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"real M=3 instances={n_inst} cells={len(rows)} "
          f"strictly non-degenerate={len(strict)} ({out['frac']}) "
          f"max support={maxsup}/4")
    print("registered branch ->", out["branch"])
    for e in out["examples"][:5]:
        print(f"   {e['uid']} d={e['d']} delta={e['delta']} "
              f"gap={e['gap']:.6f} support={ {k: round(v,4) for k,v in e['support'].items()} }")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
