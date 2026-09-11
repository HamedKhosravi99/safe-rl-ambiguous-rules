"""E11 (post-hoc, exploratory): how often is Theorem 4's optimum degenerate?

E10 found the game optimum degenerate in all 340 real cells and two
reviewers concluded the mixture structure is empirically inert. That
conclusion does not follow from E10, because every priceable instance we
compile carries exactly M=2 readings. This measures the degeneracy rate
as a function of M on realizable instances.

NOT REGISTERED IN ADVANCE. It was written after E10's falsification of a
collapse conjecture, and it is reported as exploratory. What it can
establish is that the structure is not vacuous; what it cannot establish
is anything about real rules, because the instances here are synthetic.

Generator. A finite policy class, which is realizable by construction:
policy i has return r_i and cost c_i over M readings, reading w is
honoured by i iff c_i[w] <= d, and V_T = max{r_i : honours all of T}.
One policy honours every reading at low return, which is exactly
Assumption (fallback), so V_T is finite for every T and the game is
always feasible.

Endpoint: the fraction of instances where the degenerate mixture (mass
delta on the empty set, 1-delta on the full set) is STRICTLY suboptimal.
Reading the LP's returned support is not a test -- see E10, where ties
between V_empty and V_{single} made the returned vertex meaningless.

Run:  PYTHONPATH=. python3 -m saorl.e11_degeneracy_rate
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/e2e" / "e11_degeneracy_rate.json"

M_RANGE = (2, 3, 4, 5, 6)
DELTAS = (0.01, 0.05, 0.10, 0.15)
TRIALS_PER_M = 1500
SEED = 1
TOL = 1e-9


def subset_values(r, c, d, M):
    V = {}
    for k in range(M + 1):
        for T in itertools.combinations(range(M), k):
            ok = np.ones(len(r), dtype=bool)
            for w in T:
                ok &= c[:, w] <= d
            V[frozenset(T)] = float(r[ok].max()) if ok.any() else -np.inf
    return V


def game_value(V, M, delta):
    keys = list(V.keys())
    obj = np.array([-V[T] for T in keys])
    A_ub = np.array([[1.0 if w not in T else 0.0 for T in keys]
                     for w in range(M)])
    res = linprog(obj, A_ub=A_ub, b_ub=np.full(M, delta),
                  A_eq=np.ones((1, len(keys))), b_eq=[1.0],
                  bounds=(0, None), method="highs")
    return None if res.status != 0 else float(-res.fun)


def main() -> None:
    rng = np.random.default_rng(SEED)
    per_m = {}
    for M in M_RANGE:
        strict = total = 0
        for _ in range(TRIALS_PER_M):
            N = int(rng.integers(M + 2, 16))
            r = rng.random(N)
            c = rng.random((N, M))
            d = float(rng.uniform(0.2, 0.8))
            r[0] = rng.uniform(0.0, 0.25)      # fallback policy ...
            c[0, :] = 0.0                      # ... honours every reading
            delta = float(rng.choice(DELTAS))
            V = subset_values(r, c, d, M)
            v = game_value(V, M, delta)
            if v is None:
                continue
            v_deg = (delta * V[frozenset()]
                     + (1 - delta) * V[frozenset(range(M))])
            total += 1
            strict += v > v_deg + TOL
        per_m[M] = dict(n=total, nondegenerate=int(strict),
                        frac=round(strict / total, 4) if total else None)
        print(f"M={M}: {strict}/{total} strictly non-degenerate "
              f"({per_m[M]['frac']})")

    tot_n = sum(v["n"] for v in per_m.values())
    tot_s = sum(v["nondegenerate"] for v in per_m.values())
    fracs = [per_m[M]["frac"] for M in M_RANGE]
    out = dict(
        registration="post-hoc, exploratory (not registered in advance)",
        generator="finite policy class with a fallback honouring all readings",
        seed=SEED, deltas=list(DELTAS), trials_per_M=TRIALS_PER_M,
        per_M=per_m, overall_n=tot_n, overall_nondegenerate=tot_s,
        overall_frac=round(tot_s / tot_n, 4) if tot_n else None,
        monotone_in_M=all(a <= b + 1e-12 for a, b in zip(fracs, fracs[1:])),
    )
    os.makedirs(OUT.parent, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\noverall {tot_s}/{tot_n} ({out['overall_frac']}), "
          f"monotone in M: {out['monotone_in_M']}")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
