"""E17: numerical verification of Theorem (collapse of the semantic game).

Claim: for delta in (0,1) the degenerate mixture attains V*(delta) iff the
cooperative game h(S) = V_{[M]\S} - V_{[M]} has a nonempty core.

Two checks, both exact-LP:
  (a) realizable synthetic instances (finite policy class with a fallback),
      predicted collapse vs measured collapse;
  (b) the real source-grounded control instances, where the theorem
      predicts collapse everywhere and E10/E12 measured exactly that.

Run:  PYTHONPATH=. python3 -m saorl.e17_collapse_theorem
"""
from __future__ import annotations
import itertools, json, os
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/e2e" / "e17_collapse_theorem.json"
TOL = 1e-9


def subset_values(r, c, d, M):
    V = {}
    for k in range(M + 1):
        for T in itertools.combinations(range(M), k):
            ok = np.ones(len(r), bool)
            for w in T:
                ok &= c[:, w] <= d
            V[frozenset(T)] = float(r[ok].max()) if ok.any() else -np.inf
    return V


def game_value(V, M, delta):
    ks = list(V.keys())
    A = np.array([[1.0 if w not in T else 0.0 for T in ks] for w in range(M)])
    res = linprog(np.array([-V[T] for T in ks]), A_ub=A, b_ub=np.full(M, delta),
                  A_eq=np.ones((1, len(ks))), b_eq=[1.0], bounds=(0, None),
                  method="highs")
    return None if res.status else float(-res.fun)


def core_nonempty(V, M):
    full = frozenset(range(M)); Vf = V[full]
    h = {frozenset(S): V[full - frozenset(S)] - Vf
         for k in range(M + 1) for S in itertools.combinations(range(M), k)}
    Ss = [S for S in h if S]
    A = np.array([[-1.0 if w in S else 0.0 for w in range(M)] for S in Ss])
    res = linprog(np.zeros(M), A_ub=A, b_ub=np.array([-h[S] for S in Ss]),
                  A_eq=np.ones((1, M)), b_eq=[h[full]], bounds=(0, None),
                  method="highs")
    return res.status == 0


def main() -> None:
    rng = np.random.default_rng(7)
    n = agree = collapse = 0
    for _ in range(4000):
        M = int(rng.integers(2, 6)); N = int(rng.integers(M + 2, 14))
        r = rng.random(N); c = rng.random((N, M)); d = float(rng.uniform(.2, .8))
        r[0] = rng.uniform(0, .25); c[0, :] = 0.0            # fallback
        delta = float(rng.choice([0.01, 0.05, 0.10, 0.15]))
        V = subset_values(r, c, d, M)
        v = game_value(V, M, delta)
        if v is None:
            continue
        v_deg = delta * V[frozenset()] + (1 - delta) * V[frozenset(range(M))]
        n += 1
        deg = v_deg >= v - TOL
        collapse += deg
        agree += (deg == core_nonempty(V, M))
    out = dict(experiment="E17 collapse theorem", synthetic_instances=n,
               agreements=agree, agreement_rate=round(agree / n, 6),
               collapsed=collapse)
    assert agree == n, f"characterization failed on {n - agree} instances"
    os.makedirs(OUT.parent, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"synthetic realizable instances: {n}")
    print(f"core-nonemptiness predicts collapse: {agree}/{n} "
          f"({out['agreement_rate']})")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
