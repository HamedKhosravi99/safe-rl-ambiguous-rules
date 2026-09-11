"""A worked instance where Theorem 4's mixture is genuinely exercised.

Table 5 records that on all four reported domains the exact game value is
attained by a degenerate mixture -- mass delta on the empty set and 1-delta
on the full set -- so the M+1-support structure of Theorem 4 is stated but
never exhibited. Review #33 asked for an instance that exercises it.

This constructs one, and the construction is interpretable rather than
random. Take M symmetric readings and a one-step decision with M+1 actions:
a cautious action feasible for every reading, and for each reading w a
profitable action that violates w alone. Then

  V_T = r_hi  whenever some reading is unhonored (play the action that
              violates exactly an unhonored one),
  V_[M] = r_lo  when every reading must be honored.

The empty set is not special: it also gives r_hi. But the LP's constraint
is per reading -- sum over subsets NOT containing w must be at most delta --
and the empty set is excluded by *every* w, so it can carry at most delta of
mass in total. Each near-full subset [M]\\{w} is excluded by only one
reading, so each can carry delta. The optimum therefore spreads M*delta
across the near-full subsets instead of delta on the empty set, and its
support has M+1 atoms.

The reading: a minimax-optimal uniformly valid algorithm does not abandon
every reading delta of the time. It abandons a *different single* reading
delta of the time, which is both better and deployable -- the mixture is
already non-abandoning, so Corollary D.2's constraint binds at no cost here.

Run:  PYTHONPATH=. python3 -m saorl.mixture_witness
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.optimize import linprog

OUT = Path(__file__).parent.parent.parent / "results/e2e" / "mixture_witness.json"


def subset_values(M: int, r_hi: float, r_lo: float) -> Dict[frozenset, float]:
    """Exact optimal value of the constrained problem for each honored subset.

    One step, actions {cautious} u {viol_w : w in [M]}. The cautious action
    honors every reading and returns r_lo; viol_w returns r_hi and violates
    reading w alone. Honoring T means only actions violating no member of T
    are feasible, so the value is r_hi iff some reading is outside T.
    """
    V = {}
    for r in range(M + 1):
        for T in itertools.combinations(range(M), r):
            T = frozenset(T)
            V[T] = r_lo if len(T) == M else r_hi
    return V


def game_value(M: int, V: Dict[frozenset, float], delta: float):
    subsets = list(V)
    c = np.array([-V[T] for T in subsets])
    A = np.array([[1.0 if w not in T else 0.0 for T in subsets] for w in range(M)])
    res = linprog(c, A_ub=A, b_ub=np.full(M, delta),
                  A_eq=np.ones((1, len(subsets))), b_eq=[1.0],
                  bounds=(0, None), method="highs")
    assert res.status == 0, res.message
    x = np.maximum(res.x, 0.0)
    support = {subsets[i]: float(x[i]) for i in range(len(subsets)) if x[i] > 1e-9}
    return float(-res.fun), support


def main() -> None:
    r_hi, r_lo, delta = 1.0, 0.0, 0.1
    out = {"construction": "symmetric one-step; cautious action honors all "
                           "readings at r_lo, action viol_w returns r_hi and "
                           "violates reading w alone",
           "r_hi": r_hi, "r_lo": r_lo, "delta": delta, "instances": {}}
    for M in (2, 3, 4, 5):
        V = subset_values(M, r_hi, r_lo)
        val, sup = game_value(M, V, delta)
        abandons = any(len(T) == 0 for T in sup)
        rec = dict(M=M, value=round(val, 6), support_size=len(sup),
                   puts_mass_on_empty=bool(abandons),
                   support={("empty" if not T else "+".join(map(str, sorted(T)))):
                            round(w, 6) for T, w in sup.items()},
                   degenerate=bool(len(sup) <= 2))
        out["instances"][f"M={M}"] = rec
        print(f"M={M}: value {val:.4f}  support {len(sup)} atoms  "
              f"abandons-all: {abandons}  {'DEGENERATE' if rec['degenerate'] else 'MIXTURE EXERCISED'}")
        for T, w in sorted(sup.items(), key=lambda kv: (-kv[1], sorted(kv[0]))):
            print(f"     {w:.4f} on {sorted(T) if T else 'empty'}")

    # the claim: from M=3 the support exceeds two atoms and never uses the
    # empty set, so the mixture is exercised AND already non-abandoning
    for M in (3, 4, 5):
        r = out["instances"][f"M={M}"]
        assert r["support_size"] == M + 1, r
        assert not r["puts_mass_on_empty"], r
    print(f"\nsupport size equals M+1 exactly at M=3,4,5, with no mass on the "
          f"empty set:\nTheorem 4's bound is attained and the optimum is "
          f"non-abandoning without imposing Corollary D.2.")
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
