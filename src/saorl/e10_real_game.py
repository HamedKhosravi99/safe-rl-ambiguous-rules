"""E10 (REGISTRATION_V10): the exact semantic game on real compiled rules.

Theorem 4's optimum has only ever been solved on the four authored
domains, where it is degenerate in three of four -- mass delta on the
empty set, the rest on the full set -- so the M+1 mixture structure is
exhibited only by the synthetic witness of Proposition D.3. This asks
whether any SOURCE-GROUNDED instance behaves differently.

Every control-suite instance carries M=2 readings, so the subset program
has four variables and the support bound is M+1 = 3. Coefficients come
from the suite's own exact occupancy solver; the game is solved by the
same mixture LP used for the authored domains. Nothing is re-fit.

Degenerate means the optimal support is contained in {empty, full}.

Run:  PYTHONPATH=. python3 -m saorl.e10_real_game
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np

from saorl.benchmark_sg.control_mdp import solve_constrained
from saorl.benchmark_sg.control_suite import compile_instance
from saorl.shadow_price import mixture_lp

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "results/conformal" / "benchmark_sg" / "control_suite.json"
OUT = ROOT / "results/e2e" / "e10_real_game.json"

BUDGETS = (0.005, 0.01, 0.02, 0.05)
DELTAS = (0.0, 0.01, 0.05, 0.10, 0.15)
NAMES = ["r1", "r2"]


class _R:
    """compile_instance wants objects with .threshold and .for_s."""

    def __init__(self, theta, for_s):
        self.threshold, self.for_s = theta, for_s


def subset_values(m: dict, d: float) -> dict:
    """V_T for every T subset of {r1, r2}, via the exact occupancy LP.

    compile_instance stores per-reading costs in C[k]; solve_constrained
    reads cA/cB, so bind them here rather than duplicating the solver."""
    m = dict(m, cA=m["C"][0], cB=m["C"][1])
    V = {}
    for k in range(3):
        for T in itertools.combinations(NAMES, k):
            bA = d if "r1" in T else np.inf
            bB = d if "r2" in T else np.inf
            sol = solve_constrained(m, bA=bA, bB=bB)
            if not sol.get("feasible"):
                return {}
            V[frozenset(T)] = sol["ret"]
    return V


def degenerate(support: dict) -> bool:
    """Support confined to the empty set and the full set."""
    full = "+".join(sorted(NAMES))
    return all(k in ("(none)", full) for k in support)


def degenerate_achieves(V: dict, delta: float, V_star: float,
                        tol: float = 1e-9) -> bool:
    """Can the DEGENERATE mixture reach the optimum?

    Reading the LP's returned support is not a test of anything: when two
    subsets share a value the solver returns whichever vertex it lands on,
    and a support of {r1, r1+r2} can carry exactly the value of
    {empty, r1+r2}. The honest question is whether mass delta on the empty
    set and 1-delta on the full set is itself optimal. It is always
    feasible, so the instance is genuinely non-degenerate only when this
    value falls strictly short of V*."""
    v_deg = delta * V[frozenset()] + (1 - delta) * V[frozenset(NAMES)]
    return v_deg >= V_star - tol


def main() -> None:
    suite = json.load(open(SUITE))
    rows, cells = [], 0
    nondegen, maxsupport, nonabandon = [], 0, []

    for inst in suite["instances"]:
        readings = [_R(r["theta"], r["for_s"]) for r in inst["readings"]]
        if len(readings) != 2:
            continue
        m = compile_instance(readings)
        for d in BUDGETS:
            V = subset_values(m, d)
            if not V:
                continue
            for delta in DELTAS:
                res = mixture_lp(V, NAMES, delta)
                sup = res["support"]
                cells += 1
                maxsupport = max(maxsupport, len(sup))
                deg_ok = degenerate_achieves(V, delta, res["V_star"])
                rec = dict(uid=inst["uid"], geometry=inst["geometry"], d=d,
                           delta=delta, V_star=res["V_star"],
                           support={k: round(v, 6) for k, v in sup.items()},
                           support_is_degenerate=degenerate(sup),
                           degenerate_optimal=deg_ok,
                           V_degenerate=delta * V[frozenset()]
                           + (1 - delta) * V[frozenset(NAMES)])
                # genuinely non-degenerate: the degenerate mixture FALLS SHORT
                if not deg_ok:
                    nondegen.append(rec)
                if "(none)" not in sup and len(sup) > 1:
                    nonabandon.append(rec)
                rows.append(rec)

    out = dict(
        registration="REGISTRATION_V10 E10",
        n_instances=len({r["uid"] for r in rows}),
        cells=cells,
        nondegenerate=len(nondegen),
        nondegenerate_frac=round(len(nondegen) / cells, 4) if cells else None,
        max_support_observed=maxsupport,
        support_bound=len(NAMES) + 1,
        nonabandoning_multi=len(nonabandon),
        branch=("1: a real instance is genuinely non-degenerate"
                if nondegen else
                "2: every real cell is degenerate -- the mixture structure "
                "is exercised only by construction"),
        examples=nondegen[:10],
        rows=rows,
    )
    os.makedirs(OUT.parent, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"instances={out['n_instances']} cells={cells} "
          f"non-degenerate={len(nondegen)} ({out['nondegenerate_frac']})  "
          f"max support={maxsupport}/{out['support_bound']}")
    print("registered branch ->", out["branch"])
    for e in nondegen[:6]:
        print("   ", e["uid"], f"d={e['d']} delta={e['delta']}", e["support"])
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
