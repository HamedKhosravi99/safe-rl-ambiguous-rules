"""E4 (REGISTRATION_V10): the screen on both families, on the fixture universe.

Does a natural-language rule force set defense? On Prometheus the paper
answers with compiled control models, which is why the published figure is
Prometheus-only. Here the same question is asked on the artifacts as they
stand: every reading already carries a cost vector over the fixture universe
behind the published dominance rates, and `dominance.nu_pi` already optimises
over the free policy class on it.

For a pool with readings psi_1..psi_M over fixtures F, a policy selects
S subset of F, its utility is |S|, and it honours psi_k at budget d when the
mean psi_k-cost over S is at most d. Reading psi_k SUFFICES at d when the
psi_k-optimal selection also satisfies every other reading at d. The screen
FIRES when no reading suffices -- set defense is then forced.

Run:  PYTHONPATH=. python3 -m saorl.benchmark_sg.e4_screen
"""
from __future__ import annotations

import json
import os
from typing import List, Sequence

BUDGETS = (0.01, 0.02, 0.05, 0.10)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "results/e2e", "e4_screen.json")


def best_selection(v: Sequence[float], d: float) -> List[int]:
    """Largest S with mean v-cost <= d. Greedy on ascending cost is optimal:
    for a fixed |S| the cheapest |S| fixtures minimise the mean, so the
    feasible sizes are a prefix-closed set and the largest is a prefix."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    best, run = [], 0.0
    for n, i in enumerate(order, start=1):
        run += v[i]
        if run / n <= d + 1e-12:
            best = order[:n]
    return best


def suffices(vectors: Sequence[Sequence[float]], k: int, d: float) -> bool:
    S = best_selection(vectors[k], d)
    if not S:
        return False
    for j, w in enumerate(vectors):
        if sum(w[i] for i in S) / len(S) > d + 1e-12:
            return False
    return True


def screen_fires(vectors: Sequence[Sequence[float]], d: float) -> bool:
    if len(vectors) < 2:
        return False
    return not any(suffices(vectors, k, d) for k in range(len(vectors)))


def main() -> None:
    from . import run_benchmark as rb

    from .parse import parse_prometheus, parse_kyverno
    pt, _ = parse_prometheus()
    kt, _ = parse_kyverno()
    bank = rb.prom_threshold_bank(pt) if hasattr(rb, "prom_threshold_bank") else None
    if bank is None:
        from .parse import prom_threshold_bank
        bank = prom_threshold_bank(pt)
    recs = [rb._record(rb.build_prom_pool(t, bank), t, rb._prom_skeleton(t)) for t in pt]
    recs += [rb._record(rb.build_kyv_pool(t), t, rb._kyv_skeleton(t)) for t in kt]
    by_family = {}
    for r in recs:
        by_family.setdefault(r.family, []).append(r)

    out = {"registration": "REGISTRATION_V10 E4", "budgets": list(BUDGETS),
           "families": {}}
    for fam, rs in sorted(by_family.items()):
        rec = {"n_pools": len(rs), "fires": {}, "monotone": True}
        prev = None
        for d in BUDGETS:
            n = sum(1 for r in rs if screen_fires(r.vectors, d))
            rec["fires"][f"d={d}"] = {"n": n, "frac": round(n / len(rs), 4)}
            if prev is not None and n > prev:
                rec["monotone"] = False   # firing must weaken as d grows
            prev = n
        out["families"][fam] = rec
        print(f"{fam:11s} n={len(rs):3d}  " + "  ".join(
            f"d={d}: {rec['fires'][f'd={d}']['n']:3d} "
            f"({rec['fires'][f'd={d}']['frac']:.3f})" for d in BUDGETS)
            + f"   monotone={rec['monotone']}")

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
