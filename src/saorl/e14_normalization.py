"""E14 (REGISTRATION_V10): does the screen rate depend on the normalization?

The published free-class screen asks whether the psi_k-optimal admitted
SET keeps every reading's mean cost over that set at or below d. An
occupancy model of admission control asks something different: whether
expected cost per ARRIVAL stays under d, with no division by the admit
rate. One is a compliance rate among admitted resources, the other a
violation budget per request.

Same pools, same readings, same fixtures, same predicate. Only the
normalization moves.

Run:  PYTHONPATH=. python3 -m saorl.e14_normalization
"""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import List, Sequence

from saorl.benchmark_sg import run_benchmark as rb
from saorl.benchmark_sg.e4_screen import BUDGETS, screen_fires
from saorl.benchmark_sg.parse import (parse_kyverno, parse_prometheus,
                                      prom_threshold_bank)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/e2e" / "e14_normalization.json"


def best_selection_per_request(v: Sequence[float], d: float,
                               n_total: int) -> List[int]:
    """Largest S with TOTAL v-cost / n_total <= d (per-arrival budget).

    Greedy on ascending cost is still optimal: for a fixed |S| the
    cheapest |S| fixtures minimise the total, and the budget does not
    depend on |S|, so feasible sizes remain prefix-closed."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    best, run = [], 0.0
    for n, i in enumerate(order, start=1):
        run += v[i]
        if run / n_total <= d + 1e-12:
            best = order[:n]
    return best


def suffices_pr(vectors, k: int, d: float) -> bool:
    n = len(vectors[k])
    S = best_selection_per_request(vectors[k], d, n)
    if not S:
        return False
    return all(sum(w[i] for i in S) / n <= d + 1e-12 for w in vectors)


def screen_fires_pr(vectors, d: float) -> bool:
    if len(vectors) < 2:
        return False
    return not any(suffices_pr(vectors, k, d) for k in range(len(vectors)))


def main() -> None:
    pt, _ = parse_prometheus(); kt, _ = parse_kyverno()
    bank = prom_threshold_bank(pt)
    recs = [rb._record(rb.build_prom_pool(t, bank), t, rb._prom_skeleton(t))
            for t in pt]
    recs += [rb._record(rb.build_kyv_pool(t), t, rb._kyv_skeleton(t))
             for t in kt]
    fam = {}
    for r in recs:
        fam.setdefault(r.family, []).append(r)

    out = {"registration": "REGISTRATION_V10 E14", "budgets": list(BUDGETS),
           "families": {}}
    for f, rs in sorted(fam.items()):
        rec = {"n_pools": len(rs), "per_admitted": {}, "per_request": {}}
        for d in BUDGETS:
            a = sum(1 for r in rs if screen_fires(r.vectors, d))
            b = sum(1 for r in rs if screen_fires_pr(r.vectors, d))
            rec["per_admitted"][f"d={d}"] = round(a / len(rs), 4)
            rec["per_request"][f"d={d}"] = round(b / len(rs), 4)
        out["families"][f] = rec
        print(f"{f:11s} n={len(rs):3d}")
        for d in BUDGETS:
            print(f"   d={d:<5} per-admitted {rec['per_admitted'][f'd={d}']:.4f}"
                  f"   per-request {rec['per_request'][f'd={d}']:.4f}")
    os.makedirs(OUT.parent, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
