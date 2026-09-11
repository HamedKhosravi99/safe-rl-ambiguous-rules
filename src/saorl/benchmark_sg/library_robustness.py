"""Is the non-dominance rate a property of the rules or of our frozen grids?

Reviewers objected that 78-96% non-dominance may measure the authors'
transformation library rather than the artifacts. The library's grids are
frozen constants, so we can re-derive the rate under perturbed grids and see
whether the finding moves. Reported either way.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from saorl.benchmark_sg import candidates as C  # noqa: E402

OUT = os.path.join("results/conformal", "benchmark_sg", "library_robustness.json")

VARIANTS = {
    "frozen (as published)": dict(),
    "coarse duration grid": dict(DURATION_GRID_S=[0.0, 300.0, 900.0]),
    "fine duration grid": dict(
        DURATION_GRID_S=[0.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1200.0, 1800.0]),
    "wider threshold factors": dict(_FACTORS=(0.75, 1.33)),
    "narrow threshold factors": dict(_FACTORS=(0.97, 1.03)),
}


def nondominated_frac(recs):
    """Fraction of pools with no pointwise-dominating member."""
    n = ndom = 0
    for r in recs:
        V = r.vectors
        if not V or len(V) < 2:
            continue
        n += 1
        dominating = False
        for i, vi in enumerate(V):
            if all(all(a >= b for a, b in zip(vi, vj)) for j, vj in enumerate(V)):
                dominating = True
                break
        if not dominating:
            ndom += 1
    return (ndom / n if n else float("nan")), n


def main() -> None:
    from saorl.benchmark_sg.calib_vs_fixed import build_records
    orig_dur = list(C.DURATION_GRID_S)
    orig_fn = C._threshold_neighbours
    out = []
    hdr = f"{'library variant':<26} {'family':<11} {'pools':>6} {'non-dominated':>14}"
    print(hdr); print("-" * len(hdr))
    for name, cfg in VARIANTS.items():
        if "DURATION_GRID_S" in cfg:
            C.DURATION_GRID_S[:] = cfg["DURATION_GRID_S"]
        else:
            C.DURATION_GRID_S[:] = orig_dur
        if "_FACTORS" in cfg:
            f = cfg["_FACTORS"]
            def patched(t, bank, _f=f, _orig=orig_fn):
                below = [v for v in bank if v < t]; above = [v for v in bank if v > t]
                res = []
                if below: res.append(max(below))
                if above: res.append(min(above))
                for fac in _f:
                    v = round(t * fac, 6)
                    if v != t: res.append(v)
                seen, keep = set(), []
                for v in res:
                    if v not in seen and v != t:
                        seen.add(v); keep.append(v)
                return keep[:4]
            C._threshold_neighbours = patched
        else:
            C._threshold_neighbours = orig_fn
        recs = build_records()
        for fam, rs in recs.items():
            frac, n = nondominated_frac(rs)
            out.append(dict(variant=name, family=fam, n_pools=n, frac_non_dominated=frac))
            print(f"{name:<26} {fam:<11} {n:>6} {frac:>14.3f}")
    C.DURATION_GRID_S[:] = orig_dur
    C._threshold_neighbours = orig_fn
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
