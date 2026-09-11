"""E16 (REGISTRATION_V10): is the published screen predicate argmax-sensitive?

E15 raised the question and my first attempt to answer it was invalid: I
relaxed the ratio constraint to x in [0,1] and 229 of 278 LPs came back
fractional, so the relaxation reached selections that do not exist. The
fixture universes are small (|F| <= 18), so the question can be settled
by EXACT enumeration instead, with no relaxation anywhere.

Published predicate (e4_screen.suffices): reading k suffices at d if THE
psi_k-optimal selection -- the greedy ascending-cost prefix -- also
satisfies every other reading. When several selections attain the
psi_k-optimal size, this inspects one of them.

Achievability predicate: reading k suffices at d if the psi_k-optimal
SIZE is still attainable by some selection honouring every reading. This
does not depend on which optimizer is returned.

Both computed exactly here. The screen fires iff no reading suffices.

Run:  PYTHONPATH=. python3 -m saorl.e16_exact_screen
"""
from __future__ import annotations
import json, os
from pathlib import Path

import numpy as np

from saorl.benchmark_sg import run_benchmark as rb
from saorl.benchmark_sg.e4_screen import BUDGETS, screen_fires
from saorl.benchmark_sg.parse import parse_kyverno, parse_prometheus, prom_threshold_bank

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/e2e" / "e16_exact_screen.json"


def masks(n: int) -> np.ndarray:
    """(2^n, n) boolean membership matrix over all subsets."""
    idx = np.arange(1 << n, dtype=np.uint32)
    return ((idx[:, None] >> np.arange(n)) & 1).astype(bool)


def exact_sizes(vectors, d: float):
    """(V_k for each k, V_all) as exact maximum selection sizes."""
    V = np.asarray(vectors, dtype=float)          # (M, n)
    M, n = V.shape
    B = masks(n)                                  # (2^n, n)
    size = B.sum(1)                               # (2^n,)
    tot = B.astype(float) @ V.T                   # (2^n, M) total cost
    ok = tot <= d * size[:, None] + 1e-12         # per-reading feasibility
    ok[0, :] = False                              # empty selection is not a policy
    Vk = [int(size[ok[:, k]].max()) if ok[:, k].any() else 0 for k in range(M)]
    all_ok = ok.all(axis=1)
    V_all = int(size[all_ok].max()) if all_ok.any() else 0
    return Vk, V_all


def fires_exact(vectors, d: float) -> bool:
    if len(vectors) < 2:
        return False
    Vk, V_all = exact_sizes(vectors, d)
    return not any(V_all >= v for v in Vk)        # no reading suffices


def main() -> None:
    kt, _ = parse_kyverno()
    recs = [(("kyverno",), rb._record(rb.build_kyv_pool(t), t, rb._kyv_skeleton(t)))
            for t in kt]
    pt, _ = parse_prometheus(); bank = prom_threshold_bank(pt)
    prom = [rb._record(rb.build_prom_pool(t, bank), t, rb._prom_skeleton(t))
            for t in pt]
    fams = {"kyverno": [r for _, r in recs],
            "prometheus": [r for r in prom if len(r.vectors[0]) <= 20]}

    out = {"registration": "REGISTRATION_V10 E16", "budgets": list(BUDGETS),
           "families": {}}
    for fam, rs in sorted(fams.items()):
        rec = {"n_pools": len(rs), "published_greedy": {}, "exact_achievable": {},
               "disagreements": {}}
        for d in BUDGETS:
            g = [screen_fires(r.vectors, d) for r in rs]
            e = [fires_exact(r.vectors, d) for r in rs]
            rec["published_greedy"][f"d={d}"] = round(sum(g) / len(rs), 4)
            rec["exact_achievable"][f"d={d}"] = round(sum(e) / len(rs), 4)
            rec["disagreements"][f"d={d}"] = int(sum(a != b for a, b in zip(g, e)))
            print(f"{fam:11s} d={d:<5} published {sum(g):3d}/{len(rs)} "
                  f"({sum(g)/len(rs):.4f})   exact {sum(e):3d}/{len(rs)} "
                  f"({sum(e)/len(rs):.4f})   disagree={rec['disagreements'][f'd={d}']}")
        out["families"][fam] = rec
    tot = sum(v for f in out["families"].values() for v in f["disagreements"].values())
    out["total_disagreements"] = tot
    out["verdict"] = ("published predicate is argmax-sensitive"
                      if tot else
                      "published predicate agrees with exact achievability everywhere")
    os.makedirs(OUT.parent, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nverdict:", out["verdict"])
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
