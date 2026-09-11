"""A9 aggregation: vector-cost non-nested suite (registered taxonomy).

Applies the cell taxonomy frozen in Addendum A9 verbatim:
  clean ambiguity      both singletons satisfy their OWN reading, and each
                       violates the other retained reading
  one-sided capacity   exactly one singleton violates its own reading
  global capacity      both singletons violate their own readings
  set defense success  the vector arm satisfies BOTH retained readings
  set-learner failure  the vector arm violates at least one

Registered success threshold: >= 3 clean ambiguity cells (V6.3 governs a
shortfall). Per-seed Theorem-3 certificates are reported as pass rates,
never pooled and never best-seed-selected. Cells with no results are
disclosed by name.

Run: PYTHONPATH=. python3 experiments/dsrl_learners/aggregate_vector.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "results/dsrl", "vector_summary.json")
CLEAN_THRESHOLD = 3
BUDGET = 1.0          # normalized: cost_*_norm <= 1 means within budget
ARMS = ("single_a", "single_b", "union", "vector")

# the 9 registered cells (index -> label), from experiments/dsrl_learners/dsrl_vector.py CELLS
CELLS = ["CarRun/cpq/hi_sus", "CarCircle/cpq/hi_sus", "BallCircle/bcql/hi_sus",
         "BallRun/bcql/hi_sus", "BallRun/cpq/hi_sus", "CarRun/bcql/hi_sus",
         "CarCircle/bcql/hi_sus", "BallCircle/cpq/hi_sus",
         "BallCircle/bcql/hi_scope"]


def classify(m):
    """m[arm][channel] -> mean normalized cost. Returns taxonomy labels."""
    a_own = m["single_a"]["a"] <= BUDGET      # A satisfies its own reading
    b_own = m["single_b"]["b"] <= BUDGET
    a_hurts_b = m["single_a"]["b"] > BUDGET   # A violates the other
    b_hurts_a = m["single_b"]["a"] > BUDGET
    vec_ok = m["vector"]["a"] <= BUDGET and m["vector"]["b"] <= BUDGET
    uni_ok = m["union"]["a"] <= BUDGET and m["union"]["b"] <= BUDGET

    if a_own and b_own:
        cap = "clean ambiguity" if (a_hurts_b and b_hurts_a) else \
              "no crossing (both singletons feasible for both)"
    elif a_own or b_own:
        cap = "one-sided capacity failure"
    else:
        cap = "global capacity failure"
    return dict(capacity=cap,
                clean=(cap == "clean ambiguity"),
                set_defense=("success" if vec_ok else "set-learner failure"),
                union_defense=("success" if uni_ok else "failure"))


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "results/dsrl", "vector",
                                          "*", "vector_*.json")))
    by_cell = defaultdict(list)
    for f in files:
        d = json.load(open(f))
        if "11615686" in os.path.basename(f):
            continue                       # smoke run, excluded per A9
        key = f"{d['task']}/{d['learner']}/{d['crossing']}"
        by_cell[key].append(d)

    rows, clean_n = [], 0
    for cell, runs in sorted(by_cell.items()):
        seeds = sorted(r["seed"] for r in runs)
        m = {}
        for arm in ARMS:
            m[arm] = {
                ch: float(np.mean([r["arms"][arm]["agg"][f"cost_{ch}_norm"]
                                   for r in runs]))
                for ch in ("a", "b")}
            m[arm]["ret"] = float(np.mean(
                [r["arms"][arm]["agg"]["ret_norm"] for r in runs]))
        tax = classify(m)
        clean_n += tax["clean"]
        # per-seed Theorem-3 certificate pass rates (no pooling)
        certs = {}
        for arm in ARMS:
            ships = [(r["arms"][arm]["certs"]["a"]["ships"] and
                      r["arms"][arm]["certs"]["b"]["ships"]) for r in runs]
            certs[arm] = dict(pass_rate=round(float(np.mean(ships)), 3),
                              n_seeds=len(ships))
        price = None
        if m["vector"]["ret"] and m["single_a"]["ret"]:
            best_self = max(m["single_a"]["ret"], m["single_b"]["ret"])
            if best_self:
                price = round(1 - m["vector"]["ret"] / best_self, 4)
        rows.append(dict(cell=cell, n_seeds=len(runs), seeds=seeds,
                         costs={a: {k: round(v, 3) for k, v in m[a].items()}
                                for a in ARMS},
                         taxonomy=tax, cert_pass=certs,
                         vector_return_price=price))

    def _seen(c):
        task, learner, cross = c.split("/")
        return any(task in r["cell"] and f"/{learner}/" in r["cell"]
                   and r["cell"].endswith(cross) for r in rows)
    missing = [c for c in CELLS if not _seen(c)]
    rep = dict(
        manifest="Addendum A9 @ 2cb16c9 (+ execution note @ f79281f)",
        n_cells_reported=len(rows), n_cells_registered=len(CELLS),
        missing_cells=missing,
        clean_ambiguity_cells=clean_n,
        threshold=CLEAN_THRESHOLD,
        threshold_met=clean_n >= CLEAN_THRESHOLD,
        v6_3_branch=("not triggered" if clean_n >= CLEAN_THRESHOLD else
                     "TRIGGERED: ambiguity claim stays scoped to tabular "
                     "and source-grounded evidence"),
        set_defense_successes=sum(
            r["taxonomy"]["set_defense"] == "success" for r in rows),
        union_defense_successes=sum(
            r["taxonomy"]["union_defense"] == "success" for r in rows),
        cells=rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rep, open(OUT, "w"), indent=1)

    print(f"cells reported: {len(rows)}/{len(CELLS)}"
          + (f"  MISSING: {missing}" if missing else ""))
    for r in rows:
        t = r["taxonomy"]
        print(f"  {r['cell']:34s} n={r['n_seeds']} {t['capacity']:34s} "
              f"set={t['set_defense']:20s} union={t['union_defense']}")
    print(f"clean ambiguity cells: {clean_n} (threshold {CLEAN_THRESHOLD}) "
          f"-> {'MET' if clean_n >= CLEAN_THRESHOLD else 'SHORTFALL (V6.3)'}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
