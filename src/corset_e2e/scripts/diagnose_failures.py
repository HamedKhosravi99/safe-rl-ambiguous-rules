"""Where does the vector arm fail to certify, and is it one task family?

The neural ledger reports a bimodal certification picture: at n=1000
episodes and w=2 maximal readings the Hoeffding half-width is 0.047
against a budget of 0.050, so a cell certifies only when its worst-reading
mean is at most 0.003. Roughly 58% of corset/vector_cql cells clear that;
the rest sit near 0.27. The aggregate mean describes neither mode.

The open question is whether the failing 42% is concentrated in one
environment family, which would make it a task-design finding rather than
a method failure, or spread evenly, which would not.

Run on the cluster, where the per-cell results live:

    python -m corset_e2e.scripts.diagnose_failures

It reads results/e2e/offline/*.json and needs nothing else.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OFF = os.path.join(ROOT, "results/e2e", "offline")


def main() -> None:
    rows = []
    for p in sorted(glob.glob(os.path.join(OFF, "*.json"))):
        try:
            rows.append(json.load(open(p)))
        except json.JSONDecodeError:
            pass
    if not rows:
        print(f"no per-cell results under {OFF}")
        sys.exit(1)

    arm = [r for r in rows if r["arm"] == "corset" and r.get("learner") == "vector_cql"]
    if not arm:
        print("no corset/vector_cql cells found")
        sys.exit(1)

    bud = arm[0]["budget"]
    half = arm[0]["cert"]["half_width"]
    print(f"corset/vector_cql: {len(arm)} cells, budget {bud}, half-width {half:.4f}")
    print(f"a cell certifies only if its worst-reading mean <= {bud - half:.4f}\n")

    by = collections.defaultdict(list)
    for r in arm:
        by[r.get("family", "?")].append(r)

    hdr = f"{'family':<16} {'n':>3} {'certified':>10} {'worst mean':>11} {'worst median':>13}"
    print(hdr); print("-" * len(hdr))
    for fam, rs in sorted(by.items()):
        w = [max(r["cert"]["cost"]) for r in rs]
        s = sum(1 for r in rs if r["cert"]["ships"])
        print(f"{fam:<16} {len(rs):>3} {s:>4}/{len(rs):<5} {st.mean(w):>11.4f} {st.median(w):>13.4f}")

    # the question the table exists to answer
    fracs = {f: sum(1 for r in rs if r["cert"]["ships"]) / len(rs) for f, rs in by.items()}
    spread = max(fracs.values()) - min(fracs.values())
    print(f"\ncertification rate by family: "
          + ", ".join(f"{f} {v:.2f}" for f, v in sorted(fracs.items())))
    if spread >= 0.5:
        print(f"spread {spread:.2f}: failures are CONCENTRATED -- a task-design finding,"
              f"\n  and the aggregate understates the method on the other families.")
    else:
        print(f"spread {spread:.2f}: failures are SPREAD across families -- this is the"
              f"\n  learner or the budget, not one bad task.")

    out = os.path.join(ROOT, "results/e2e", "failure_diagnosis.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(budget=bud, half_width=half, certify_threshold=bud - half,
                   by_family={f: dict(n=len(rs),
                                      certified=sum(1 for r in rs if r["cert"]["ships"]),
                                      worst_mean=st.mean([max(r["cert"]["cost"]) for r in rs]))
                              for f, rs in by.items()},
                   spread=spread), open(out, "w"), indent=1)
    print(f"\nwrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
