"""WP-I: the arm-by-arm certificate ledger.

Reduces the trained matrix to the paper's central neural claim. An arm that
defends one retained reading is measured on BOTH channels, so the table shows
what a policy tuned to a single interpretation of the cost does to the
interpretation it ignored.

Columns per arm, aggregated over configurations and seeds:
  ships       fraction of cells whose Hoeffding upper bound clears the budget
              on EVERY maximal reading (the shipping decision itself)
  own         mean normalised cost on the channel the arm enforces
  other       mean normalised cost on the channel it does not enforce
  viol        fraction of certification episodes exceeding the budget anywhere
  ret         mean return, i.e. what the defence costs in performance

`own`/`other` are only defined for the single-channel arms; the vector, union
and fallback arms enforce or ignore everything, so both entries are the
max over channels and are reported once.

Run: python -m corset_e2e.scripts.ledger [--json results/e2e/ledger.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OFF = os.path.join(ROOT, "results/e2e", "offline")
ARMS = ["fallback", "single_1", "single_2", "union", "corset"]
OWN = {"single_1": 0, "single_2": 1}


def load():
    recs = []
    for p in sorted(glob.glob(os.path.join(OFF, "*.json"))):
        try:
            recs.append(json.load(open(p)))
        except json.JSONDecodeError:
            print(f"  skipped unreadable {os.path.basename(p)}")
    return recs


def summarise(recs):
    by = defaultdict(list)
    for r in recs:
        by[(r["arm"], r.get("learner", "none"))].append(r)
    rows = []
    for (arm, learner), rs in sorted(by.items(),
                                     key=lambda kv: (ARMS.index(kv[0][0])
                                                     if kv[0][0] in ARMS else 9,
                                                     kv[0][1])):
        n = len(rs)
        ships = sum(bool(r["cert"]["ships"]) for r in rs) / n
        viol = sum(r["cert"]["episode_violation_rate"] for r in rs) / n
        ret = sum(r["cert"]["ret"] for r in rs) / n
        bud = sum(r["budget"] for r in rs) / n
        if arm in OWN:
            i = OWN[arm]
            own = sum(r["cert"]["cost"][i] for r in rs) / n
            oth = sum(max(c for j, c in enumerate(r["cert"]["cost"]) if j != i)
                      for r in rs) / n
        else:
            own = oth = sum(max(r["cert"]["cost"]) for r in rs) / n
        rows.append(dict(arm=arm, learner=learner, cells=n, ships=ships,
                         own=own, other=oth, budget=bud, viol=viol, ret=ret))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(ROOT, "results/e2e", "ledger.json"))
    a = ap.parse_args()

    recs = load()
    if not recs:
        print("no results yet")
        return
    rows = summarise(recs)
    print(f"{len(recs)} cells over "
          f"{len({r['index'] for r in recs})} configurations, "
          f"{len({r['seed'] for r in recs})} seeds\n")
    hdr = f"{'arm':<9} {'learner':<11} {'n':>4} {'ships':>6} {'own':>7} {'other':>7} {'budget':>7} {'viol':>6} {'ret':>9}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:<9} {r['learner']:<11} {r['cells']:>4} "
              f"{r['ships']:>6.2f} {r['own']:>7.4f} {r['other']:>7.4f} "
              f"{r['budget']:>7.3f} {r['viol']:>6.2f} {r['ret']:>9.2f}")

    # the claim, stated as a comparison rather than asserted
    sing = [r for r in rows if r["arm"] in OWN]
    vec = [r for r in rows if r["arm"] == "corset"]
    if sing and vec:
        print("\nsingle-reading arms: own %.4f vs unenforced %.4f (budget %.3f)"
              % (sum(r["own"] for r in sing) / len(sing),
                 sum(r["other"] for r in sing) / len(sing),
                 sing[0]["budget"]))
        print("vector arm:          worst channel %.4f, ships %.2f"
              % (sum(r["own"] for r in vec) / len(vec),
                 sum(r["ships"] for r in vec) / len(vec)))
    json.dump(dict(n_cells=len(recs), rows=rows), open(a.json, "w"), indent=1)
    print(f"\nwrote {os.path.relpath(a.json, ROOT)}")


if __name__ == "__main__":
    main()
