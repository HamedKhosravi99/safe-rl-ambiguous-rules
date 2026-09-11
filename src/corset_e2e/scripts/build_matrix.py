"""Build the confirmatory job matrix from the accepted-task ledger.

The primary claim rests on the vector-cost arm; portability is tested with two
further optimisation mechanisms (plan G.3). Every accepted task also runs both
singletons, the union surrogate, and the fallback, so failures and abstentions
stay in the denominator.
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PRE = os.path.join(ROOT, "results/e2e", "prescreen")
SEEDS = (0, 1, 2, 3, 4)
PRIMARY_ARMS = ("single_1", "single_2", "corset", "union")
PORT_LEARNERS = ("vector_fqi", "vector_bc")


def accepted():
    out = []
    for f in sorted(glob.glob(os.path.join(PRE, "cfg_*.json"))):
        r = json.load(open(f))
        if r.get("accepted"):
            out.append(r["index"])
    return out


def matrix(size: int = 300_000):
    jobs = []
    for idx in accepted():
        for arm in PRIMARY_ARMS:
            for s in SEEDS:
                jobs.append((idx, arm, "vector_cql", size, s))
        for ln in PORT_LEARNERS:
            for s in SEEDS:
                jobs.append((idx, "corset", ln, size, s))
        jobs.append((idx, "fallback", "none", size, 0))
    return jobs


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--count":
        print(len(matrix()))
    elif len(sys.argv) > 2 and sys.argv[1] == "--job":
        i, a, l, n, s = matrix()[int(sys.argv[2])]
        print(f"{i} {a} {l} {n} {s}")
    elif len(sys.argv) > 1 and sys.argv[1] == "--accepted":
        print(" ".join(str(i) for i in accepted()))
    else:
        acc = accepted()
        print(json.dumps(dict(n_accepted=len(acc), accepted=acc,
                              n_jobs=len(matrix())), indent=1))
