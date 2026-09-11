"""REGISTRATION_V8: is the conformal correction measurable at n <= 40?

Compares, on identical splits of the V7 corpus and with identical labels:
  A) conformal, k = floor(delta*(n+1)), threshold = k-th smallest gold score;
  B) plug-in constant, the empirical delta-quantile of the same gold scores.
Primary endpoint is how often each falls BELOW the nominal level on the
held-out rules of the split -- validity, not mean coverage.

Cache-only and deterministic; no LM calls.

Run:  PYTHONPATH=. python3 -m saorl.v8_calibration_vs_constant
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
V7 = json.load(open(ROOT / "results/e2e" / "v7_calibration.json"))
OUT = ROOT / "results/e2e" / "v8_calibration_vs_constant.json"

B = 4000
N_CALS = [10, 15, 20, 25, 30, 40]
DELTAS = [0.05, 0.10, 0.20]


def conformal_q(cal: np.ndarray, delta: float) -> float:
    n = len(cal)
    k = max(1, math.floor(delta * (n + 1)))
    return float(np.sort(cal)[k - 1])


def plugin_q(cal: np.ndarray, delta: float) -> float:
    return float(np.quantile(cal, delta, method="lower"))


def main() -> None:
    ids = sorted(V7["sigmas"])
    sig = np.array([V7["sigmas"][i] for i in ids], dtype=float)
    n_all = len(sig)
    # pool sizes per unit, for the efficiency endpoint
    cache = json.load(open(Path(__file__).parent / "plausibility_cache_v7.json"))
    pools = {u: np.array([float(np.mean(v)) for v in cache["units"][u]["scores"].values()])
             for u in ids}

    rng = np.random.default_rng(0)
    splits = {n: [rng.permutation(n_all) for _ in range(B)] for n in N_CALS}

    rows = {}
    for delta in DELTAS:
        for n_cal in N_CALS:
            acc = defaultdict(list)
            same = 0
            for perm in splits[n_cal]:
                ci, ti = perm[:n_cal], perm[n_cal:]
                cal, test = sig[ci], sig[ti]
                qa, qb = conformal_q(cal, delta), plugin_q(cal, delta)
                same += (qa == qb)
                for tag, q in (("conformal", qa), ("plugin", qb)):
                    acc[f"{tag}_cov"].append(float(np.mean(test >= q)))
                    acc[f"{tag}_size"].append(
                        float(np.mean([np.sum(pools[ids[j]] >= q) for j in ti])))
            nominal = 1.0 - delta
            r = {}
            for tag in ("conformal", "plugin"):
                cov = np.array(acc[f"{tag}_cov"])
                r[tag] = dict(
                    below_nominal=round(float(np.mean(cov < nominal - 1e-12)), 4),
                    mean_cov=round(float(cov.mean()), 4),
                    mean_size=round(float(np.mean(acc[f"{tag}_size"])), 3))
            r["same_threshold_frac"] = round(same / B, 4)
            r["nominal"] = nominal
            rows[f"delta={delta}_ncal={n_cal}"] = r
            print(f"delta={delta:<5} n={n_cal:<3} "
                  f"below-nominal  conformal {r['conformal']['below_nominal']:.3f} "
                  f"plugin {r['plugin']['below_nominal']:.3f}   "
                  f"mean cov {r['conformal']['mean_cov']:.3f}/"
                  f"{r['plugin']['mean_cov']:.3f}   "
                  f"size {r['conformal']['mean_size']:.2f}/"
                  f"{r['plugin']['mean_size']:.2f}   "
                  f"same-thresh {r['same_threshold_frac']:.2f}")

    # cluster-respecting variant: hold out whole parent groups (8 units)
    from saorl.conformal_v7 import _parent_group
    grp = {i: _parent_group(i) for i in ids}
    groups = sorted(set(grp.values()))
    gidx = {g: np.array([j for j, i in enumerate(ids) if grp[i] == g]) for g in groups}
    rng2 = np.random.default_rng(1)
    clus = {}
    for delta in DELTAS:
        below = {"conformal": [], "plugin": []}
        for _ in range(B):
            order = rng2.permutation(len(groups))
            cal_g = [groups[j] for j in order[:len(groups) // 2]]
            test_g = [groups[j] for j in order[len(groups) // 2:]]
            cal = sig[np.concatenate([gidx[g] for g in cal_g])]
            test = sig[np.concatenate([gidx[g] for g in test_g])]
            for tag, q in (("conformal", conformal_q(cal, delta)),
                           ("plugin", plugin_q(cal, delta))):
                below[tag].append(float(np.mean(test >= q)) < 1 - delta - 1e-12)
        clus[f"delta={delta}"] = {t: round(float(np.mean(v)), 4) for t, v in below.items()}
        print(f"[cluster-split, 4 groups cal / 4 test] delta={delta}: "
              f"below-nominal conformal {clus[f'delta={delta}']['conformal']:.3f} "
              f"plugin {clus[f'delta={delta}']['plugin']:.3f}")

    json.dump(dict(registration="REGISTRATION_V8", B=B, n=n_all,
                   rows=rows, cluster_split=clus), open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
