"""REGISTRATION_V9: the coverage claim on 138 independent real rules.

Reads the frozen source-grounded test pass, keeps its IN_DSL rows -- real
alerts from eight public repositories, mechanically compiled golds, frozen
deterministic score -- and runs the delta sweep and the conformal-vs-plugin
comparison on a corpus whose score is fine enough for the threshold to move.

No LM calls; nothing re-scored.

Run:  PYTHONPATH=. python3 -m saorl.v9_independent_corpus
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
ROWS = json.load(open(ROOT / "results/e2e" / "e2e_test_rows_v3_complete.json"))
OUT = ROOT / "results/e2e" / "v9_independent_corpus.json"

B = 4000
N_CALS = [20, 30, 40, 60, 80]
DELTAS = [0.01, 0.05, 0.10, 0.20]


def qhat(cal: np.ndarray, delta: float) -> float:
    n = len(cal)
    k = max(1, math.floor(delta * (n + 1)))
    return float(np.sort(cal)[k - 1])


def plugin(cal: np.ndarray, delta: float) -> float:
    return float(np.quantile(cal, delta, method="lower"))


def main() -> None:
    rows = [r for r in ROWS if r.get("label") == "IN_DSL"
            and r.get("gold_score") is not None]
    s = np.array([r["gold_score"] for r in rows], dtype=float)
    repos = np.array([r["repo"] for r in rows])
    uniq = sorted(set(repos))
    print(f"corpus: {len(rows)} IN_DSL rows, {len(set(s))} distinct scores, "
          f"{len(uniq)} repos")
    for rp in uniq:
        print(f"   {rp:34s} {int((repos == rp).sum())}")

    rng = np.random.default_rng(11)
    out = {"n": len(rows), "distinct_scores": len(set(s)),
           "repos": {rp: int((repos == rp).sum()) for rp in uniq},
           "row_splits": {}, "repo_splits": {}, "plugin": {}}

    # ---- E-1/E-2/E-4: row splits ------------------------------------------
    for delta in DELTAS:
        for n_cal in N_CALS:
            qs, covs, pq, pcov, diff = [], [], [], [], 0
            for _ in range(B):
                p = rng.permutation(len(s))
                cal, test = s[p[:n_cal]], s[p[n_cal:]]
                q, qp = qhat(cal, delta), plugin(cal, delta)
                qs.append(q); pq.append(qp); diff += (q != qp)
                covs.append(float(np.mean(test >= q)))
                pcov.append(float(np.mean(test >= qp)))
            out["row_splits"][f"d={delta}_n={n_cal}"] = dict(
                mean_qhat=round(float(np.mean(qs)), 4),
                mean_cov=round(float(np.mean(covs)), 4),
                nominal=round(1 - delta, 2),
                clears=bool(np.mean(covs) >= 1 - delta))
            out["plugin"][f"d={delta}_n={n_cal}"] = dict(
                differ_frac=round(diff / B, 4),
                mean_cov=round(float(np.mean(pcov)), 4),
                clears=bool(np.mean(pcov) >= 1 - delta))
        r = out["row_splits"][f"d={delta}_n=60"]
        p_ = out["plugin"][f"d={delta}_n=60"]
        print(f"[rows n=60] delta={delta:<5} qhat={r['mean_qhat']:.4f}  "
              f"cov={r['mean_cov']:.4f} vs nominal {r['nominal']:.2f} "
              f"{'OK' if r['clears'] else 'BELOW'}   |  plugin cov "
              f"{p_['mean_cov']:.4f} {'OK' if p_['clears'] else 'BELOW'}, "
              f"differs {p_['differ_frac']:.2f}")

    # E-1: do the four levels separate?
    qs_by_delta = {d: out["row_splits"][f"d={d}_n=60"]["mean_qhat"] for d in DELTAS}
    out["distinct_qhat_levels"] = len(set(round(v, 4) for v in qs_by_delta.values()))
    print(f"\nE-1: distinct mean thresholds across the four levels = "
          f"{out['distinct_qhat_levels']}/4   {qs_by_delta}")

    # ---- E-3: repo (cluster) splits ---------------------------------------
    rng2 = np.random.default_rng(12)
    idx = {rp: np.where(repos == rp)[0] for rp in uniq}
    for delta in DELTAS:
        covs = []
        for _ in range(B):
            order = rng2.permutation(len(uniq))
            cal_r = [uniq[j] for j in order[:len(uniq) // 2]]
            test_r = [uniq[j] for j in order[len(uniq) // 2:]]
            cal = s[np.concatenate([idx[r] for r in cal_r])]
            test = s[np.concatenate([idx[r] for r in test_r])]
            if len(cal) < 5:
                continue
            covs.append(float(np.mean(test >= qhat(cal, delta))))
        m = float(np.mean(covs))
        out["repo_splits"][f"d={delta}"] = dict(
            mean_cov=round(m, 4), nominal=round(1 - delta, 2),
            clears=bool(m >= 1 - delta),
            gap_vs_rows=round(m - out["row_splits"][f"d={delta}_n=60"]["mean_cov"], 4))
        print(f"[repos 4v4] delta={delta:<5} cov={m:.4f} vs nominal "
              f"{1-delta:.2f} {'OK' if m >= 1-delta else 'BELOW'}  "
              f"(gap vs row splits {out['repo_splits'][f'd={delta}']['gap_vs_rows']:+.4f})")

    json.dump(dict(registration="REGISTRATION_V9", B=B, **out),
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
