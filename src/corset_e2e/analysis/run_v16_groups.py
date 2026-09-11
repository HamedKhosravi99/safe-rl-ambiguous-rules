"""V16 stages A-C: unary containment, joint set retrieval, R_group@N.

Everything OOF on dev per REGISTRATION_V16.md. Writes
results/e2e/v16_groups_report.json (tables) and results/e2e/v16_pools.json
(per-unit top-50 reranked groups, consumed by the composition stage).

Run: PYTHONPATH=. python3 corset_e2e/analysis/run_v16_groups.py [limit]
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(ROOT, "results/e2e")
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.generator.joint_metric_sets import (  # noqa: E402
    FOLDS, K_GRID, fit_pair, fit_qprior, fit_reranker, fit_unary,
    load_units, search_groups)


def weights_of(units, hit_rids):
    n = len(units)
    raw = sum(1 for u in units if u["rid"] in hit_rids) / n
    seen, dn, dh = set(), 0, 0
    per = defaultdict(lambda: [0, 0])
    for u in units:
        if u["textkey"] not in seen:
            seen.add(u["textkey"])
            dn += 1
            dh += u["rid"] in hit_rids
        c = per[u["repo"]]
        c[0] += 1
        c[1] += u["rid"] in hit_rids
    macro = sum(c[1] / c[0] for c in per.values()) / len(per)
    return dict(raw=raw, distinct=dh / dn, repo_macro=macro)


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"[load] dev units (limit={limit or 'all'})")
    units = load_units("dev", limit=limit)
    print(f"    {len(units)} units")

    print("[A] unary per fold + containment")
    wu = {f: fit_unary(units, f) for f in range(FOLDS)}
    contain = {K: set() for K in K_GRID}
    for u in units:
        U = u["X"] @ wu[u["fold"]]
        order = np.argsort(-U)
        names = [u["names"][i] for i in order[:max(K_GRID)]]
        for K in K_GRID:
            if u["M"] <= set(names[:K]):
                contain[K].add(u["rid"])
    best = max(len(v) for v in contain.values())
    tbl = {}
    for K in K_GRID:
        tbl[K] = weights_of(units, contain[K])
        print(f"    K={K:3d}  raw {tbl[K]['raw']:6.1%}  "
              f"distinct {tbl[K]['distinct']:6.1%}  "
              f"repo-macro {tbl[K]['repo_macro']:6.1%}")
    K_STAR = min(K for K in K_GRID
                 if len(contain[K]) >= best - 0.01 * len(units))
    print(f"    frozen K* = {K_STAR} (smallest within 1pt of best)")

    print("[B] pairwise + q-prior per fold")
    wp = {f: fit_pair(units, f) for f in range(FOLDS)}
    Wq = {f: fit_qprior(units, f) for f in range(FOLDS)}

    print("[C] set search")
    pools = {}
    for i, u in enumerate(units):
        f = u["fold"]
        pool, _U = search_groups(u, wu[f], wp[f], Wq[f], K_STAR)
        pools[u["rid"]] = pool
        if (i + 1) % 200 == 0:
            print(f"    searched {i + 1}/{len(units)}")

    print("[C] rerank per fold")
    wr = {f: fit_reranker(pools, units, f) for f in range(FOLDS)}
    hits_at = {N: set() for N in (1, 5, 10, 20, 50)}
    top50 = {}
    for u in units:
        pool = pools[u["rid"]]
        w = wr[u["fold"]]
        scored = sorted(((float(phi @ w), k) for k, _s0, phi in pool),
                        key=lambda z: -z[0])
        gold = frozenset(u["M"])
        rank = next((i for i, (_s, k) in enumerate(scored) if k == gold),
                    None)
        for N in hits_at:
            if rank is not None and rank < N:
                hits_at[N].add(u["rid"])
        top50[u["rid"]] = [[sorted(k), s] for s, k in scored[:50]]

    cv = json.load(open(os.path.join(OUT, "vocab_ceiling_dev.json")))
    rep = dict(n=len(units), containment={str(k): v for k, v in tbl.items()},
               K_star=K_STAR, recall={}, efficiency={})
    print(f"\n{'stage':<26s} {'raw':>8s} {'distinct':>10s} {'repo-macro':>11s}")
    print(f"{'gold set in vocabulary':<26s} {cv['C_V_raw']:8.1%} "
          f"{cv['C_V_distinct']:10.1%} {cv['C_V_repo_macro']:11.1%}")
    ck = tbl[K_STAR]
    print(f"{'inside unary top-' + str(K_STAR):<26s} {ck['raw']:8.1%} "
          f"{ck['distinct']:10.1%} {ck['repo_macro']:11.1%}")
    for N in (1, 5, 10, 20, 50):
        wN = weights_of(units, hits_at[N])
        rep["recall"][str(N)] = wN
        print(f"{'group recall@' + str(N):<26s} {wN['raw']:8.1%} "
              f"{wN['distinct']:10.1%} {wN['repo_macro']:11.1%}")
    eff = rep["recall"]["50"]["distinct"] / cv["C_V_distinct"]
    rep["efficiency"] = dict(
        distinct=eff,
        raw=rep["recall"]["50"]["raw"] / cv["C_V_raw"])
    print(f"\nretrieval efficiency (distinct R@50 / C_V) = {eff:.3f}  "
          f"(gate >= 0.85)")

    json.dump(rep, open(os.path.join(OUT, "v16_groups_report.json"), "w"),
              indent=1)
    json.dump(top50, open(os.path.join(OUT, "v16_pools.json"), "w"))
    print("wrote v16_groups_report.json, v16_pools.json")


if __name__ == "__main__":
    main()
