"""V40 (Phase E): the three regimes along the capability ladder, at every
budget.

V14's face ladder computed, per (pool, rung k, predicate subset A), the
three-way verdict at the operating budget only:
    value screen fires              -> irreducible
    value clears, face sufficient   -> irrelevant
    value clears, face not suff.    -> optimizer-resolvable
This module runs the identical machinery (`_face_need`, `_face_need_fixture`
from face_ladder, `_grouped` from class_ladder) at all five budgets and
tabulates, per family, rung and budget, how many pools sit in each regime,
plus each pool's transition path as capability grows.  Registration V40.

Run: PYTHONPATH=. python3 -m saorl.benchmark_sg.regime_ladder
Writes results/e2e/regime_ladder.json
"""
from __future__ import annotations

import itertools
import json
import os
import time
from collections import Counter
from typing import Dict, List

import numpy as np

from .class_ladder import _grouped
from .face_ladder import _face_need, _face_need_fixture, SUBSET_CAP
from .parse import parse_prometheus, parse_kyverno, prom_threshold_bank
from .evaluate import build_prom_pool, build_kyv_pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "results/e2e", "regime_ladder.json")
BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10)
REGIMES = ("irrelevant", "optimizer_resolvable", "irreducible")
ORDER = {r: i for i, r in enumerate(REGIMES)}


def verdict(cell: dict) -> str:
    if cell["value_fires"]:
        return "irreducible"
    return "irrelevant" if not cell["face_need"] else "optimizer_resolvable"


def main() -> None:
    pt, _ = parse_prometheus()
    kt, _ = parse_kyverno()
    bank = prom_threshold_bank(pt)
    pools_out: List[dict] = []
    t0 = time.time()
    for fam, targets, builder in (("prometheus", pt, lambda t: build_prom_pool(t, bank)),
                                  ("kyverno", kt, build_kyv_pool)):
        for ti, t in enumerate(targets):
            pool = builder(t)
            uid = f'{getattr(t, "name", getattr(t, "policy_name", "?"))}#{ti}'
            vecs = np.asarray([np.asarray(c.vector, float) for c in pool.classes])
            K, F = vecs.shape
            ks = sorted({0, 1, 2} & set(range(K + 1)))
            rec = dict(uid=uid, family=fam, K=K, F=int(F), budgets={})
            for d in BUDGETS:
                rungs: Dict[str, dict] = {}
                for k in ks:
                    cnt = Counter()
                    for A in itertools.islice(itertools.combinations(range(K), k), SUBSET_CAP):
                        n_g, m = _grouped(vecs, A)
                        cnt[verdict(_face_need(n_g, m, K, d))] += 1
                    n_sub = sum(cnt.values())
                    modal = max(REGIMES, key=lambda r: (cnt[r], -ORDER[r]))
                    rungs[str(k)] = dict(n_subsets=n_sub, counts=dict(cnt), modal=modal,
                                         some_resolvable=bool(cnt["optimizer_resolvable"] > 0),
                                         all_irreducible=bool(cnt["irreducible"] == n_sub and n_sub > 0))
                fr = verdict(_face_need_fixture(vecs, d))
                rungs[str(K)] = dict(n_subsets=1, counts={fr: 1}, modal=fr,
                                     some_resolvable=(fr == "optimizer_resolvable"),
                                     all_irreducible=(fr == "irreducible"))
                path = [rungs[str(k)]["modal"] for k in ks + ([K] if K not in ks else [])]
                idx = [ORDER[p] for p in path]
                rec["budgets"][str(d)] = dict(rungs=rungs, path=path,
                                              monotone_forward=bool(all(a <= b for a, b in zip(idx, idx[1:]))),
                                              first_resolvable=next((k for k, p in zip(ks + ([K] if K not in ks else []), path) if p == "optimizer_resolvable"), None),
                                              first_irreducible=next((k for k, p in zip(ks + ([K] if K not in ks else []), path) if p == "irreducible"), None))
            pools_out.append(rec)
            print(f"  [{fam}] {uid:44s} K={K} F={F}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- aggregate: rung x budget x regime, per family ---------------------
    agg = {}
    for fam in ("prometheus", "kyverno"):
        P = [p for p in pools_out if p["family"] == fam]
        table = {}
        for d in BUDGETS:
            table[str(d)] = {}
            for rung in ("0", "1", "2", "free"):
                cnt = Counter(); some_res = 0; n = 0
                for p in P:
                    R = p["budgets"][str(d)]["rungs"]
                    key = str(p["K"]) if rung == "free" else rung
                    if key not in R:
                        continue
                    if rung != "free" and int(rung) >= p["K"]:
                        continue                      # rung coincides with free; count once
                    cnt[R[key]["modal"]] += 1; some_res += R[key]["some_resolvable"]; n += 1
                table[str(d)][rung] = dict(n=n, **{r: cnt[r] for r in REGIMES}, some_resolvable=some_res)
            paths = [p["budgets"][str(d)] for p in P]
            table[str(d)]["paths"] = dict(
                monotone_forward=sum(x["monotone_forward"] for x in paths),
                ever_resolvable=sum(x["first_resolvable"] is not None for x in paths),
                ever_irreducible=sum(x["first_irreducible"] is not None for x in paths),
                resolvable_before_irreducible=sum(1 for x in paths if x["first_resolvable"] is not None and
                                                  (x["first_irreducible"] is None or x["first_resolvable"] < x["first_irreducible"])),
                n=len(paths))
        agg[fam] = table

    payload = dict(registration="V40 (REGISTRATION_V28.md): three regimes along the capability ladder at all budgets; "
                                f"subset cap {SUBSET_CAP}; free rung at fixture level", budgets=list(BUDGETS),
                   subset_cap=SUBSET_CAP, runtime_s=time.time() - t0, aggregate=agg, pools=pools_out)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"wrote {OUT} in {time.time()-t0:.0f}s")
    for fam in ("prometheus", "kyverno"):
        print(f"\n=== {fam}: pools per regime (modal over subsets), rung x budget ===")
        print(f"{'d':>7} | {'rung':>4} | {'n':>3} | {'irrelevant':>10} {'resolvable':>10} {'irreducible':>11} | some-resolvable")
        for d in BUDGETS:
            for rung in ("0", "1", "2", "free"):
                v = agg[fam][str(d)][rung]
                print(f"{d:>7} | {rung:>4} | {v['n']:>3} | {v['irrelevant']:>10} {v['optimizer_resolvable']:>10} {v['irreducible']:>11} | {v['some_resolvable']}")
            pth = agg[fam][str(d)]["paths"]
            print(f"        paths: monotone irrelevant->resolvable->irreducible {pth['monotone_forward']}/{pth['n']}; ever resolvable {pth['ever_resolvable']}; resolvable before irreducible {pth['resolvable_before_irreducible']}")


if __name__ == "__main__":
    main()
