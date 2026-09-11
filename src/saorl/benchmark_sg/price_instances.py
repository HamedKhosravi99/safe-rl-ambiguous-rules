"""W10: price of ambiguity at the independent-instance level.

Pools that share a compiled geometry (monitoring) or a pool signature
(admission) are numerically identical objects; a pool-weighted median
counts them repeatedly.  This emits, from the archives only:

  * monitoring, exact compiled suite: every distinct geometry with its
    relative price, its multiplicity and its screen verdict -- all of them,
    since there are three;
  * admission, free class: the 44 pools grouped by pool signature (11
    distinct instances), pool-weighted and instance-weighted medians, and an
    instance-level bootstrap interval of the median (10,000 resamples, seed
    fixed), labelled descriptive;
  * monitoring, free class: the same grouping (53 distinct instances).

Run: PYTHONPATH=. python3 -m saorl.benchmark_sg.price_instances
Writes results/e2e/price_instances.json
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "results/e2e", "price_instances.json")
OP = 0.05
B = 10000
SEED = 20260902


def boot_median(vals, rng):
    vals = np.asarray(vals, dtype=float)
    meds = [np.median(rng.choice(vals, size=vals.size, replace=True)) for _ in range(B)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main():
    rng = np.random.default_rng(SEED)
    nn = json.load(open(os.path.join(ROOT, "results/e2e", "exact_nonnested.json")))
    geoms = defaultdict(list)
    for i in nn["instances"]:
        c = i["budgets"]["0.05"]
        geoms[i["geometry"]].append(dict(uid=i["uid"], rpoa=c["rpoa"], fires=c["fires_value"], n_states=i["n_states"]))
    mon_compiled = []
    for g, lst in sorted(geoms.items(), key=lambda kv: -len(kv[1])):
        r = {x["rpoa"] for x in lst}
        assert len(r) == 1, g
        assert len({x["fires"] for x in lst}) == 1, g
        mon_compiled.append(dict(geometry=g, n_pools=len(lst), rel_price=lst[0]["rpoa"], fires=lst[0]["fires"],
                                 n_states=lst[0]["n_states"], rules=[x["uid"] for x in lst]))
    pcb = json.load(open(os.path.join(ROOT, "results/e2e", "policy_class_budget.json")))
    fam = {}
    for family in ("kyverno", "prometheus"):
        rows = [r for r in pcb["rows"] if r["policy_class"] == "free" and r["family"] == family
                and abs(r["budget"] - OP) < 1e-9]
        by_sig = defaultdict(list)
        for r in rows:
            by_sig[r["pool_sig"]].append(r)
        inst_vals = []
        groups = []
        for sig, lst in sorted(by_sig.items(), key=lambda kv: -len(kv[1])):
            v = {round(r["rpoa"], 9) for r in lst}
            assert len(v) == 1, (family, sig, v)
            inst_vals.append(lst[0]["rpoa"])
            groups.append(dict(pool_sig=sig, n_pools=len(lst), rel_price=lst[0]["rpoa"],
                               fires=lst[0]["screen_fires"], rules=[r["rule_id"] for r in lst]))
        pool_vals = [r["rpoa"] for r in rows]
        agg = pcb["aggregate"][f"price/{family}_free"][str(OP)]
        assert agg["n_pools"] == len(rows) and agg["n_distinct_instances"] == len(by_sig)
        assert abs(agg["pool_weighted"]["median"] - float(np.median(pool_vals))) < 1e-9
        assert abs(agg["instance_weighted"]["median"] - float(np.median(inst_vals))) < 1e-9
        lo, hi = boot_median(inst_vals, rng)
        fam[family] = dict(n_pools=len(rows), n_distinct_instances=len(by_sig),
                           pool_weighted_median=float(np.median(pool_vals)),
                           instance_weighted_median=float(np.median(inst_vals)),
                           instance_min=float(min(inst_vals)), instance_max=float(max(inst_vals)),
                           n_positive_instances=int(sum(1 for x in inst_vals if x > 1e-12)),
                           instance_bootstrap_median_ci95=[lo, hi], bootstrap=dict(B=B, seed=SEED, unit="distinct instance"),
                           largest_duplicate_group=max(len(l) for l in by_sig.values()),
                           groups=groups)
    res = dict(registration="W10 instance-level price restatement, 2026-09-02, archives only; bootstrap descriptive",
               operating_budget=OP, monitoring_compiled=mon_compiled, free_class=fam)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(dict(monitoring_compiled=mon_compiled,
                          free={k: {kk: vv for kk, vv in v.items() if kk != "groups"} for k, v in fam.items()}), indent=1))


if __name__ == "__main__":
    main()
