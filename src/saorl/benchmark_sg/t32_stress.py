"""T3.2 companion: four-split stress table over the 352-rule E0-D corpus.

IID LOO / chronological / repo-held-out / ecosystem-held-out, with
cluster (source-file) bootstrap CIs. Diagnosis of shift, not
certification (A8 companion; the beyond-exchangeability proposition is
stated only if its estimability derivation closes).

Run: SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.t32_stress
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..conformal import conformal_threshold
from .evaluate import build_prom_pool
from .g0_audit import dev_targets
from .parse import prom_threshold_bank
from .run_benchmark import _prom_skeleton, _record

_ROOT = Path(__file__).resolve().parents[3]
_OUT = _ROOT / "results/conformal" / "e0" / "stress_table.json"
DELTA = 0.1
ECOSYS = {"kube-prometheus": "k8s-core", "cluster-monitoring-operator":
          "k8s-core", "thanos": "vendor", "mimir": "vendor"}


def cluster_boot(bits_by_cluster, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    cids = list(bits_by_cluster)
    out = []
    for _ in range(n):
        pick = rng.choice(len(cids), len(cids), replace=True)
        allbits = [b for j in pick for b in bits_by_cluster[cids[j]]]
        out.append(np.mean(allbits))
    return [round(float(np.percentile(out, q)), 4) for q in (2.5, 97.5)]


def eval_split(cal, test):
    gs = [r.gold_score for r in cal]
    q = conformal_threshold(gs, DELTA)
    bits_by_cluster = defaultdict(list)
    for r in test:
        bits_by_cluster[r.t.file].append(r.gold_score >= q)
    bits = [b for v in bits_by_cluster.values() for b in v]
    return dict(n_cal=len(cal), n_test=len(test),
                coverage=round(float(np.mean(bits)), 4),
                cluster_boot_95=cluster_boot(bits_by_cluster),
                n_clusters=len(bits_by_cluster))


def main():
    devs = dev_targets()
    bank = prom_threshold_bank(devs)
    recs = []
    for t in devs:
        rec = _record(build_prom_pool(t, bank), t, _prom_skeleton(t))
        rec.t = t
        recs.append(rec)
    out = {}
    # IID LOO
    gs = [r.gold_score for r in recs]
    bits_by_cluster = defaultdict(list)
    for i, r in enumerate(recs):
        q = conformal_threshold([g for j, g in enumerate(gs) if j != i],
                                DELTA)
        bits_by_cluster[r.t.file].append(r.gold_score >= q)
    bits = [b for v in bits_by_cluster.values() for b in v]
    out["iid_loo"] = dict(n=len(recs),
                          coverage=round(float(np.mean(bits)), 4),
                          cluster_boot_95=cluster_boot(bits_by_cluster))
    # chronological (median last_commit_ts; missing ts -> calibration)
    ts = [(r.t.last_commit_ts or 0) for r in recs]
    med = float(np.median([x for x in ts if x > 0]))
    cal = [r for r, x in zip(recs, ts) if x <= med]
    test = [r for r, x in zip(recs, ts) if x > med]
    out["chronological"] = eval_split(cal, test)
    # repo-held-out
    for repo in sorted({r.t.repo for r in recs}):
        cal = [r for r in recs if r.t.repo != repo]
        test = [r for r in recs if r.t.repo == repo]
        out[f"heldout_{repo}"] = eval_split(cal, test)
    # ecosystem-held-out
    for eco in ("k8s-core", "vendor"):
        cal = [r for r in recs if ECOSYS[r.t.repo] != eco]
        test = [r for r in recs if ECOSYS[r.t.repo] == eco]
        out[f"heldout_eco_{eco}"] = eval_split(cal, test)
    _OUT.write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v.get("coverage"), v.get("cluster_boot_95"))


if __name__ == "__main__":
    main()
