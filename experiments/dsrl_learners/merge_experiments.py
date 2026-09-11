#!/usr/bin/env python3
"""Merge per-seed SLURM-array shards of saorl.experiments into ONE results file.

The kill-test driver (saorl.experiments) writes one JSON per process, and its
paired Wilcoxon signed-rank statistic is computed *across seeds*. When we shard
the seed axis over a SLURM array (one seed per element, each writing
results/arr_<id>/experiments_*.json), no single shard has all the seeds -- so we
pool every shard's raw per-seed ``records`` here and re-run the identical
``summarize`` over the union. The output is byte-for-byte the same shape as a
single-process run, so paper/make_table.py picks it up unchanged.

Usage:
  python experiments/dsrl_learners/merge_experiments.py [shard_glob_root] [--out results]
    shard_glob_root default: results   (scans results/arr_*/experiments_*.json
                                        plus any results/**/experiments_*.json)
Writes results/experiments_merged_<stamp>.json and prints the merged summary.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

# Allow `python experiments/dsrl_learners/merge_experiments.py` from the repo root: experiments/dsrl_learners/ lands on
# sys.path[0], so add the repo root (its parent) to import the saorl package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saorl.experiments import Record, summarize  # noqa: E402


def load_shard_records(root: str):
    """Collect (records, config) from every experiments_*.json under root, but
    skip already-merged outputs so re-runs don't double-count."""
    files = sorted(glob.glob(os.path.join(root, "**", "experiments_*.json"),
                             recursive=True))
    files = [f for f in files if "experiments_merged_" not in os.path.basename(f)]
    recs, cfgs, used = [], [], []
    for f in files:
        try:
            blob = json.load(open(f))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] skip {f}: {e}")
            continue
        rs = blob.get("records")
        if not rs:
            continue
        recs.extend(rs)
        cfgs.append(blob.get("config", {}))
        used.append(f)
    return recs, cfgs, used


def main():
    ap = argparse.ArgumentParser(description="merge saorl.experiments seed shards")
    ap.add_argument("root", nargs="?", default="results",
                    help="dir to scan recursively for experiments_*.json shards")
    ap.add_argument("--out", default="results", help="output dir")
    args = ap.parse_args()

    rec_dicts, cfgs, used = load_shard_records(args.root)
    if not rec_dicts:
        raise SystemExit(f"no experiments_*.json shards with records under {args.root}/")

    # dedupe by (domain, model, seed): sharding is disjoint, but guard against a
    # re-submitted element overwriting -- keep the last occurrence.
    by_key = {}
    for d in rec_dicts:
        by_key[(d["domain"], d["model"], d["seed"])] = d
    records = [Record(**d) for d in by_key.values()]

    # eps / learners must be identical across shards; take them from the configs.
    eps = next((c.get("eps", 0.05) for c in cfgs if "eps" in c), 0.05)
    learners = next((c["learners"] for c in cfgs if c.get("learners")), [])
    domains = sorted({r.domain for r in records})
    seeds = sorted({r.seed for r in records})

    print(f"# merged {len(used)} shard file(s) -> {len(records)} unique records")
    print(f"# domains={domains}  learners={learners}  seeds(n={len(seeds)})={seeds}")
    summary = summarize(records, eps, learners)

    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(args.out, f"experiments_merged_{stamp}.json")
    from dataclasses import asdict
    json.dump(dict(
        config=dict(seeds=len(seeds), seed_list=seeds, domains=domains,
                    learners=learners, eps=eps, merged_from=len(used)),
        records=[asdict(r) for r in records],
        summary=summary,
    ), open(out_path, "w"), indent=2)
    print(f"\n# wrote {out_path}")


if __name__ == "__main__":
    main()
