#!/usr/bin/env python3
"""Smoke test for the DSRL 20-seed merge logic in scripts/paper/make_corset_tables.py.

Validates BEFORE the merge touches the shipped table (standing rule: review +
smoke-test code faithful to the math before running the full pipeline):

  1. _dedup_runs unit test (hand-checkable): a duplicated seed collapses to one
     record (last-wins), a distinct seed survives, a record missing a key field
     is dropped.
  2. No-double-count invariant on the REAL result dirs (base sweep + topup4):
     after dedup, every (task, learner, reading) cell has all-distinct seeds.
  3. Per-cell seed-count report for the main arm (merge-progress visibility).

Does NOT call write()/gen_dsrl(), so the committed generated/gen_dsrl.tex is
never clobbered by partial data. Run: python3 experiments/dsrl_learners/test_dsrl_merge.py
"""
import os
import sys
import glob
import json
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))
from make_corset_tables import _dedup_runs  # noqa: E402

# keep in lockstep with gen_dsrl()'s base_dirs
BASE_DIRS = ["results/dsrl/sweep", "results/dsrl/sweep_caps",
             "results/dsrl/sweep_cdt", "results/dsrl/sweep_coptidice",
             "results/dsrl/ext_ext",
             "results/dsrl/ext_p20seed", "results/dsrl/topup",
             "results/dsrl/topup2", "results/dsrl/topup4"]
KEYS = ("task", "learner", "reading", "seed")

fails = []


def check(cond, msg):
    print(("  ok  " if cond else "  FAIL") + " " + msg)
    if not cond:
        fails.append(msg)


def unit_test_dedup():
    print("[1] _dedup_runs unit test (hand-checkable)")
    recs = [
        {"task": "T", "learner": "L", "reading": "single", "seed": 0, "v": 1.0},
        {"task": "T", "learner": "L", "reading": "single", "seed": 0, "v": 9.0},  # dup seed 0 (last wins)
        {"task": "T", "learner": "L", "reading": "single", "seed": 1, "v": 2.0},
        {"task": "T", "learner": "L", "reading": "saorl", "seed": 0, "v": 3.0},   # diff reading -> distinct
        {"task": "T", "learner": "L", "reading": "single", "v": 7.0},             # missing seed -> dropped
    ]
    out = _dedup_runs(recs, KEYS)
    check(len(out) == 3, f"5 raw -> 3 deduped (got {len(out)})")
    single = [r for r in out if r["reading"] == "single"]
    seeds = sorted(r["seed"] for r in single)
    check(seeds == [0, 1], f"single-reading seeds == [0,1] (got {seeds})")
    s0 = next(r for r in single if r["seed"] == 0)
    check(s0["v"] == 9.0, f"last-write-wins for dup seed 0 (v=={s0['v']}, want 9.0)")
    check(any(r["reading"] == "saorl" for r in out),
          "different reading kept as a distinct cell")


def load_raw():
    recs = []
    for d in BASE_DIRS:
        for f in sorted(glob.glob(os.path.join(ROOT, d, "**", "dsrl_sweep_*.json"),
                                  recursive=True)):
            try:
                blob = json.load(open(f))
            except Exception as e:
                print(f"  [warn] skip {f}: {e}")
                continue
            for r in blob.get("runs", []):
                if r.get("status") == "ok" and tuple(r.get("U", [])) == (10, 20, 40):
                    recs.append(r)
    return recs


def real_data_invariant():
    print("[2] no-double-count invariant on real dirs (base + topup4)")
    raw = load_raw()
    if not raw:
        print("  [skip] no DSRL records on disk yet (nothing to check)")
        return
    dedup = _dedup_runs(raw, KEYS)
    print(f"  raw ok-records={len(raw)}  deduped={len(dedup)}  "
          f"removed={len(raw) - len(dedup)} duplicate (task,learner,reading,seed)")
    # core invariant: within each (task,learner,reading) cell, seeds are distinct
    by_cell = defaultdict(list)
    for r in dedup:
        by_cell[(r["task"], r["learner"], r["reading"])].append(r["seed"])
    bad = {k: v for k, v in by_cell.items() if len(v) != len(set(v))}
    check(not bad, f"every deduped cell has all-distinct seeds ({len(bad)} violations)")
    # sanity: dedup on already-deduped data is a fixed point
    check(len(_dedup_runs(dedup, KEYS)) == len(dedup),
          "dedup is idempotent (fixed point)")


def seed_report():
    print("[3] per-cell seed counts, MAIN arm (single & saorl) — merge progress")
    dedup = _dedup_runs(load_raw(), KEYS)
    if not dedup:
        print("  [skip] no records yet")
        return
    tasks = sorted({r["task"] for r in dedup})
    learners = ["bcql", "cpq", "coptidice", "cdt", "caps"]
    both_ok = 0
    for t in tasks:
        cells = []
        for ln in learners:
            ns = len({r["seed"] for r in dedup if r["task"] == t
                      and r["learner"] == ln and r["reading"] == "single"})
            na = len({r["seed"] for r in dedup if r["task"] == t
                      and r["learner"] == ln and r["reading"] == "saorl"})
            cells.append(f"{ln}:{ns}/{na}")
            both_ok += min(ns, na)
        print(f"  {t.replace('Offline','').replace('-v0',''):12} " + "  ".join(cells))
    print(f"  (cell = single_seeds/saorl_seeds; target 20/20 each; "
          f"sum(min)={both_ok})")


if __name__ == "__main__":
    unit_test_dedup()
    real_data_invariant()
    seed_report()
    print()
    if fails:
        print(f"SMOKE TEST FAILED: {len(fails)} assertion(s)")
        for m in fails:
            print("  - " + m)
        sys.exit(1)
    print("SMOKE TEST PASSED")
