"""Run a contiguous slice of the training matrix inside one allocation.

Submitting 279 single-run array elements makes every run wait for its own
scheduling slot, and on a busy partition the scheduler grants only a couple at
a time regardless of the array throttle. Packing instead asks for a few wide
allocations and parallelises across cores inside each one, which converts
queue contention into intra-node parallelism we control.

Each worker pins torch to a small thread count so N workers on one node do not
oversubscribe: the networks are small enough that thread-level parallelism
inside a single run buys much less than running more runs at once.

Run: python -m corset_e2e.scripts.run_pack --start 0 --count 24 --workers 12
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.scripts.build_matrix import matrix  # noqa: E402


def one(job, steps, threads):
    idx, arm, learner, size, seed = job
    out = os.path.join(ROOT, "results/e2e", "offline",
                       f"cfg{idx:02d}_{arm}_{learner}_n{size}_s{seed}.json")
    if os.path.exists(out):
        return (job, "cached", 0.0)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), PYTHONPATH=ROOT)
    cmd = [sys.executable, "-m", "corset_e2e.learners.vector_offline",
           "--index", str(idx), "--arm", arm, "--learner", learner,
           "--size", str(size), "--seed", str(seed), "--steps", str(steps),
           "--device", "cpu"]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    if p.returncode != 0:
        return (job, "FAIL: " + (p.stderr.strip().splitlines() or ["?"])[-1][:160], dt)
    return (job, "ok", dt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200_000)
    a = ap.parse_args()

    jobs = matrix()[a.start:a.start + a.count]
    print(f"pack: {len(jobs)} runs, {a.workers} workers x {a.threads} threads",
          flush=True)
    done = fail = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one, j, a.steps, a.threads): j for j in jobs}
        for f in as_completed(futs):
            job, status, dt = f.result()
            done += 1
            if status.startswith("FAIL"):
                fail += 1
            print(f"  [{done}/{len(jobs)}] cfg{job[0]} {job[1]} s{job[4]} "
                  f"{status} {dt/60:.1f}m", flush=True)
    print(f"pack complete: {done} run, {fail} failed", flush=True)


if __name__ == "__main__":
    main()
