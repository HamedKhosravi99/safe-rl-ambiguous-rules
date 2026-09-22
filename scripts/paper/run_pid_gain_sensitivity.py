#!/usr/bin/env python3
"""PID-Lagrangian gain sensitivity for the agent-service study: rerun the pid learner of scope_agent_learn with its
gains scaled by a factor (0.5 and 2 in the paper), everything else identical, into results/e2e/scope_agent_learn_pid_gains_x<f>.json.
Run from code/: OMP_NUM_THREADS=1 V50_WORKERS=9 PYTHONPATH=src python3 scripts/paper/run_pid_gain_sensitivity.py 0.5 2"""
import json, os, sys, time
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from saorl.benchmark_sg import scope_agent_learn as L

def _task(task):
    return L.run_task(task, learners=("pid",))

def main():
    for f in [float(x) for x in sys.argv[1:]] or [0.5, 2.0]:
        os.environ["PID_GAIN_FACTOR"] = f"{f:g}"          # read by scope_agent_learn at import in every spawned worker
        gains = {k: (v * f if k in ("kp", "ki", "kd") else v) for k, v in L.PID_GAINS.items()}
        tasks = L.certified_instances(); t0 = time.time()
        with Pool(int(os.environ.get("V50_WORKERS", "9"))) as pool: outs = pool.map(_task, tasks, chunksize=1)
        rows = [r for o in outs for r in o["rows"]]
        out = os.path.join(ROOT, "results/e2e", f"scope_agent_learn_pid_gains_x{f:g}.json")
        json.dump(dict(registration="V50 learned arms, PID gain sensitivity", gain_factor=f, gains=gains, n_grid=list(L.N_GRID), seeds=len(L.SEEDS),
                       learners=["pid"], tasks=[o["task"] for o in outs], rows=rows, seconds=time.time() - t0), open(out, "w"))
        print("wrote", out, "rows", len(rows), "in %.0fs" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
