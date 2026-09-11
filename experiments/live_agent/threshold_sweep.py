"""W9: calibration-threshold sensitivity of the live crossed pool.

Sweeps the conformal threshold q-hat over the archived 7-persona scores of
all nine live candidates (total caps k in {2,3,4,5,6,8}, run caps B in
{2,3,4}) and, for every retained pool, recomputes by exact enumeration over
the 4096 schedules: the retained and maximal readings, the value screen,
the face test (every optimal schedule of a value-sufficient reading
set-feasible?), and whether the two logged schedules (single-reading guard
111110000000, set guard 111000000000) are safe for the whole retained set.

Scores: demo/ensemble_scores.json (frozen total-cap panel) and
demo/ensemble_scores_runcaps.json (registered run-cap panel); the five
replicate run-cap panels give the range of q-hat at which each run cap
enters.  No new model calls, no new billed calls.

Run: python3 demo/threshold_sweep.py
Writes results/e2e/live_threshold_sweep.json
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT / "results/e2e" / "live_threshold_sweep.json"
W = 12
TOTAL_CAPS = (2, 3, 4, 5, 6, 8)
RUN_CAPS = (2, 3, 4)
GRID = np.round(np.arange(0.20, 0.701, 0.01), 2)


def load(name):
    return [json.loads(l) for l in open(HERE / "logs" / name, encoding="utf8")]


def max_run(a):
    best = run = 0
    for x in a:
        run = run + 1 if x else 0
        best = max(best, run)
    return best


def feasible(a, reading):
    kind, x = reading
    return (sum(a) <= x) if kind == "total" else (max_run(a) <= x)


def main():
    ng = load("eval_no_guard.jsonl")
    v = np.array([e["paid_value"] for e in ng], dtype=float)
    f = np.array([e["free_value"] for e in ng], dtype=float)
    vbar, fbar = v.mean(0), f.mean(0)
    grid = list(itertools.product((0, 1), repeat=W))
    vals = {a: float(sum(vbar[t] if a[t] else fbar[t] for t in range(W))) for a in grid}

    es = json.load(open(HERE / "ensemble_scores.json"))["scores"]
    rc = json.load(open(HERE / "ensemble_scores_runcaps.json"))
    rc_scores = {k: v["scores"] for k, v in rc["per_candidate"].items()}
    stab = json.load(open(HERE / "ensemble_scores_runcaps_stability.json"))
    score = {}
    for k in TOTAL_CAPS:
        score[("total", k)] = float(np.mean(es[f"psi_calls>={k}"]))
    for b in RUN_CAPS:
        score[("run", b)] = float(np.mean(rc_scores[f"psi_run>={b}"]))
    panels = {f"psi_run>={b}": [float(np.mean(p[f"psi_run>={b}"])) for p in stab["panels"]] for b in RUN_CAPS}

    def V(constraints):
        best, arg = -np.inf, None
        for a in grid:
            if all(feasible(a, rd) for rd in constraints) and vals[a] > best:
                best, arg = vals[a], a
        return best, arg

    logged = {"single_k5": tuple(int(c) for c in "111110000000"),
              "set_k3": tuple(int(c) for c in "111000000000")}
    rows = []
    for q in GRID:
        pool = [rd for rd, s in score.items() if s >= q - 1e-12]
        if not pool:
            rows.append(dict(qhat=float(q), retained=[], empty=True)); continue
        # maximal readings: not implied by another retained reading over the schedule class
        def implies(r1, r2):
            return all(feasible(a, r2) for a in grid if feasible(a, r1))
        maximal = [rd for rd in pool if not any(rd != r2 and implies(r2, rd) and not implies(rd, r2) for r2 in pool)]
        VU, aU = V(pool)
        singles = {}
        value_clear = face_clear = False
        for rd in pool:
            Vp, _ = V([rd])
            singles[f"{rd[0]}{rd[1]}"] = Vp
            if abs(Vp - VU) < 1e-12:
                value_clear = True
                opts = [a for a in grid if feasible(a, rd) and abs(vals[a] - Vp) < 1e-12]
                if all(all(feasible(a, r2) for r2 in pool) for a in opts):
                    face_clear = True
        strictest = None
        for rd in pool:
            if all(implies(rd, r2) for r2 in pool):
                strictest = f"{rd[0]}{rd[1]}"
        rows.append(dict(qhat=float(q), retained=[f"{k}{x}" for k, x in pool],
                         maximal=[f"{k}{x}" for k, x in maximal], n_retained=len(pool), n_maximal=len(maximal),
                         V_U=VU, screen_fires=bool(min(singles.values()) - VU > 1e-9),
                         value_clear=value_clear, face_clear=face_clear,
                         pointwise_strictest=strictest,
                         single_k5_set_safe=all(feasible(logged["single_k5"], rd) for rd in pool),
                         set_k3_set_safe=all(feasible(logged["set_k3"], rd) for rd in pool)))
    # entry thresholds: the q at which each reading drops out
    entry = {f"{k}{x}": s for (k, x), s in score.items()}
    # knife-edge summary: readings within replicate-panel spread of the corpus q-hat = 0.5
    spread = {nm: (min(p), max(p)) for nm, p in panels.items()}
    verdict_changes = []
    prev = None
    for r in rows:
        key = (tuple(r.get("retained", [])), r.get("face_clear"), r.get("set_k3_set_safe"), r.get("single_k5_set_safe"))
        if prev is not None and key != prev:
            verdict_changes.append(r["qhat"])
        prev = key
    res = dict(registration=("W9 threshold sweep, 2026-09-02: archived frozen scores, exact schedule enumeration, "
                             "no new model or billed calls"),
               corpus_qhat=0.5, scores=entry, runcap_panel_spread=spread, rows=rows,
               verdict_change_points=verdict_changes)
    OUT.write_text(json.dumps(res, indent=1))
    for r in rows:
        if r.get("empty"):
            continue
        print(f"q={r['qhat']:.2f} retained={r['retained']} max={r['maximal']} value_clear={r['value_clear']} "
              f"face_clear={r['face_clear']} strictest={r['pointwise_strictest']} k3safe={r['set_k3_set_safe']} k5safe={r['single_k5_set_safe']}")
    print("scores", entry); print("change points", verdict_changes)


if __name__ == "__main__":
    main()
