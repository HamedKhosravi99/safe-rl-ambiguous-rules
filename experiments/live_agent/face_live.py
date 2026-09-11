"""V13 endpoint 4: policy sufficiency on the live crossed pool, by
exhaustive enumeration of the 4096 schedules (REGISTRATION_V13.md).

The published replay establishes the existential: some value-optimal
schedule under the value-sufficient reading obeys the whole crossed pool
(and the deployed guard's own tie-break does not obey the stipulated
strictest run cap). This script asks the FACE question the V13 theorem
poses: do ALL value-optimal schedules obey the pool? Reuses the replay's
own value model (per-slot paid/free means from the logged sessions) and
reading semantics, and asserts agreement with the archived replay values.

Run: PYTHONPATH=. python3 demo/face_live.py
Writes results/e2e/face_live.json
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REPLAY = ROOT / "results/e2e" / "live_crossing_replay.json"
KEEP = ROOT / "results/e2e" / "live_crossing_keep.json"
OUT = ROOT / "results/e2e" / "face_live.json"

W = 12


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


def main() -> None:
    ng, sg, cg = (load(f"eval_{n}.jsonl") for n in ("no_guard", "single", "corset"))
    v = np.array([e["paid_value"] for e in ng], dtype=float)
    f = np.array([e["free_value"] for e in ng], dtype=float)
    vbar, fbar = v.mean(0), f.mean(0)

    def value(a):
        return float(sum(vbar[t] if a[t] else fbar[t] for t in range(W)))

    grid = list(itertools.product((0, 1), repeat=W))
    vals = {a: value(a) for a in grid}

    rep = json.load(open(REPLAY))
    keep = json.load(open(KEEP))
    pools = {}
    # the two pools the paper reports: the stipulated crossed pool and the
    # calibrated pool retained by the frozen ensemble
    pools["stipulated"] = [("total", k) for k in (3, 4, 5, 6, 8)] + \
                          [("run", b) for b in (2, 3, 4)]
    def parse_name(nm):
        # psi_calls>=k is the total<=k reading, psi_run>=k the run<=k one
        # (the archived retained set: totals 3..8 plus run<=3)
        return ("run" if "run" in nm else "total", int(nm.split(">=")[1]))

    cal = [parse_name(nm) for nm in
           keep["frozen_retained"] + keep["runcap_retained"]]
    pools["calibrated"] = cal

    out = {}
    for name, pool in pools.items():
        VU = max(vals[a] for a in grid
                 if all(feasible(a, rd) for rd in pool))
        rows = {}
        for rd in pool:
            Vp = max(vals[a] for a in grid if feasible(a, rd))
            if abs(Vp - VU) > 1e-12:
                continue                     # not value-sufficient
            face = [a for a in grid if feasible(a, rd)
                    and abs(vals[a] - Vp) < 1e-12]
            unsafe = [a for a in face
                      if not all(feasible(a, r2) for r2 in pool)]
            rows[f"{rd[0]}{rd[1]}"] = dict(
                V=Vp, face_size=len(face), n_unsafe=len(unsafe),
                exists_safe=len(unsafe) < len(face),
                policy_sufficient=not unsafe)
        out[name] = dict(V_U=VU, value_sufficient=rows)
        # reconcile with the archived replay where it reports the same pool
    arch_vu = rep.get("crossed", {}).get("V_U") or rep.get("V_U")
    if arch_vu is not None:
        assert abs(out["stipulated"]["V_U"] - arch_vu) < 1e-9, \
            (out["stipulated"]["V_U"], arch_vu)

    payload = dict(
        registration="V13 endpoint 4 (REGISTRATION_V13.md): exact face "
                     "enumeration over 4096 schedules on the live pools",
        pools=out)
    json.dump(payload, open(OUT, "w"), indent=1)
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
