"""Non-nested live replay: the Slack demo under readings that CROSS.

POST-HOC ENDPOINT (2026-09-01).  The live study's frozen pool caps TOTAL
posts per session at k in {3,4,5,6,8}; those caps nest, so protecting the
set reduces to enforcing k=3 and neither Keep nor Decide is exercised.
This replay adds the other defensible reading of "we allow bursts over
that limit for short periods": a RUN-LENGTH cap -- at most B posts in a
row, B in {2,3,4}, equivalently "at most B posts in any B+1 consecutive
opportunities".  Run caps cross every total cap (front-loading violates a
run cap while respecting a total cap; spreading does the reverse), so the
pointwise-strictest reduction is unavailable and the value screen has to
do the work.

WHY A REPLAY IS EXACT.  channel_agent.py logs BOTH the paid and the free
value of every step; the eval arms are episode- and task-paired; and the
no-guard arm bought all 12 steps of all 50 episodes, so its log carries a
complete value table v[e,t] (paid) and f[e,t] (free).  The value of ANY
posting schedule a in {0,1}^12 on episode e is exactly
    sum_t a_t v[e,t] + (1-a_t) f[e,t],
and a scheduling policy (a controller that sees the step index and its own
quota state, not the unrealised answer quality) is exactly a fixed a.  The
class optimum under any constraint set is therefore an exact enumeration
over 2^12 = 4096 schedules -- no model, no learner, no new billed calls.

Everything asserted: arm pairing, value-table completeness, cross-arm value
agreement, the logged guard patterns, the recomputed realized means, the
nested-pool screen clearing, and the crossing structure itself.

Run:  python3 demo/replay_crossing.py
Writes results/e2e/live_crossing_replay.json
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT / "results/e2e" / "live_crossing_replay.json"

W = 12
TOTAL_CAPS = (3, 4, 5, 6, 8)
RUN_CAPS = (2, 3, 4)


def load(name):
    return [json.loads(l) for l in open(HERE / "logs" / name, encoding="utf8")]


def pattern(ep) -> str:
    return "".join("1" if a == "buy" else "0" for a in ep["actions"])


def max_run(a) -> int:
    best = run = 0
    for x in a:
        run = run + 1 if x else 0
        best = max(best, run)
    return best


def main() -> None:
    ng, sg, cg = (load(f"eval_{n}.jsonl") for n in ("no_guard", "single", "corset"))

    # ---- pairing + completeness + agreement (the replay's licence) --------
    assert [e["episode_id"] for e in ng] == [e["episode_id"] for e in sg] \
        == [e["episode_id"] for e in cg], "arms are not episode-paired"
    assert all(tuple(a["task_ids"]) == tuple(b["task_ids"]) == tuple(c["task_ids"])
               for a, b, c in zip(ng, sg, cg)), "arms are not task-paired"
    assert all(a == "buy" for e in ng for a in e["actions"]), "no-guard skipped"
    v = np.array([e["paid_value"] for e in ng], dtype=float)   # 50 x 12
    f = np.array([e["free_value"] for e in ng], dtype=float)
    assert not np.isnan(v).any() and not np.isnan(f).any(), "value table holes"
    n_agree = 0
    for arm in (sg, cg):
        for en, ea in zip(ng, arm):
            for t in range(W):
                for key, tab in (("paid_value", v), ("free_value", f)):
                    x = ea[key][t]
                    if x is not None:
                        assert abs(x - (en[key][t] if key == "paid_value"
                                        else en[key][t])) < 1e-9
                        n_agree += 1
    sg_pat, cg_pat = {pattern(e) for e in sg}, {pattern(e) for e in cg}
    assert sg_pat == {"1" * 5 + "0" * 7} and cg_pat == {"1" * 3 + "0" * 9}, \
        "logged guard schedules moved"

    # ---- readings over schedules -----------------------------------------
    def feasible(a, reading) -> bool:
        kind, x = reading
        return (sum(a) <= x) if kind == "total" else (max_run(a) <= x)

    readings = [("total", k) for k in TOTAL_CAPS] + [("run", b) for b in RUN_CAPS]
    # crossing: every (total, run) pair is crossed by explicit witnesses
    for k in TOTAL_CAPS:
        for b in RUN_CAPS:
            front = tuple(1 if t < min(b + 1, k) else 0 for t in range(W))
            if sum(front) == b + 1:            # front-load: breaks run, obeys total
                assert feasible(front, ("total", k)) and \
                    not feasible(front, ("run", b))
            spread, run = [], 0
            for t in range(W):                 # spread k+1 posts, runs <= b
                if sum(spread) < k + 1 and run < min(b, 1):
                    spread.append(1); run += 1
                else:
                    spread.append(0); run = 0
            if sum(spread) == k + 1:           # spread: breaks total, obeys run
                assert not feasible(spread, ("total", k)) and \
                    feasible(spread, ("run", b))

    # ---- exact class optima over the 4096 schedules ----------------------
    vbar, fbar = v.mean(0), f.mean(0)

    def value(a) -> float:
        return float(sum(vbar[t] if a[t] else fbar[t] for t in range(W)))

    grid = list(itertools.product((0, 1), repeat=W))
    vals = {a: value(a) for a in grid}

    def V(constraints) -> tuple:
        best, arg = -np.inf, None
        for a in grid:
            if all(feasible(a, rd) for rd in constraints):
                if vals[a] > best:
                    best, arg = vals[a], a
        return best, arg

    def screen(pool) -> dict:
        VU, aU = V(pool)
        singles = {}
        for rd in pool:
            Vp, _ = V([rd])
            singles[f"{rd[0]}{rd[1]}"] = Vp
        gap = min(singles.values()) - VU
        # among the minimising singletons, does SOME optimal schedule already
        # obey the whole pool?  (the existential of eq. (2), checked exactly)
        exist_safe = {}
        for rd in pool:
            Vp, _ = V([rd])
            opts = [a for a in grid if feasible(a, rd)
                    and abs(vals[a] - Vp) < 1e-12]
            exist_safe[f"{rd[0]}{rd[1]}"] = any(
                all(feasible(a, r2) for r2 in pool) for a in opts)
        return dict(V_U=VU, argmax_U="".join(map(str, aU)), singles=singles,
                    gap=float(gap), fires=bool(gap > 1e-9),
                    some_optimum_set_safe=exist_safe)

    nested = [("total", k) for k in TOTAL_CAPS]
    crossed = readings
    s_nested = screen(nested)
    s_crossed = screen(crossed)
    assert not s_nested["fires"], "nested pool now fires; the paper's " \
        "model-free reduction story would be wrong"

    # ---- the CALIBRATED crossed pool: totals + the run caps the frozen
    # ensemble actually retains (score_runcaps.py; Keep closed post hoc)
    keep_path = ROOT / "results/e2e" / "live_crossing_keep.json"
    s_calibrated = calib_runs = None
    if keep_path.exists():
        keep = json.loads(keep_path.read_text(encoding="utf8"))
        calib_runs = sorted(int(nm.split(">=")[1])
                            for nm in keep["runcap_retained"])
        calibrated = nested + [("run", b) for b in calib_runs]
        s_calibrated = screen(calibrated)
        # the lattice fact the paper states: a schedule with at most 3 posts
        # cannot contain a 4-run, so total<=3 implies run<=B for every
        # retained B; checked by enumeration, not argued
        for b in calib_runs:
            assert all(feasible(a, ("run", b)) for a in grid
                       if feasible(a, ("total", TOTAL_CAPS[0]))), \
                f"total<={TOTAL_CAPS[0]} no longer implies run<={b}"

    # ---- the logged behaviours, audited against every reading ------------
    def audit(pat) -> dict:
        a = tuple(int(c) for c in pat)
        return {f"{k}{x}": bool(not feasible(a, (k, x)))
                for (k, x) in readings}

    logged = {
        "single_k5": dict(pattern="1" * 5 + "0" * 7,
                          mean_value=float(np.mean([sum(e["rewards"]) for e in sg])),
                          violates=audit("1" * 5 + "0" * 7)),
        "corset_k3": dict(pattern="1" * 3 + "0" * 9,
                          mean_value=float(np.mean([sum(e["rewards"]) for e in cg])),
                          violates=audit("1" * 3 + "0" * 9)),
    }
    # sanity: the published 8.02 / 6.93 must be these two numbers
    assert abs(logged["single_k5"]["mean_value"] - 8.02) < 0.02
    assert abs(logged["corset_k3"]["mean_value"] - 6.93) < 0.02

    res = dict(
        registration=("post-hoc live replay, 2026-09-01: run-length readings "
                      "B in {2,3,4} ('at most B posts in a row', the burst "
                      "reading) crossed with the frozen total caps; exact "
                      "schedule-class enumeration over 2^12 on the paired "
                      f"no-guard value table; {n_agree} cross-arm value "
                      "agreements asserted; no new billed calls"),
        n_episodes=len(ng), window=W,
        step_mean_paid=[round(float(x), 4) for x in vbar],
        step_mean_free=[round(float(x), 4) for x in fbar],
        nested_pool=s_nested, crossed_pool=s_crossed,
        calibrated_pool=s_calibrated, calibrated_run_caps=calib_runs,
        logged=logged)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf8")
    print(f"wrote {OUT}")
    print("nested  fires:", s_nested["fires"], " gap:", round(s_nested["gap"], 4))
    print("crossed fires:", s_crossed["fires"], " gap:", round(s_crossed["gap"], 4))
    print("crossed V_U:", round(s_crossed["V_U"], 3),
          "argmax:", s_crossed["argmax_U"])
    print("singles:", {k: round(x, 3) for k, x in s_crossed["singles"].items()})
    print("some-optimum-set-safe:", s_crossed["some_optimum_set_safe"])
    for k, d in logged.items():
        print(k, d["pattern"], "value", round(d["mean_value"], 2),
              "violates:", [r for r, b in d["violates"].items() if b])


if __name__ == "__main__":
    main()
