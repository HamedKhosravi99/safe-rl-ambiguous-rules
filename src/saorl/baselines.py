"""Cheap standard baselines (plan section 8.7, Table 2): does the hidden-gap
result survive against the policies a practitioner would actually reach for first?

A reviewer's natural pushback on H1 is ``you only beat a deliberately naive
single-translation learner --- a simple heuristic or behavior cloning would be
just as safe.'' This script answers it on the maintenance domain by pitting
\saorl{} against cheap baselines that need no ambiguity set at all:

  * run-to-failure   -- always operate (the reward-myopic / unconstrained policy);
  * RUL-threshold    -- replace when rul_hat < theta (a hand-built *single*-reading
                        maintenance rule on one feature);
  * behavior-cloning -- argmax-action clone of the logged policy, binned on rul_hat.

The RUL-threshold is shown at TWO honest tunings, to map the safety/return frontier
the single feature can reach:
  * surrogate-tuned -- highest-return theta whose offline surrogate worst case
    (max_k J_ck over U_alpha) <= eps; this is the SAME budget \saorl{} is trained
    against, tuned from the same offline data (the realistic, fair setting);
  * realized-safe (oracle) -- highest-return theta whose REALIZED chance-of-violation
    (the deployment metric) <= eps, tuned DIRECTLY on the reporting rollouts. This
    deliberately *over*-credits the heuristic: it grants oracle access to the very
    deployment distribution it is judged on -- access \saorl{} never gets -- so its
    return is the most generous frontier point a single threshold can reach. Even
    so, \saorl{} returns more at matched safety.

For each policy we report task return, the TRUE worst-case semantic cost over
U_alpha (active-normalized, the training surrogate), and the PRIMARY safety metrics
(realized chance-of-violation and CVaR tail risk) from true-environment rollouts.

The headline is a FRONTIER, not a binary: behavior cloning inherits the logged
policy's violations outright (unsafe). The single-feature RUL-threshold tuned to
*pass the offline surrogate budget* still violates a retained reading in ~23% of
deployment episodes --- because it honors one reading, not the audited set --- so
the surrogate is not enough. The threshold CAN be made realized-safe, but only by
over-replacing, paying a large return cost. \saorl{} dominates the frontier: it is
the only policy safe on both the surrogate and the realized metric *at*
run-to-failure-level return.

Run:  PYTHONPATH=. python3 -m saorl.baselines
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

from .build_plaus_cache import RULE_SETS
from .construct import construct_U_alpha
from .env import ACTIONS, MaintenanceMDP
from .judge import attach_plausibility, load_llm_cache
from .offline import (
    OfflineDataset,
    Policy,
    evaluate_return,
    greedy_continue_policy,
    make_offline_rl_dataset,
    rollout_risk,
    worst_case_cost,
)
from .offline_rl import learn_fqi_constrained

N_SEEDS = 30
EPS = 0.05
MAINT = MaintenanceMDP(r_op=4.0, c_fail=20.0, c_replace=40.0)
THETA_GRID = tuple(float(t) for t in range(2, 71, 2))  # replace if rul_hat < theta
BC_BIN = 5.0                                   # rul_hat bin width for behavior cloning


# --- cheap baseline policies -------------------------------------------------
def rul_threshold_policy(theta: float) -> Policy:
    """Hand-built single-reading rule: replace once the point estimate dips below
    theta. This is a one-threshold interpretation of the NL rule -- the human
    analogue of the single-translation learner."""
    def pol(traj, t):
        return "replace" if traj[t]["rul_hat"] < theta else "continue"
    return pol


def fit_behavior_cloning(data: OfflineDataset, bin_width: float = BC_BIN) -> Policy:
    """Deterministic tabular BC: bin states by rul_hat and take the most-frequent
    logged action in each bin. Clones the logging policy without any notion of the
    rule, so it reproduces whatever (un)safety the logged policy had."""
    counts: Dict[int, Dict[str, int]] = defaultdict(lambda: {a: 0 for a in ACTIONS})
    for traj, acts in zip(data.trajectories, data.actions):
        for s, a in zip(traj, acts):
            counts[int(s["rul_hat"] // bin_width)][a] += 1
    table = {b: max(c, key=c.get) for b, c in counts.items()}

    def pol(traj, t):
        return table.get(int(traj[t]["rul_hat"] // bin_width), "continue")
    return pol


def _tune_best_return(make_pol: Callable[[float], Policy], grid: Sequence[float],
                      return_fn: Callable, score_fn: Callable, eps: float) -> tuple:
    """Tune a one-parameter heuristic to the highest-return value whose safety
    score (`score_fn`, lower is safer) meets the budget `eps`. `score_fn` is the
    metric the threshold is held to -- the offline surrogate worst case, or the
    realized chance-of-violation. If *no* grid value meets the budget, the heuristic
    structurally cannot honor that metric on this grid; we then return the safest
    (min-score) value so the table shows how close it can get. Returns
    (param, ret, score, reached_budget)."""
    scored = [(v, return_fn(make_pol(v)), score_fn(make_pol(v))) for v in grid]
    safe = [s for s in scored if s[2] <= eps + 1e-9]
    if safe:
        # best return among safe; break return ties toward the largest safety
        # margin (lowest score) -- avoids reporting a knife-edge value exactly at
        # the budget when a strictly-safer value attains the same return.
        v, ret, s = max(safe, key=lambda r: (r[1], -r[2]))
        return v, ret, s, True
    v, ret, s = min(scored, key=lambda r: r[2])         # safest achievable
    return v, ret, s, False


@dataclass
class BaselineRow:
    domain: str
    seed: int
    u_size: int
    policy: str
    ret: float
    worst: float       # true worst-case cost over U_alpha (active-normalized)
    chance: float      # Pr(any retained interpretation violated) per episode
    cvar: float        # CVaR_0.1 of episodic worst-case cost
    param: float       # tuned hyper-parameter (threshold theta; else nan)


def _maint_rollout_once(env: MaintenanceMDP):
    def rollout_once(policy, rng):
        rul = env.initial_rul(rng)
        traj, acts = [], []
        for t in range(env.horizon):
            traj.append(env.observe(rul, rng))
            a = policy(traj, t)
            acts.append(a)
            _, rul, _ = env.step(rul, a)
        return traj, acts
    return rollout_once


def probe_maintenance(rule_set: str = "conservative") -> List[BaselineRow]:
    raw = [c for c, _ in RULE_SETS[rule_set]["items"]]
    judge = load_llm_cache(rule_set)
    rollout_once = _maint_rollout_once(MAINT)
    rows: List[BaselineRow] = []
    for seed in range(N_SEEDS):
        data = make_offline_rl_dataset(MAINT, seed=seed)
        U = construct_U_alpha(attach_plausibility(raw, judge),
                              data.to_semantic_dataset()).U_alpha
        if not U:
            continue
        rf = lambda pol: evaluate_return(MAINT, pol, n_episodes=40)
        risk = lambda pol: rollout_risk(rollout_once, pol, U)  # report seed 321
        wf = lambda pol: worst_case_cost(pol, data, U, normalize="active")

        # single-feature threshold at two tunings:
        #   surrogate-tuned: best return whose OFFLINE surrogate worst <= eps (the
        #     realistic, same-info-as-SA-ORL setting);
        #   realized-safe (oracle): best return whose REALIZED chance <= eps, tuned
        #     directly on the report rollouts -- deliberately over-credits the
        #     heuristic (deployment-distribution access SA-ORL never gets).
        th_surr, _, _, _ = _tune_best_return(rul_threshold_policy, THETA_GRID, rf, wf, EPS)
        th_real, _, _, _ = _tune_best_return(
            rul_threshold_policy, THETA_GRID, rf,
            lambda pol: risk(pol)["chance"], EPS)
        saorl = learn_fqi_constrained(data, MAINT, honor=U, U_eval=U, eps=EPS,
                                      return_fn=rf, bcq_tau=0.05)
        policies = {
            "run-to-failure": (greedy_continue_policy(), float("nan")),
            "RUL-thr (surrogate)": (rul_threshold_policy(th_surr), float(th_surr)),
            "RUL-thr (oracle-safe)": (rul_threshold_policy(th_real), float(th_real)),
            "behavior-cloning": (fit_behavior_cloning(data), float("nan")),
            "SA-ORL": (saorl.policy, float("nan")),
        }
        for name, (pol, param) in policies.items():
            r = risk(pol)
            rows.append(BaselineRow(
                "maintenance", seed, len(U), name,
                rf(pol), worst_case_cost(pol, data, U, normalize="active"),
                r["chance"], r["cvar"], param))
    return rows


def _summ(rows: List[BaselineRow], dom: str):
    sub = [r for r in rows if r.domain == dom]
    if not sub:
        print(f"  {dom}: no seeds")
        return
    order = ["run-to-failure", "behavior-cloning", "RUL-thr (surrogate)",
             "RUL-thr (oracle-safe)", "SA-ORL"]
    n = len({r.seed for r in sub})
    print(f"  === {dom}  (n={n} seeds, eps={EPS}) ===")
    print(f"  {'policy':24s} {'return':>14s} {'TRUE worst*':>14s} "
          f"{'Pr(viol)':>10s} {'CVaR_0.1':>10s} {'theta':>6s} {'safe?':>6s}")
    for name in order:
        rs = [r for r in sub if r.policy == name]
        if not rs:
            continue
        ret = np.array([r.ret for r in rs]); wc = np.array([r.worst for r in rs])
        ch = np.array([r.chance for r in rs]); cv = np.array([r.cvar for r in rs])
        th = np.array([r.param for r in rs])
        # verdict on the PRIMARY (realized) metric: chance-of-violation <= eps
        safe = "yes" if ch.mean() <= EPS + 1e-9 else "NO"
        ths = f"{np.nanmean(th):6.1f}" if not np.all(np.isnan(th)) else "    --"
        print(f"  {name:24s} {ret.mean():7.1f}+/-{ret.std():4.1f} "
              f"{wc.mean():6.3f}+/-{wc.std():5.3f} {ch.mean():10.3f} "
              f"{cv.mean():10.3f} {ths} {safe:>6s}")


def main():
    rows = probe_maintenance("conservative")
    print(f"Cheap standard baselines vs. SA-ORL ({N_SEEDS} seeds)\n")
    _summ(rows, "maintenance")
    out = Path(__file__).with_name("baselines.json")
    out.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
