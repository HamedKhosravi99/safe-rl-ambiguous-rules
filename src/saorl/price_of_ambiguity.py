"""E3/E6: the price of ambiguity -- predicted bound vs realized, and the value
of one clarification query.

Theory being tested (paper Thms 3 and 5, finite-horizon corollary):

  For each admissible intent psi in U, let pi_psi be the return-optimal policy
  honoring psi alone and pi_U the robust policy honoring all of U.  The
  policy-surgery bound says

    V_psi - V_U  <=  (r_max - r_min) * E_{pi_psi}[ (H - tau_D)+ ],

  where tau_D is the first time pi_psi takes the forbidden action on a
  state-action pair where some retained reading fires but psi does not (the
  disputed region), H the episode length, and r_max/r_min the per-step reward
  range.  The right side is estimable from rollouts of pi_psi alone; the left
  side is the realized price.  Falsifiable content: (i) the bound holds for
  every psi, seed, domain; (ii) domains where optimal behavior never enters
  the disputed region (tau_D = infinity, predicted price 0) must show realized
  price ~ 0 EVEN IF |U| > 1 -- ambiguity present but free; (iii) the value of
  one clarification query VoQ = sum_psi p_psi V_psi - V_U (posterior p from
  normalized ensemble plausibility) is nonnegative and bounded by the same
  quantity, giving the ask-vs-act rule: ask iff VoQ > query cost.

Run under the conformal construction:
  SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.price_of_ambiguity
Writes results/conformal/price/price_report.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .experiments import EPS, LAM_GRID, _domain_specs
from .offline import worst_case_cost
from .offline_rl import learn_fqi_constrained

N_SEEDS = 15
N_EPISODES = 300          # rollouts per (policy, seed) for tau_D estimation
DOMAINS = ("synthetic", "real", "gridworld", "budget")


def _fqi(spec_learners, data, env, honor, U, ret_fn):
    fn = spec_learners.get("fqi")
    return fn(data, env, honor=honor, U_eval=U, eps=EPS, return_fn=ret_fn,
              bcq_tau=0.05, lam_grid=LAM_GRID)


def _reward_range(data) -> tuple:
    flat = [r for ep in data.rewards for r in ep]
    return (float(min(flat)), float(max(flat)))


def _disputed_stats(rollout_once, policy, psi, U, n_episodes: int, seed: int):
    """Under `policy` (trained to honor psi alone): per episode, the first step
    tau where the taken action incurs cost 1 under SOME retained reading but 0
    under psi; returns mean (H - tau)+ and the fraction of episodes that ever
    enter the disputed region."""
    rng = np.random.default_rng(10_000 + seed)
    rema, hits = [], 0
    for _ in range(n_episodes):
        traj, acts = rollout_once(policy, rng)
        tau = None
        for t, a in enumerate(acts):
            worst = max(c.cost(traj, t, a) for c in U)
            own = psi.cost(traj, t, a)
            if worst > own:
                tau = t
                break
        if tau is None:
            rema.append(0.0)
        else:
            hits += 1
            rema.append(float(len(acts) - tau))
    return float(np.mean(rema)), hits / n_episodes


def _fallback_action(data, U) -> str:
    """The universal fallback a0: an action with c_psi(s, a0) = 0 for every
    reading (any action outside the union of forbidden sets; deterministic
    choice = first in sorted order)."""
    all_actions = sorted({a for ep in data.actions for a in ep})
    forbidden = set().union(*(c.forbidden_actions for c in U))
    safe = [a for a in all_actions if a not in forbidden]
    if not safe:
        raise RuntimeError("no universal fallback action in the logged data")
    return safe[0]


def _union_guard(U, a0: str, greedy_action: str):
    """The training-free zero-violation fallback policy pi0: take the
    domain's reward-greedy default action unless SOME retained reading's
    predicate fires, in which case take the universal fallback a0.  By
    construction c_k(pi0) = 0 identically for every retained k."""

    def pol(traj, t):
        if any(c.fires(traj, t) for c in U):
            return a0
        return greedy_action

    return pol


def _surgery_policy(base_policy, psi, U, a0: str, greedy_action: str):
    """Theorem 3's witness pi': follow base (the psi-oracle) until it would
    take a forbidden action on a pair some OTHER retained reading forbids,
    then follow the zero-violation union-guard pi0 for the rest of the
    episode.  Pre-switch costs sit on pairs where every retained cost is
    <= psi's cost (truncation of the psi-oracle's own <= eps budget);
    post-switch costs are identically 0 under pi0.  Hence pi' is U-feasible,
    V_U >= V(pi'), and V_psi - V(pi') is a constructive, training-free upper
    bound ('surgery certificate') on the price of ambiguity."""
    guard = _union_guard(U, a0, greedy_action)
    state = {"switched_at": None}

    def pol(traj, t):
        if t == 0:
            state["switched_at"] = None
        if state["switched_at"] is not None:
            return guard(traj, t)
        a = base_policy(traj, t)
        disputed = any(c.cost(traj, t, a) > psi.cost(traj, t, a) for c in U)
        if disputed:
            state["switched_at"] = t
            return guard(traj, t)
        return a

    return pol


def run(seeds: Sequence[int] = tuple(range(N_SEEDS))) -> dict:
    assert os.environ.get("SAORL_CONFORMAL") == "1", \
        "run with SAORL_CONFORMAL=1: the price table is defined over the conformal sets"
    t0 = time.time()
    specs = _domain_specs(DOMAINS, ["fqi"])
    out: Dict[str, dict] = {}
    for dname, spec in specs.items():
        rows: List[dict] = []
        for seed in seeds:
            data, env, U, ret_fn, _ = spec.build(seed)
            rollout_once = spec.rollout(env)
            r_min, r_max = _reward_range(data)
            robust = _fqi(spec.learners, data, env, U, U, ret_fn)
            v_u = robust.ret
            plas = np.array([c.plaus_mean() for c in U])
            post = plas / plas.sum() if plas.sum() > 0 else np.ones(len(U)) / len(U)
            a0 = _fallback_action(data, U)
            forbidden = set().union(*(c.forbidden_actions for c in U))
            greedy_action = sorted(forbidden)[0]   # domains have one forbidden (= greedy) action
            per_psi = []
            for c, p in zip(U, post):
                orac = _fqi(spec.learners, data, env, [c], U, ret_fn)
                mean_rem, hit_frac = _disputed_stats(
                    rollout_once, orac.policy, c, U, N_EPISODES, seed)
                surg = _surgery_policy(orac.policy, c, U, a0, greedy_action)
                v_surgery = ret_fn(surg)
                surgery_worst = worst_case_cost(surg, data, U, normalize="active")
                per_psi.append(dict(
                    psi=c.name, posterior=float(p),
                    v_psi=orac.ret,
                    realized_price=orac.ret - v_u,
                    surgery_certificate=orac.ret - v_surgery,
                    v_surgery=v_surgery,
                    surgery_worst=float(surgery_worst),
                    predicted_bound=(r_max - r_min) * mean_rem,
                    mean_remaining_after_hit=mean_rem,
                    hit_fraction=hit_frac,
                    bound_holds=bool(orac.ret - v_u <= (r_max - r_min) * mean_rem + 1e-9),
                ))
            voq = float(sum(p["posterior"] * p["v_psi"] for p in per_psi) - v_u)
            rows.append(dict(
                seed=seed, v_robust=v_u, u_names=[c.name for c in U],
                r_range=[r_min, r_max], per_psi=per_psi,
                wpoa_realized=float(max(p["realized_price"] for p in per_psi)),
                wpoa_surgery=float(max(p["surgery_certificate"] for p in per_psi)),
                wpoa_predicted=float(max(p["predicted_bound"] for p in per_psi)),
                voq=voq,
            ))
        agg = dict(
            wpoa_realized=float(np.mean([r["wpoa_realized"] for r in rows])),
            wpoa_realized_sd=float(np.std([r["wpoa_realized"] for r in rows])),
            wpoa_surgery=float(np.mean([r["wpoa_surgery"] for r in rows])),
            wpoa_surgery_sd=float(np.std([r["wpoa_surgery"] for r in rows])),
            wpoa_predicted=float(np.mean([r["wpoa_predicted"] for r in rows])),
            voq=float(np.mean([r["voq"] for r in rows])),
            voq_sd=float(np.std([r["voq"] for r in rows])),
            bound_violations=int(sum(1 for r in rows for p in r["per_psi"]
                                     if not p["bound_holds"])),
            n_bound_checks=int(sum(len(r["per_psi"]) for r in rows)),
        )
        out[dname] = dict(rows=rows, agg=agg)
        print(f"[{dname}] WPoA realized={agg['wpoa_realized']:.2f} "
              f"surgery<={agg['wpoa_surgery']:.2f} "
              f"closed-form<={agg['wpoa_predicted']:.2f}  VoQ={agg['voq']:.2f}  "
              f"bound viol {agg['bound_violations']}/{agg['n_bound_checks']}")
    report = dict(config=dict(seeds=list(seeds), eps=EPS, n_episodes=N_EPISODES,
                              construction="conformal"),
                  domains=out, elapsed_s=time.time() - t0)
    return report


def main() -> None:
    report = run()
    out_dir = Path(__file__).parent.parent.parent / "results/conformal" / "price"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "price_report.json"
    json.dump(report, open(path, "w"), indent=1)
    print(f"wrote {path}  ({report['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
