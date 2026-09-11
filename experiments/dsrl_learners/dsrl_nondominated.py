"""P3.2: non-dominated neural cells on the DSRL/OSRL benchmark.

Executes the P3.2 addendum of paper/Plans/EVIDENCE_FREEZE_MANIFEST.md
(committed 2e0936c before any run): per (task, learner, seed) cell, train
THREE arms at the same episodic cost_limit 10 --

  single-hi : dataset costs relabeled to psi_hi  = 1{speed > s_hi}
  single-sus: dataset costs relabeled to psi_sus = 1{run(speed > s_lo) >= k}
  set       : pointwise max of the two channels (union budget)

with speed_t = ||obs_t[2:4]|| / scale (Ball 0.2, Car 1.0; slices verified
in bullet_safety_gym/envs/agents.py), s_lo/s_hi = Q60/Q90 of dataset
speeds, k = 10, run-lengths reset at episode boundaries. Evaluation rolls
20 live episodes per arm and scores BOTH readings plus the native cost.

Usage (one array element):
  python experiments/dsrl_learners/dsrl_nondominated.py --task OfflineCarRun-v0 --learner cpq \
      --seed 0 --update_steps 100000 --eval_episodes 20 --device cuda \
      --out results/dsrl/nondom/arr_0
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dsrl_sweep import (CONFIG, _NoLogger, build, make_cfg)  # noqa: E402
import gymnasium as gym  # noqa: E402
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa: E402
from osrl.common import TransitionDataset  # noqa: E402
from osrl.common.exp_util import seed_all  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

COST_LIMIT = 10
K_SUSTAIN = 10
Q_LO, Q_HI = 60.0, 90.0
VEL = {  # task prefix -> (slice, scale); pinned in the addendum
    "OfflineBall": (slice(2, 4), 0.2),
    "OfflineCar": (slice(2, 4), 1.0),
}


def vel_spec(task: str):
    for pref, spec in VEL.items():
        if task.startswith(pref):
            return spec
    raise ValueError(f"no pinned velocity slice for {task}")


def speeds_of(obs: np.ndarray, task: str) -> np.ndarray:
    sl, scale = vel_spec(task)
    return np.linalg.norm(np.asarray(obs)[..., sl], axis=-1) / scale


def episode_starts(data: dict) -> np.ndarray:
    n = len(data["observations"])
    done = np.zeros(n, dtype=bool)
    for key in ("terminals", "timeouts"):
        if key in data:
            done |= np.asarray(data[key], dtype=bool)
    starts = np.zeros(n, dtype=bool)
    starts[0] = True
    starts[1:] = done[:-1]
    return starts


def channels(data: dict, task: str):
    """(c_hi, c_sus, quantiles, witness stats) from the offline dataset."""
    sp = speeds_of(data["observations"], task)
    s_lo, s_hi = np.percentile(sp, [Q_LO, Q_HI])
    c_hi = (sp > s_hi).astype(np.float32)
    over = sp > s_lo
    starts = episode_starts(data)
    run = np.zeros(len(sp), dtype=np.int64)
    r = 0
    for i in range(len(sp)):
        if starts[i]:
            r = 0
        r = r + 1 if over[i] else 0
        run[i] = r
    c_sus = (run >= K_SUSTAIN).astype(np.float32)
    both = float(np.mean((c_hi > 0) & (c_sus > 0)))
    hi_only = float(np.mean((c_hi > 0) & (c_sus == 0)))
    sus_only = float(np.mean((c_sus > 0) & (c_hi == 0)))
    return c_hi, c_sus, dict(s_lo=float(s_lo), s_hi=float(s_hi),
                             k=K_SUSTAIN, frac_hi_only=hi_only,
                             frac_sus_only=sus_only, frac_both=both)


@torch.no_grad()
def eval_readings(model, env, task, s_lo, s_hi, episodes, reward_scale,
                  learner):
    """Mirror of OSRL's rollout loop, scoring both readings + native cost.
    Each learner's act() is called exactly as its own trainer's rollout
    does (bcql: act(obs); cpq: act(obs, True, True))."""
    rows = []
    for _ in range(episodes):
        obs, _info = env.reset()
        ret, c_nat, c_hi, c_sus, run, steps = 0.0, 0.0, 0.0, 0.0, 0, 0
        for _t in range(model.episode_len):
            sp = float(speeds_of(np.asarray(obs), task))
            c_hi += float(sp > s_hi)
            run = run + 1 if sp > s_lo else 0
            c_sus += float(run >= K_SUSTAIN)
            if learner == "cpq":
                act, _ = model.act(obs, True, True)
            else:
                act, _ = model.act(obs)
            obs, reward, terminated, truncated, info = env.step(act)
            ret += float(reward)
            c_nat += float(info["cost"])
            steps += 1
            if terminated or truncated:
                break
        rows.append(dict(ret=ret / reward_scale, cost_hi=c_hi,
                         cost_sus=c_sus, cost_native=c_nat, len=steps))
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg.update({f"viol_{k[5:]}": float(np.mean([r[k] > COST_LIMIT
                                                for r in rows]))
                for k in ("cost_hi", "cost_sus", "cost_native")})
    return agg, rows


def train_arm(task, learner, seed, arm, costs, device, update_steps,
              eval_episodes, num_workers, s_lo, s_hi):
    cfg = make_cfg(learner, task, COST_LIMIT, seed, device, update_steps,
                   eval_episodes)
    seed_all(cfg["seed"])
    env = gym.make(task)
    data = env.get_dataset()
    env.set_target_cost(cfg["cost_limit"])
    max_r = float(env.max_episode_reward)
    min_r = float(env.min_episode_reward)
    data = env.pre_process_data(data, cfg["outliers_percent"],
                                cfg["noise_scale"], cfg["inpaint_ranges"],
                                cfg["epsilon"], cfg["density"])
    data = dict(data)
    data["costs"] = np.asarray(costs, dtype=np.float32)
    env = wrap_env(env, reward_scale=cfg["reward_scale"])
    env = OfflineEnvWrapper(env)
    dataset = TransitionDataset(data, reward_scale=cfg["reward_scale"],
                                cost_scale=cfg["cost_scale"],
                                state_init=False)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"],
                        pin_memory=(num_workers > 0),
                        num_workers=num_workers)
    model, trainer = build(learner, env, cfg, dataset=dataset)
    it = iter(loader)
    t0 = time.time()
    for _step in range(cfg["update_steps"]):
        obs, next_obs, act, rew, cost, done = [b.to(device)
                                               for b in next(it)]
        trainer.train_one_step(obs, next_obs, act, rew, cost, done)
    agg, rows = eval_readings(model, env, task, s_lo, s_hi,
                              cfg["eval_episodes"], cfg["reward_scale"],
                              learner)
    agg.update(arm=arm, train_sec=round(time.time() - t0, 1),
               max_epi_r=max_r, min_epi_r=min_r,
               ret_norm=(agg["ret"] - min_r) / (max_r - min_r)
               if max_r > min_r else None,
               cost_hi_norm=agg["cost_hi"] / COST_LIMIT,
               cost_sus_norm=agg["cost_sus"] / COST_LIMIT,
               cost_native_norm=agg["cost_native"] / COST_LIMIT)
    return agg, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--learner", required=True, choices=["bcql", "cpq"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--update_steps", type=int, default=100000)
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out", default="results/dsrl/nondom")
    args = ap.parse_args()

    # channels are computed ONCE from the same preprocessing the arms see
    cfg0 = make_cfg(args.learner, args.task, COST_LIMIT, args.seed,
                    args.device, args.update_steps, args.eval_episodes)
    env0 = gym.make(args.task)
    data0 = env0.get_dataset()
    data0 = env0.pre_process_data(data0, cfg0["outliers_percent"],
                                  cfg0["noise_scale"],
                                  cfg0["inpaint_ranges"],
                                  cfg0["epsilon"], cfg0["density"])
    c_hi, c_sus, wit = channels(data0, args.task)
    del env0, data0

    out = dict(manifest="EVIDENCE_FREEZE_MANIFEST.md@2e0936c (P3.2)",
               task=args.task, learner=args.learner, seed=args.seed,
               cost_limit=COST_LIMIT, update_steps=args.update_steps,
               eval_episodes=args.eval_episodes, witness=wit, arms={})
    print(f"[{args.task}/{args.learner}/s{args.seed}] witness: {wit}",
          flush=True)
    for arm, costs in (("single_hi", c_hi), ("single_sus", c_sus),
                       ("set", np.maximum(c_hi, c_sus))):
        agg, rows = train_arm(args.task, args.learner, args.seed, arm,
                              costs, args.device, args.update_steps,
                              args.eval_episodes, args.num_workers,
                              wit["s_lo"], wit["s_hi"])
        out["arms"][arm] = dict(agg=agg, episodes=rows)
        print(f"  arm={arm}: ret_norm={agg['ret_norm']} "
              f"hi={agg['cost_hi_norm']:.2f} sus={agg['cost_sus_norm']:.2f} "
              f"native={agg['cost_native_norm']:.2f} "
              f"({agg['train_sec']}s)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(
        args.out, f"nondom_{args.task}_{args.learner}_s{args.seed}_{stamp}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
