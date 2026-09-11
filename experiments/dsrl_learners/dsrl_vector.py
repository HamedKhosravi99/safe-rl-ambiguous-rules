"""A9: vector-cost non-nested neural suite (make8 v4 Tier 5).

Per (cell, seed): four arms at the same episodic budget (10) —
  single_a   dataset costs = channel A
  single_b   dataset costs = channel B
  union      pointwise max(A, B)  (the P3.2 surrogate, kept as comparator)
  vector     costs = [A, B] with per-reading critics and independent
             multipliers (experiments/dsrl_learners/vector_models.py)

Crossings: hi_sus (spike Q90 vs sustained Q60/k=10, the P3.2 channels)
and hi_scope (spike Q90 vs speed>Q75 while |pos|>Q75(|pos|); Ball pos =
obs[0:2], pinned like the VEL slices). Evaluation: 100 independent
episodes per arm; per-seed Theorem-3 certificates: exact Clopper-Pearson
upper bound on the episode-violation rate per channel at 1-δ_ev/2
(δ_ev = 0.05), ship rule CP-upper <= 0.10; the Hoeffding expected-cost
bound is reported and flagged (precomputed vacuous at n=100 — disclosed
in A9). No best-seed selection: every seed is certified.

Usage (one array task = one cell x seed):
  python3 dsrl_vector.py --cell K --seed S [--update_steps N]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dsrl_sweep import _NoLogger, make_cfg  # noqa: E402
from dsrl_nondominated import (COST_LIMIT, K_SUSTAIN, Q_LO, Q_HI,  # noqa: E402
                               channels, episode_starts, speeds_of,
                               vel_spec)
from vector_models import VectorBCQL, VectorCPQ  # noqa: E402
import gymnasium as gym  # noqa: E402
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa: E402
from osrl.algorithms.bcql import BCQL, BCQLTrainer  # noqa: E402
from osrl.algorithms.cpq import CPQ, CPQTrainer  # noqa: E402
from osrl.common import TransitionDataset  # noqa: E402
from osrl.common.exp_util import seed_all  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

Q_SCOPE = 75.0
DELTA_EV = 0.05
SHIP_RATE = 0.10
POS = {"OfflineBall": (slice(0, 2), 1.0)}   # pinned position slice

CELLS = [
    ("OfflineCarRun-v0", "cpq", "hi_sus"),
    ("OfflineCarCircle-v0", "cpq", "hi_sus"),
    ("OfflineBallCircle-v0", "bcql", "hi_sus"),
    ("OfflineBallRun-v0", "bcql", "hi_sus"),
    ("OfflineBallRun-v0", "cpq", "hi_sus"),
    ("OfflineCarRun-v0", "bcql", "hi_sus"),
    ("OfflineCarCircle-v0", "bcql", "hi_sus"),
    ("OfflineBallCircle-v0", "cpq", "hi_sus"),
    ("OfflineBallCircle-v0", "bcql", "hi_scope"),
]
ARMS = ("single_a", "single_b", "union", "vector")


def pos_of(obs, task):
    for pref, (sl, scale) in POS.items():
        if task.startswith(pref):
            return np.linalg.norm(np.asarray(obs)[..., sl], axis=-1) / scale
    raise ValueError(f"no pinned position slice for {task}")


def channels_scope(data, task):
    sp = speeds_of(data["observations"], task)
    s_hi = np.percentile(sp, Q_HI)
    s_sc = np.percentile(sp, Q_SCOPE)
    pr = pos_of(data["observations"], task)
    r_sc = np.percentile(pr, Q_SCOPE)
    c_a = (sp > s_hi).astype(np.float32)
    c_b = ((sp > s_sc) & (pr > r_sc)).astype(np.float32)
    both = float(np.mean((c_a > 0) & (c_b > 0)))
    return c_a, c_b, dict(s_hi=float(s_hi), s_scope=float(s_sc),
                          r_scope=float(r_sc),
                          frac_a_only=float(np.mean((c_a > 0) & (c_b == 0))),
                          frac_b_only=float(np.mean((c_b > 0) & (c_a == 0))),
                          frac_both=both)


def build_arm(learner, env, cfg, arm):
    sd = env.observation_space.shape[0]
    ad = env.action_space.shape[0]
    ma = env.action_space.high[0]
    vec = (arm == "vector")
    if learner == "bcql":
        cls = VectorBCQL if vec else BCQL
        kw = dict(state_dim=sd, action_dim=ad, max_action=ma,
                  a_hidden_sizes=cfg["a_hidden_sizes"],
                  c_hidden_sizes=cfg["c_hidden_sizes"],
                  vae_hidden_sizes=cfg["vae_hidden_sizes"],
                  sample_action_num=cfg["sample_action_num"],
                  PID=cfg["PID"], gamma=cfg["gamma"], tau=cfg["tau"],
                  lmbda=cfg["lmbda"], beta=cfg["beta"], phi=cfg["phi"],
                  num_q=cfg["num_q"], num_qc=cfg["num_qc"],
                  cost_limit=cfg["cost_limit"],
                  episode_len=cfg["episode_len"], device=cfg["device"])
        if vec:
            kw["n_costs"] = 2
        model = cls(**kw)
        trainer = BCQLTrainer(model, env, logger=_NoLogger(),
                              actor_lr=cfg["actor_lr"],
                              critic_lr=cfg["critic_lr"],
                              vae_lr=cfg["vae_lr"],
                              reward_scale=cfg["reward_scale"],
                              cost_scale=cfg["cost_scale"],
                              device=cfg["device"])
    else:
        cls = VectorCPQ if vec else CPQ
        kw = dict(state_dim=sd, action_dim=ad, max_action=ma,
                  a_hidden_sizes=cfg["a_hidden_sizes"],
                  c_hidden_sizes=cfg["c_hidden_sizes"],
                  vae_hidden_sizes=cfg["vae_hidden_sizes"],
                  sample_action_num=cfg["sample_action_num"],
                  gamma=cfg["gamma"], tau=cfg["tau"], beta=cfg["beta"],
                  num_q=cfg["num_q"], num_qc=cfg["num_qc"],
                  qc_scalar=cfg["qc_scalar"], cost_limit=cfg["cost_limit"],
                  episode_len=cfg["episode_len"], device=cfg["device"])
        if vec:
            kw["n_costs"] = 2
        model = cls(**kw)
        trainer = CPQTrainer(model, env, logger=_NoLogger(),
                             actor_lr=cfg["actor_lr"],
                             critic_lr=cfg["critic_lr"],
                             alpha_lr=cfg["alpha_lr"], vae_lr=cfg["vae_lr"],
                             reward_scale=cfg["reward_scale"],
                             cost_scale=cfg["cost_scale"],
                             device=cfg["device"])
    return model, trainer


def cp_upper(v, n, conf):
    from scipy.stats import beta as _b
    return float(_b.ppf(conf, v + 1, n - v)) if v < n else 1.0


@torch.no_grad()
def eval_arm(model, env, task, crossing, spec, episodes, reward_scale,
             learner):
    rows = []
    for _ in range(episodes):
        obs, _info = env.reset()
        ret, c_nat, c_a, c_b, run, steps = 0.0, 0.0, 0.0, 0.0, 0, 0
        for _t in range(model.episode_len):
            sp = float(speeds_of(np.asarray(obs), task))
            if crossing == "hi_sus":
                c_a += float(sp > spec["s_hi"])
                run = run + 1 if sp > spec["s_lo"] else 0
                c_b += float(run >= K_SUSTAIN)
            else:
                pr = float(pos_of(np.asarray(obs), task))
                c_a += float(sp > spec["s_hi"])
                c_b += float(sp > spec["s_scope"] and pr > spec["r_scope"])
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
        rows.append(dict(ret=ret / reward_scale, cost_a=c_a, cost_b=c_b,
                         cost_native=c_nat, len=steps))
    n = len(rows)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    certs = {}
    for ch in ("a", "b"):
        v = int(sum(r[f"cost_{ch}"] > COST_LIMIT for r in rows))
        up = cp_upper(v, n, 1 - DELTA_EV / 2)
        H = model.episode_len
        hoeff = agg[f"cost_{ch}"] + H * np.sqrt(
            np.log(2 * 2 / DELTA_EV) / (2 * n))
        certs[ch] = dict(viol=v, n=n, cp_upper=round(up, 4),
                         ships=up <= SHIP_RATE,
                         hoeffding_expcost_upper=round(float(hoeff), 2),
                         hoeffding_vacuous=bool(hoeff > COST_LIMIT))
        agg[f"viol_{ch}"] = v / n
    return agg, certs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--update_steps", type=int, default=100000)
    ap.add_argument("--eval_episodes", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out", default="results/dsrl/vector")
    args = ap.parse_args()
    task, learner, crossing = CELLS[args.cell]

    cfg0 = make_cfg(learner, task, COST_LIMIT, args.seed, args.device,
                    args.update_steps, args.eval_episodes)
    env0 = gym.make(task)
    data0 = env0.get_dataset()
    data0 = env0.pre_process_data(data0, cfg0["outliers_percent"],
                                  cfg0["noise_scale"],
                                  cfg0["inpaint_ranges"],
                                  cfg0["epsilon"], cfg0["density"])
    if crossing == "hi_sus":
        c_a, c_b, wit = channels(data0, task)
        spec = dict(s_lo=wit["s_lo"], s_hi=wit["s_hi"])
    else:
        c_a, c_b, wit = channels_scope(data0, task)
        spec = dict(s_hi=wit["s_hi"], s_scope=wit["s_scope"],
                    r_scope=wit["r_scope"])
    del env0, data0

    out = dict(task=task, learner=learner, crossing=crossing,
               seed=args.seed, witness=wit, arms={})
    for arm in ARMS:
        cfg = make_cfg(learner, task, COST_LIMIT, args.seed, args.device,
                       args.update_steps, args.eval_episodes)
        seed_all(cfg["seed"])
        env = gym.make(task)
        data = env.get_dataset()
        env.set_target_cost(cfg["cost_limit"])
        max_r = float(env.max_episode_reward)
        min_r = float(env.min_episode_reward)
        data = env.pre_process_data(data, cfg["outliers_percent"],
                                    cfg["noise_scale"],
                                    cfg["inpaint_ranges"], cfg["epsilon"],
                                    cfg["density"])
        data = dict(data)
        if arm == "single_a":
            data["costs"] = np.asarray(c_a, dtype=np.float32)
        elif arm == "single_b":
            data["costs"] = np.asarray(c_b, dtype=np.float32)
        elif arm == "union":
            data["costs"] = np.maximum(c_a, c_b).astype(np.float32)
        else:
            data["costs"] = np.stack([c_a, c_b], 1).astype(np.float32)
        env = wrap_env(env, reward_scale=cfg["reward_scale"])
        env = OfflineEnvWrapper(env)
        dataset = TransitionDataset(data,
                                    reward_scale=cfg["reward_scale"],
                                    cost_scale=cfg["cost_scale"],
                                    state_init=False)
        loader = DataLoader(dataset, batch_size=cfg["batch_size"],
                            pin_memory=(args.num_workers > 0),
                            num_workers=args.num_workers)
        model, trainer = build_arm(learner, env, cfg, arm)
        it = iter(loader)
        t0 = time.time()
        for _step in range(cfg["update_steps"]):
            obs, next_obs, act, rew, cost, done = [
                b.to(args.device) for b in next(it)]
            trainer.train_one_step(obs, next_obs, act, rew, cost, done)
        agg, certs = eval_arm(model, env, task, crossing, spec,
                              cfg["eval_episodes"], cfg["reward_scale"],
                              learner)
        agg.update(train_sec=round(time.time() - t0, 1),
                   ret_norm=(agg["ret"] - min_r) / (max_r - min_r)
                   if max_r > min_r else None,
                   cost_a_norm=agg["cost_a"] / COST_LIMIT,
                   cost_b_norm=agg["cost_b"] / COST_LIMIT)
        out["arms"][arm] = dict(agg=agg, certs=certs)
        print(f"[{task}/{learner}/{crossing}] seed {args.seed} arm {arm}: "
              f"ret_norm={agg['ret_norm']:.3f} "
              f"cA={agg['cost_a_norm']:.2f} cB={agg['cost_b_norm']:.2f} "
              f"violA={agg['viol_a']:.2f} violB={agg['viol_b']:.2f}",
              flush=True)
    os.makedirs(args.out, exist_ok=True)
    aid = os.environ.get("SLURM_ARRAY_JOB_ID", "local")
    tid = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    path = os.path.join(args.out,
                        f"vector_{args.cell}_{args.seed}_{aid}_{tid}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
