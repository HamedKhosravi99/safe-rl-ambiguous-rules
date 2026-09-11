#!/usr/bin/env python
# ---------------------------------------------------------------------------
# SA-ORL vs single-reading on the STANDARD DSRL/OSRL offline-safe-RL benchmark
# (Liu et al. 2023, arXiv:2306.09303 -- the benchmark recent papers report on).
#
# WHY THIS EXISTS
# ---------------
# The kill test in saorl.experiments runs on our own domains. Reviewers also want
# OURS-vs-SOTA on the *standard* benchmark. DSRL/OSRL fix a single cost threshold
# `target_cost` per run; OSRL's published learners (BCQL=BCQ-Lagrangian, CPQ, ...)
# are *real SOTA* offline-safe-RL methods. Our thesis: when the safety constraint
# is an AMBIGUOUS natural-language rule, that single threshold is itself ONE
# interpretation. We model the audited retained ambiguity set U over the canonical
# OSRL thresholds {10,20,40} and compare, for an identical SOTA inner learner L on
# a real DSRL task:
#
#   single  (standard practice): adopt one plausible reading c* = max(U); train L
#           to target c*; deploy.                        <- "SOTA, single reading"
#   SA-ORL  (ours): honor the WHOLE retained set; train L to the strictest retained
#           reading c_lo = min(U); deploy.               <- "SOTA wrapped in SA-ORL"
#
# Both are the SAME learner; the only difference is the SA-ORL set-vs-single rule,
# which is exactly the paper's contribution (the SET, not the learner).
#
# METRIC (DSRL-normalized). normalized_reward = (R - R_min)/(R_max - R_min); a
# reading c is satisfied iff normalized_cost_c = raw_cost / c <= 1. The kill signal:
# the single-reading policy looks safe against its OWN c* but its WORST retained
# normalized cost  raw_cost/min(U) > 1  (it silently violates the strict reading),
# while SA-ORL keeps worst-retained cost <= 1 -- at comparable reward.
#
# Faithfulness: per-task hyper-parameters are taken verbatim from OSRL's own
# published default configs (examples/configs/*_configs.py in the cloned repo);
# we override ONLY {cost_limit, seed, device, update_steps, eval_episodes}.
#
#   python experiments/dsrl_learners/dsrl_sweep.py --tasks OfflineCarCircle-v0 --learners bcql,cpq \
#       --seeds 0,1,2 --ambiguity 10,20,40 --update_steps 100000 --device cuda \
#       --out results
#   python experiments/dsrl_learners/dsrl_sweep.py --smoke      # ~2 min CPU end-to-end correctness check
# ---------------------------------------------------------------------------
import os
import sys
import json
import time
import argparse
import datetime
from dataclasses import asdict

import numpy as np
import torch

# OSRL example configs live in the cloned repo (not the pip package); add to path.
OSRL_REPO = os.environ.get("OSRL_REPO", os.path.expanduser("~/scratch/OSRL"))
if OSRL_REPO not in sys.path:
    sys.path.insert(0, OSRL_REPO)

import bullet_safety_gym  # noqa: F401  (registers Bullet* offline tasks)
import dsrl  # noqa: F401            (registers Offline* tasks + dataset urls)
import gymnasium as gym
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from torch.utils.data import DataLoader

from osrl.common import TransitionDataset, SequenceDataset
from osrl.common.exp_util import seed_all
from osrl.algorithms import (BCQL, BCQLTrainer, CPQ, CPQTrainer,
                             COptiDICE, COptiDICETrainer, CDT, CDTTrainer)

from examples.configs.bcql_configs import BCQL_DEFAULT_CONFIG
from examples.configs.cpq_configs import CPQ_DEFAULT_CONFIG
from examples.configs.coptidice_configs import COptiDICE_DEFAULT_CONFIG
from examples.configs.cdt_configs import CDT_DEFAULT_CONFIG

CONFIG = {"bcql": BCQL_DEFAULT_CONFIG, "cpq": CPQ_DEFAULT_CONFIG,
          "coptidice": COptiDICE_DEFAULT_CONFIG, "cdt": CDT_DEFAULT_CONFIG}
LEARNER_LONG = {"bcql": "BCQ-Lagrangian", "cpq": "CPQ", "coptidice": "COptiDICE",
                "cdt": "CDT", "caps": "CAPS"}


class _NoLogger:
    """OSRL trainers expect a logger object; we don't want wandb. Any method call
    becomes a no-op (trainers only *store* scalars during train/eval)."""
    def __getattr__(self, _name):
        def _noop(*a, **k):
            return None
        return _noop


def make_cfg(learner, task, cost_limit, seed, device, update_steps, eval_episodes):
    cfg = asdict(CONFIG[learner][task]())          # OSRL per-task published defaults
    cfg["cost_limit"] = int(cost_limit)            # <-- the only constraint knob
    cfg["seed"] = int(seed)
    cfg["device"] = device
    if update_steps is not None:
        cfg["update_steps"] = int(update_steps)
    if eval_episodes is not None:
        cfg["eval_episodes"] = int(eval_episodes)
    return cfg


def build(learner, env, cfg, dataset=None):
    sd = env.observation_space.shape[0]
    ad = env.action_space.shape[0]
    ma = env.action_space.high[0]
    if learner == "bcql":
        model = BCQL(state_dim=sd, action_dim=ad, max_action=ma,
                     a_hidden_sizes=cfg["a_hidden_sizes"], c_hidden_sizes=cfg["c_hidden_sizes"],
                     vae_hidden_sizes=cfg["vae_hidden_sizes"], sample_action_num=cfg["sample_action_num"],
                     PID=cfg["PID"], gamma=cfg["gamma"], tau=cfg["tau"], lmbda=cfg["lmbda"],
                     beta=cfg["beta"], phi=cfg["phi"], num_q=cfg["num_q"], num_qc=cfg["num_qc"],
                     cost_limit=cfg["cost_limit"], episode_len=cfg["episode_len"], device=cfg["device"])
        trainer = BCQLTrainer(model, env, logger=_NoLogger(),
                              actor_lr=cfg["actor_lr"], critic_lr=cfg["critic_lr"], vae_lr=cfg["vae_lr"],
                              reward_scale=cfg["reward_scale"], cost_scale=cfg["cost_scale"], device=cfg["device"])
    elif learner == "cpq":
        model = CPQ(state_dim=sd, action_dim=ad, max_action=ma,
                    a_hidden_sizes=cfg["a_hidden_sizes"], c_hidden_sizes=cfg["c_hidden_sizes"],
                    vae_hidden_sizes=cfg["vae_hidden_sizes"], sample_action_num=cfg["sample_action_num"],
                    gamma=cfg["gamma"], tau=cfg["tau"], beta=cfg["beta"], num_q=cfg["num_q"],
                    num_qc=cfg["num_qc"], qc_scalar=cfg["qc_scalar"], cost_limit=cfg["cost_limit"],
                    episode_len=cfg["episode_len"], device=cfg["device"])
        trainer = CPQTrainer(model, env, logger=_NoLogger(),
                             actor_lr=cfg["actor_lr"], critic_lr=cfg["critic_lr"], alpha_lr=cfg["alpha_lr"],
                             vae_lr=cfg["vae_lr"], reward_scale=cfg["reward_scale"],
                             cost_scale=cfg["cost_scale"], device=cfg["device"])
    elif learner == "coptidice":
        # COptiDICE needs three dataset-derived inputs (initial-state proportion and
        # per-dim obs/action stds), obtained from the SAME TransitionDataset the
        # other learners use via get_dataset_states() -- see OSRL train_coptidice.py.
        if dataset is None:
            raise ValueError("coptidice requires the dataset for init stats")
        init_s_prop, obs_std, act_std = dataset.get_dataset_states()
        model = COptiDICE(state_dim=sd, action_dim=ad, max_action=ma,
                          f_type=cfg["f_type"], init_state_propotion=init_s_prop,
                          observations_std=obs_std, actions_std=act_std,
                          a_hidden_sizes=cfg["a_hidden_sizes"], c_hidden_sizes=cfg["c_hidden_sizes"],
                          gamma=cfg["gamma"], alpha=cfg["alpha"], cost_ub_epsilon=cfg["cost_ub_epsilon"],
                          num_nu=cfg["num_nu"], num_chi=cfg["num_chi"], cost_limit=cfg["cost_limit"],
                          episode_len=cfg["episode_len"], device=cfg["device"])
        trainer = COptiDICETrainer(model, env, logger=_NoLogger(),
                                   actor_lr=cfg["actor_lr"], critic_lr=cfg["critic_lr"],
                                   scalar_lr=cfg["scalar_lr"], reward_scale=cfg["reward_scale"],
                                   cost_scale=cfg["cost_scale"], device=cfg["device"])
    else:
        raise ValueError(f"unknown learner {learner}")
    return model, trainer


def train_one(task, learner, cost_limit, seed, device, update_steps, eval_episodes,
              num_workers, probe_every=0):
    """Train one OSRL learner to one target cost on one DSRL task; return raw eval.

    If probe_every>0, periodically evaluate during training and record a
    (step, raw_return, raw_cost) convergence trajectory -- used by the timing/
    convergence probe to size update_steps for the full sweep from MEASURED
    curves rather than a guess."""
    cfg = make_cfg(learner, task, cost_limit, seed, device, update_steps, eval_episodes)
    seed_all(cfg["seed"])
    if device == "cpu":
        torch.set_num_threads(cfg.get("threads", 4))

    env = gym.make(task)
    data = env.get_dataset()                       # offline; reads scratch cache
    env.set_target_cost(cfg["cost_limit"])
    # normalization references are fixed per task -- capture before wrapping
    max_r = float(env.max_episode_reward)
    min_r = float(env.min_episode_reward)
    data = env.pre_process_data(data, cfg["outliers_percent"], cfg["noise_scale"],
                                cfg["inpaint_ranges"], cfg["epsilon"], cfg["density"])
    env = wrap_env(env, reward_scale=cfg["reward_scale"])
    env = OfflineEnvWrapper(env)

    # Build the dataset BEFORE the model: COptiDICE reads dataset-derived init
    # stats (get_dataset_states) at construction; bcql/cpq ignore the dataset arg.
    # COptiDICE alone needs state_init=True: it populates an "is_init" flag that
    # get_dataset_states() reads AND makes each batch a 7-tuple (..., is_init) that
    # COptiDICETrainer.train_one_step unpacks. bcql/cpq use the 6-tuple (no is_init).
    dataset = TransitionDataset(data, reward_scale=cfg["reward_scale"], cost_scale=cfg["cost_scale"],
                                state_init=(learner == "coptidice"))
    loader = DataLoader(dataset, batch_size=cfg["batch_size"],
                        pin_memory=(num_workers > 0), num_workers=num_workers)
    model, trainer = build(learner, env, cfg, dataset=dataset)
    it = iter(loader)
    trajectory = []
    t0 = time.time()
    for step in range(cfg["update_steps"]):
        batch = next(it)
        if learner == "coptidice":
            # COptiDICE's update consumes the whole transition batch at once.
            trainer.train_one_step([b.to(device) for b in batch])
        else:
            obs, next_obs, act, rew, cost, done = [b.to(device) for b in batch]
            trainer.train_one_step(obs, next_obs, act, rew, cost, done)
        if probe_every and (step + 1) % probe_every == 0:
            pr, pc, _pl = trainer.evaluate(min(eval_episodes or 10, 10))
            elapsed = time.time() - t0
            trajectory.append({"step": step + 1, "raw_return": float(pr),
                               "raw_cost": float(pc), "sec": round(elapsed, 1)})
            print(f"    [probe] step={step+1:>6} R={pr:7.1f} cost={pc:7.2f} "
                  f"({elapsed:.0f}s, {elapsed/(step+1)*1000:.1f}ms/step)", flush=True)
    ret, cost, length = trainer.evaluate(cfg["eval_episodes"])
    out = {
        "raw_return": float(ret), "raw_cost": float(cost), "ep_length": float(length),
        "episode_len": int(cfg["episode_len"]), "update_steps": int(cfg["update_steps"]),
        "eval_episodes": int(cfg["eval_episodes"]), "max_epi_r": max_r, "min_epi_r": min_r,
        "train_sec": round(time.time() - t0, 1),
    }
    if trajectory:
        out["trajectory"] = trajectory
    return out


def train_cdt(task, seed, U, device, update_steps, eval_episodes, num_workers,
              batch_size=None):
    """Train ONE Constrained Decision Transformer on a DSRL task and read off BOTH
    kill-test readings from the SAME policy.

    CDT is a return/cost-conditioned sequence model: the cost threshold is supplied
    at INFERENCE via `target_cost`, not baked into training (unlike BCQL/CPQ/
    COptiDICE whose `cost_limit` enters the objective). The single-vs-SA-ORL contrast
    is therefore two *conditionings* of one trained policy -- single honors the
    loosest reading c*=max(U), SA-ORL the strictest c_lo=min(U) -- which is both more
    faithful to CDT and half the compute of training twice. Each cost target is
    paired with its OSRL-published target return (4/5 of our tasks publish costs
    {10,20,40} exactly; CarCircle publishes {10,20,50}, so c*=40 borrows the
    nearest published cost's return). Everything else is verbatim from OSRL's
    train_cdt.py; we override only {seed, device, update_steps, eval_episodes,
    batch_size}. Returns {"single": raw_eval, "saorl": raw_eval}."""
    cfg = asdict(CDT_DEFAULT_CONFIG[task]())
    cfg["seed"] = int(seed)
    cfg["device"] = device
    if update_steps is not None:
        cfg["update_steps"] = int(update_steps)
    if eval_episodes is not None:
        cfg["eval_episodes"] = int(eval_episodes)
    if batch_size is not None:
        cfg["batch_size"] = int(batch_size)
    seed_all(cfg["seed"])
    if device == "cpu":
        torch.set_num_threads(cfg.get("threads", 4))

    env = gym.make(task)
    data = env.get_dataset()
    env.set_target_cost(min(U))            # display only; eval passes target_cost explicitly
    max_r = float(env.max_episode_reward)
    min_r = float(env.min_episode_reward)
    # density handling mirrors train_cdt.py; all five of our tasks use density=1.0.
    cbins = rbins = max_npb = min_npb = None
    if cfg["density"] != 1.0:
        from dsrl.infos import DENSITY_CFG
        dcfg = DENSITY_CFG[task + "_density" + str(cfg["density"])]
        cbins, rbins = dcfg["cbins"], dcfg["rbins"]
        max_npb, min_npb = dcfg["max_npb"], dcfg["min_npb"]
    data = env.pre_process_data(data, cfg["outliers_percent"], cfg["noise_scale"],
                                cfg["inpaint_ranges"], cfg["epsilon"], cfg["density"],
                                cbins=cbins, rbins=rbins, max_npb=max_npb, min_npb=min_npb)
    env = wrap_env(env, reward_scale=cfg["reward_scale"])
    env = OfflineEnvWrapper(env)

    model = CDT(state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                max_action=env.action_space.high[0],
                embedding_dim=cfg["embedding_dim"], seq_len=cfg["seq_len"],
                episode_len=cfg["episode_len"], num_layers=cfg["num_layers"],
                num_heads=cfg["num_heads"], attention_dropout=cfg["attention_dropout"],
                residual_dropout=cfg["residual_dropout"], embedding_dropout=cfg["embedding_dropout"],
                time_emb=cfg["time_emb"], use_rew=cfg["use_rew"], use_cost=cfg["use_cost"],
                cost_transform=cfg["cost_transform"], add_cost_feat=cfg["add_cost_feat"],
                mul_cost_feat=cfg["mul_cost_feat"], cat_cost_feat=cfg["cat_cost_feat"],
                action_head_layers=cfg["action_head_layers"], cost_prefix=cfg["cost_prefix"],
                stochastic=cfg["stochastic"], init_temperature=cfg["init_temperature"],
                target_entropy=-env.action_space.shape[0]).to(device)
    trainer = CDTTrainer(model, env, logger=_NoLogger(),
                         learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
                         betas=cfg["betas"], clip_grad=cfg["clip_grad"],
                         lr_warmup_steps=cfg["lr_warmup_steps"], reward_scale=cfg["reward_scale"],
                         cost_scale=cfg["cost_scale"], loss_cost_weight=cfg["loss_cost_weight"],
                         loss_state_weight=cfg["loss_state_weight"], cost_reverse=cfg["cost_reverse"],
                         no_entropy=cfg["no_entropy"], device=device)

    # cost_transform matches the eval-time convention; linear is the published default.
    ct = (lambda x: 70 - x) if cfg["linear"] else (lambda x: 1 / (x + 10))
    dataset = SequenceDataset(
        data, seq_len=cfg["seq_len"], reward_scale=cfg["reward_scale"], cost_scale=cfg["cost_scale"],
        deg=cfg["deg"], pf_sample=cfg["pf_sample"], max_rew_decrease=cfg["max_rew_decrease"],
        beta=cfg["beta"], augment_percent=cfg["augment_percent"], cost_reverse=cfg["cost_reverse"],
        max_reward=cfg["max_reward"], min_reward=cfg["min_reward"], pf_only=cfg["pf_only"],
        rmin=cfg["rmin"], cost_bins=cfg["cost_bins"], npb=cfg["npb"], cost_sample=cfg["cost_sample"],
        cost_transform=ct, start_sampling=cfg["start_sampling"], prob=cfg["prob"],
        random_aug=cfg["random_aug"], aug_rmin=cfg["aug_rmin"], aug_rmax=cfg["aug_rmax"],
        aug_cmin=cfg["aug_cmin"], aug_cmax=cfg["aug_cmax"], cgap=cfg["cgap"], rstd=cfg["rstd"],
        cstd=cfg["cstd"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"],
                        pin_memory=(num_workers > 0), num_workers=num_workers)
    it = iter(loader)
    t0 = time.time()
    for _step in range(cfg["update_steps"]):
        batch = next(it)
        states, actions, returns, costs_return, time_steps, mask, episode_cost, costs = [
            b.to(device) for b in batch]
        trainer.train_one_step(states, actions, returns, costs_return, time_steps, mask,
                               episode_cost, costs)
    train_sec = round(time.time() - t0, 1)

    # one model, two readings: target_return = published return paired with each cost
    # (nearest published cost when the exact threshold isn't in the schedule).
    pairs = {int(c): float(r) for (r, c) in cfg["target_returns"]}
    def ret_for(cost):
        return pairs[cost] if cost in pairs else pairs[min(pairs, key=lambda k: abs(k - cost))]
    out = {}
    for reading, c in (("single", max(U)), ("saorl", min(U))):
        tr = ret_for(c)
        # exactly OSRL train_cdt.py's eval scaling (cost_reverse=False for all our tasks)
        if cfg["cost_reverse"]:
            ret, cost, length = trainer.evaluate(cfg["eval_episodes"], tr * cfg["reward_scale"],
                                                 (cfg["episode_len"] - c) * cfg["cost_scale"])
        else:
            ret, cost, length = trainer.evaluate(cfg["eval_episodes"], tr * cfg["reward_scale"],
                                                 c * cfg["cost_scale"])
        out[reading] = {
            "raw_return": float(ret), "raw_cost": float(cost), "ep_length": float(length),
            "episode_len": int(cfg["episode_len"]), "update_steps": int(cfg["update_steps"]),
            "eval_episodes": int(cfg["eval_episodes"]), "max_epi_r": max_r, "min_epi_r": min_r,
            "train_sec": train_sec, "target_return": float(tr), "target_cost": int(c),
        }
    return out


def train_caps(task, seed, U, device, update_steps, eval_episodes, num_workers,
               batch_size=None, num_heads=5):
    """Train ONE CAPS multi-head IQL model (Chemingui et al., AAAI 2025) on a DSRL
    task and read off BOTH kill-test readings from the SAME policy.

    CAPS (Constraint-Adaptive Policy Switching) trains K actor heads that span the
    reward<->cost trade-off, then at INFERENCE selects, per state, the head whose
    estimated cost-Q stays under a supplied `cost_limit` while maximizing reward-Q
    (trainer.evaluate_switch -> rollout_switch -> select_head_q). The cost threshold
    is thus a pure inference-time knob -- exactly the property the kill test needs --
    so single (c*=max(U)) and SA-ORL (c_lo=min(U)) are two switching evaluations of
    ONE trained model, like CDT (more faithful to CAPS, half the compute of training
    twice). Hyper-parameters are verbatim from CAPS's published per-task config
    (examples/configs/capsiql_configs.py in the cloned repo); we override only
    {seed, device, update_steps, eval_episodes, batch_size, num_heads}. num_heads
    defaults to 5 -- CAPS's own documented setting (the example in its train_capsiql.py
    is `--num_heads 5`); more heads only *helps* CAPS's switching, so this faithful
    default does not sandbag the baseline. Returns {"single": raw_eval, "saorl": raw_eval}.

    Integration note: CapsIQL/CapsIQLTrainer and the multi-head networks ship as an
    overlay (osrl/algorithms/capsiql.py, osrl/common/net_caps.py, capsiql_configs.py)
    added to the OSRL clone -- imported lazily here so the other learners never depend
    on it."""
    from osrl.algorithms.capsiql import CapsIQL, CapsIQLTrainer
    from examples.configs.capsiql_configs import CapsIQL_DEFAULT_CONFIG

    cfg = asdict(CapsIQL_DEFAULT_CONFIG[task]())
    cfg["seed"] = int(seed)
    cfg["device"] = device
    if update_steps is not None:
        cfg["update_steps"] = int(update_steps)
    if eval_episodes is not None:
        cfg["eval_episodes"] = int(eval_episodes)
    if batch_size is not None:
        cfg["batch_size"] = int(batch_size)
    cfg["num_heads"] = int(num_heads)
    seed_all(cfg["seed"])
    if device == "cpu":
        torch.set_num_threads(cfg.get("threads", 4))

    env = gym.make(task)
    data = env.get_dataset()
    env.set_target_cost(min(U))            # display only; eval passes cost_limit explicitly
    max_r = float(env.max_episode_reward)
    min_r = float(env.min_episode_reward)
    # density handling mirrors train_capsiql.py; all five of our tasks use density=1.0.
    cbins = rbins = max_npb = min_npb = None
    if cfg["density"] != 1.0:
        from dsrl.infos import DENSITY_CFG
        dcfg = DENSITY_CFG[task + "_density" + str(cfg["density"])]
        cbins, rbins = dcfg["cbins"], dcfg["rbins"]
        max_npb, min_npb = dcfg["max_npb"], dcfg["min_npb"]
    data = env.pre_process_data(data, cfg["outliers_percent"], cfg["noise_scale"],
                                cfg["inpaint_ranges"], cfg["epsilon"], cfg["density"],
                                cbins=cbins, rbins=rbins, max_npb=max_npb, min_npb=min_npb)
    env = wrap_env(env, reward_scale=cfg["reward_scale"])
    env = OfflineEnvWrapper(env)

    model = CapsIQL(state_dim=env.observation_space.shape[0],
                    action_dim=env.action_space.shape[0],
                    max_action=env.action_space.high[0],
                    num_heads=cfg["num_heads"], iql_deterministic=cfg["iql_deterministic"],
                    hidden_dim=cfg["hidden_dim"], iql_tau=cfg["iql_tau"],
                    iql_tau_cost=cfg["iql_tau_cost"], beta=cfg["beta"],
                    beta_cost=cfg["beta_cost"], tau=cfg["tau"], gamma=cfg["gamma"],
                    episode_len=cfg["episode_len"], device=device)
    model.to(device)
    trainer = CapsIQLTrainer(model, env, logger=_NoLogger(),
                             actor_lr=cfg["actor_lr"], q_lr=cfg["q_lr"],
                             value_lr=cfg["value_lr"], max_steps=cfg["max_steps"],
                             reward_scale=cfg["reward_scale"], cost_scale=cfg["cost_scale"],
                             device=device)

    dataset = TransitionDataset(data, reward_scale=cfg["reward_scale"],
                                cost_scale=cfg["cost_scale"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"],
                        pin_memory=(num_workers > 0), num_workers=num_workers)
    it = iter(loader)
    t0 = time.time()
    for step in range(cfg["update_steps"]):
        batch = next(it)
        obs, next_obs, act, rew, cost, done = [b.to(device) for b in batch]
        trainer.train_one_step(obs, next_obs, act, rew, cost, done, step)
    train_sec = round(time.time() - t0, 1)

    # one model, two readings: constraint-adaptive switching at each retained threshold.
    out = {}
    for reading, c in (("single", max(U)), ("saorl", min(U))):
        env.set_target_cost(c)
        ret, cost, length = trainer.evaluate_switch(c, cfg["eval_episodes"])
        out[reading] = {
            "raw_return": float(ret), "raw_cost": float(cost), "ep_length": float(length),
            "episode_len": int(cfg["episode_len"]), "update_steps": int(cfg["update_steps"]),
            "eval_episodes": int(cfg["eval_episodes"]), "max_epi_r": max_r, "min_epi_r": min_r,
            "train_sec": train_sec, "target_cost": int(c), "num_heads": int(cfg["num_heads"]),
        }
    return out


def normalize(raw, U):
    """Attach DSRL-normalized reward and per-reading + worst-case normalized cost."""
    R, C = raw["raw_return"], raw["raw_cost"]
    nr = (R - raw["min_epi_r"]) / (raw["max_epi_r"] - raw["min_epi_r"] + 1e-12)
    per = {str(c): C / c for c in U}                       # normalized cost per reading
    worst = max(per.values())                              # = C / min(U)
    raw["normalized_reward"] = float(nr)
    raw["normalized_cost_per_reading"] = {k: float(v) for k, v in per.items()}
    raw["worst_retained_normalized_cost"] = float(worst)
    raw["satisfies_all_retained"] = bool(worst <= 1.0)
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="OfflineCarCircle-v0")
    ap.add_argument("--learners", default="bcql,cpq")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ambiguity", default="10,20,40",
                    help="retained set U of plausible cost thresholds (ascending)")
    ap.add_argument("--update_steps", type=int, default=None)
    ap.add_argument("--eval_episodes", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=None,
                    help="override config batch_size (e.g. shrink for CPU smoke runs)")
    ap.add_argument("--caps_num_heads", type=int, default=5,
                    help="CAPS policy heads (its documented default; more heads only helps CAPS)")
    ap.add_argument("--probe_every", type=int, default=0,
                    help="if >0, eval every N steps to log a convergence trajectory")
    ap.add_argument("--out", default="results")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end correctness run (overrides most flags)")
    args = ap.parse_args()

    if args.smoke:
        args.tasks = "OfflineCarCircle-v0"
        args.learners = "bcql,cpq"
        args.seeds = "0"
        args.update_steps = 400
        args.eval_episodes = 2
        args.device = "cpu"
        args.num_workers = 0

    if not torch.cuda.is_available() and args.device == "cuda":
        print("[warn] cuda not available -> falling back to cpu", flush=True)
        args.device = "cpu"

    tasks = [t for t in args.tasks.split(",") if t]
    learners = [l for l in args.learners.split(",") if l]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    U = sorted(int(x) for x in args.ambiguity.split(","))
    c_single, c_saorl = max(U), min(U)   # single = loosest plausible; SA-ORL = strictest

    print(f"[dsrl_sweep] device={args.device} tasks={tasks} learners={learners} "
          f"seeds={seeds} U={U} single(c*)={c_single} saorl(c_lo)={c_saorl} "
          f"steps={args.update_steps}", flush=True)

    runs = []
    for task in tasks:
        for learner in learners:
            for seed in seeds:
                # CDT is return/cost-conditioned: ONE trained model is read at two
                # cost conditionings (single c*=max(U), SA-ORL c_lo=min(U)) -- more
                # faithful to CDT and half the compute of training twice.
                if learner == "cdt":
                    print(f"[run] {task} | cdt | seed={seed} (1 model, 2 readings)", flush=True)
                    try:
                        raws = train_cdt(task, seed, U, args.device, args.update_steps,
                                         args.eval_episodes, args.num_workers, args.batch_size)
                        for reading, c in (("single", c_single), ("saorl", c_saorl)):
                            rec = normalize(raws[reading], U)
                            rec.update(task=task, learner="cdt", learner_long=LEARNER_LONG["cdt"],
                                       reading=reading, target_cost=c, seed=seed, U=U, status="ok")
                            print(f"   -> {reading}(c={c}): R={rec['raw_return']:.1f} "
                                  f"cost={rec['raw_cost']:.2f} nR={rec['normalized_reward']:.3f} "
                                  f"worstNC={rec['worst_retained_normalized_cost']:.3f} "
                                  f"safe_all={rec['satisfies_all_retained']}", flush=True)
                            runs.append(rec)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        for reading, c in (("single", c_single), ("saorl", c_saorl)):
                            runs.append(dict(task=task, learner="cdt", reading=reading,
                                             target_cost=c, seed=seed, U=U, status="error",
                                             error=repr(e)[:300]))
                    continue
                # CAPS is also one model read at two inference-time cost limits: train
                # K reward<->cost heads once, switch-evaluate at single c*=max(U) and
                # SA-ORL c_lo=min(U). Same one-model-two-readings logic as CDT.
                if learner == "caps":
                    print(f"[run] {task} | caps | seed={seed} (1 model, 2 readings)", flush=True)
                    try:
                        raws = train_caps(task, seed, U, args.device, args.update_steps,
                                          args.eval_episodes, args.num_workers, args.batch_size,
                                          num_heads=args.caps_num_heads)
                        for reading, c in (("single", c_single), ("saorl", c_saorl)):
                            rec = normalize(raws[reading], U)
                            rec.update(task=task, learner="caps", learner_long=LEARNER_LONG["caps"],
                                       reading=reading, target_cost=c, seed=seed, U=U, status="ok")
                            print(f"   -> {reading}(c={c}): R={rec['raw_return']:.1f} "
                                  f"cost={rec['raw_cost']:.2f} nR={rec['normalized_reward']:.3f} "
                                  f"worstNC={rec['worst_retained_normalized_cost']:.3f} "
                                  f"safe_all={rec['satisfies_all_retained']}", flush=True)
                            runs.append(rec)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        for reading, c in (("single", c_single), ("saorl", c_saorl)):
                            runs.append(dict(task=task, learner="caps", reading=reading,
                                             target_cost=c, seed=seed, U=U, status="error",
                                             error=repr(e)[:300]))
                    continue
                for reading, c in (("single", c_single), ("saorl", c_saorl)):
                    tag = f"{task} | {learner} | {reading}(c={c}) | seed={seed}"
                    print(f"[run] {tag}", flush=True)
                    try:
                        raw = train_one(task, learner, c, seed, args.device,
                                        args.update_steps, args.eval_episodes, args.num_workers,
                                        probe_every=args.probe_every)
                        rec = normalize(raw, U)
                        rec.update(task=task, learner=learner, learner_long=LEARNER_LONG[learner],
                                   reading=reading, target_cost=c, seed=seed, U=U, status="ok")
                        print(f"   -> R={rec['raw_return']:.1f} cost={rec['raw_cost']:.2f} "
                              f"nR={rec['normalized_reward']:.3f} "
                              f"worstNC={rec['worst_retained_normalized_cost']:.3f} "
                              f"safe_all={rec['satisfies_all_retained']} "
                              f"({rec['train_sec']}s)", flush=True)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        rec = dict(task=task, learner=learner, reading=reading, target_cost=c,
                                   seed=seed, U=U, status="error", error=repr(e)[:300])
                    runs.append(rec)

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(args.out, f"dsrl_sweep_{stamp}.json")
    blob = dict(created=stamp, device=args.device, tasks=tasks, learners=learners,
                seeds=seeds, U=U, c_single=c_single, c_saorl=c_saorl,
                update_steps=args.update_steps, runs=runs)
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"[dsrl_sweep] wrote {path}  ({sum(r.get('status')=='ok' for r in runs)}/{len(runs)} ok)",
          flush=True)


if __name__ == "__main__":
    main()
