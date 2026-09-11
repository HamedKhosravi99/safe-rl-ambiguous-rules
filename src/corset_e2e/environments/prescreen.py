"""WP-F.4: frozen prescreen. A task is admitted only if all four criteria pass.

  1 own-reading feasibility   J_c1(pi_1) <= d and J_c2(pi_2) <= d
  2 bidirectional crossing    J_c2(pi_1) >= 3d and J_c1(pi_2) >= 3d, judged on
                              the LOWER confidence bound, not the point estimate
  3 joint feasibility         max_k J_ck(pi_12) <= d
  4 nontrivial return         J_r(pi_12) >= J_r(pi_0) + 0.10 (J_r(unc) - J_r(pi_0))

Criteria are evaluated on independent simulator episodes with seeds disjoint
from training. Every configuration is written to the ledger with its reason
code, accepted or not (plan F.7).

Run one config:  python -m corset_e2e.environments.prescreen --index 7
Probe geometry:  python -m corset_e2e.environments.prescreen --probe
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.environments.families import (  # noqa: E402
    HORIZON, Config, candidate_configs, make, region_of)

OUT = os.path.join(ROOT, "results/e2e", "prescreen")
CROSS_MULT = 3.0
EVAL_EPISODES = 300


def boot_lo(x: np.ndarray, n=2000, seed=0, q=0.05) -> float:
    """One-sided lower confidence bound on a mean, by episode bootstrap."""
    rng = np.random.default_rng(seed)
    m = [float(rng.choice(x, size=len(x), replace=True).mean()) for _ in range(n)]
    return float(np.quantile(m, q))


def fallback_eval(cfg: Config, episodes=EVAL_EPISODES, seed0=50_000):
    rs, cs = [], []
    for e in range(episodes):
        env = make(cfg, seed0 + e)
        s = env.reset(seed0 + e)
        R, C, done = 0.0, np.zeros(env.n_costs), False
        while not done:
            s, r, c, done, _ = env.step(env.fallback_action(s))
            R += r; C += c
        rs.append(R); cs.append(C / HORIZON)
    return float(np.mean(rs)), np.asarray(cs)


def scripted_probe(cfg: Config, episodes=60):
    """Cheap geometry check with hand-written policies; no learning."""
    fam = cfg.family
    out = {}
    if fam == "navigation":
        # upper corridor fast, lower corridor forceful, and a gentle traversal
        pols = {"upper_fast": lambda s, t: np.array([1.0, 0.55 if t < 12 else -0.25]),
                "lower_force": lambda s, t: np.array([1.0, -0.75 if t < 12 else 0.35]),
                "upper_gentle": lambda s, t: np.array([0.22, 0.16 if t < 20 else -0.06])}
    elif fam == "locomotion":
        pols = {"lean_gait": lambda s, t: np.array([0.9, 0.95]),
                "upright_hard": lambda s, t: np.array([1.0, 0.0]),
                "slow_upright": lambda s, t: np.array([0.15, 0.0])}
    else:
        pols = {"rul_only": None, "anom_only": None, "both": None}
    if fam == "maintenance":
        def mk(mode):
            def f(env, s):
                rul, anom, streak = s[0], s[1], s[2] * max(cfg.params["k"], 1)
                svc = False
                if mode in ("rul", "both") and rul < cfg.params["theta_R"] + 0.06:
                    svc = True
                if mode in ("anom", "both") and streak >= cfg.params["k"] - 1:
                    svc = True
                return np.array([1.0 if svc else -1.0])
            return f
        pols = {"rul_only": mk("rul"), "anom_only": mk("anom"), "both": mk("both")}
    for name, pol in pols.items():
        rs, cs = [], []
        for e in range(episodes):
            env = make(cfg, 900_000 + e)
            s = env.reset(900_000 + e)
            R, C, done, t = 0.0, np.zeros(env.n_costs), False, 0
            while not done:
                a = pol(env, s) if fam == "maintenance" else pol(s, t)
                s, r, c, done, _ = env.step(a)
                R += r; C += c; t += 1
            rs.append(R); cs.append(C / HORIZON)
        cs = np.asarray(cs)
        out[name] = dict(ret=round(float(np.mean(rs)), 3),
                         c1=round(float(cs[:, 0].mean()), 4),
                         c2=round(float(cs[:, 1].mean()), 4))
    return out


def run_config(idx: int, steps: int, device: str, seed: int = 0) -> dict:
    import torch
    from corset_e2e.learners.sac_lag import evaluate, train_online
    cfg = candidate_configs()[idx]
    ckpt_dir = os.path.join(ROOT, "results/e2e", "oracles")
    os.makedirs(ckpt_dir, exist_ok=True)
    d = cfg.budget
    limit = d * HORIZON                      # episodic violation budget
    env_fn = lambda s=0: make(cfg, s)
    oracles = {}
    for tag, mask in (("unc", [False, False]), ("p1", [True, False]),
                      ("p2", [False, True]), ("p12", [True, True])):
        ag = train_online(env_fn, mask, [limit, limit], steps=steps,
                          device=device, seed=seed)
        R, Cm, Call, _ = evaluate(ag, env_fn, episodes=EVAL_EPISODES)
        oracles[tag] = dict(ret=R, c=Cm.tolist(), per_ep=Call)
        # keep the oracle: it is the behaviour policy for the offline dataset
        torch.save(ag.actor.state_dict(),
                   os.path.join(ckpt_dir, f"cfg{idx:02d}_{tag}.pt"))
    Rfb, Cfb = fallback_eval(cfg)

    def lo(tag, ch):
        return boot_lo(oracles[tag]["per_ep"][:, ch], seed=idx * 10 + ch)

    c1_p1, c2_p2 = oracles["p1"]["c"][0], oracles["p2"]["c"][1]
    cross_21_lo = lo("p1", 1)                 # pi_1 judged on reading 2
    cross_12_lo = lo("p2", 0)
    joint = max(oracles["p12"]["c"])
    span = oracles["unc"]["ret"] - Rfb
    reason = "ACCEPT"
    if not (c1_p1 <= d and c2_p2 <= d):
        reason = "REJECT_OWN_FEASIBILITY"
    elif not (cross_21_lo >= CROSS_MULT * d and cross_12_lo >= CROSS_MULT * d):
        reason = ("REJECT_ONE_WAY_ONLY"
                  if max(cross_21_lo, cross_12_lo) >= CROSS_MULT * d
                  else "REJECT_CROSS_MARGIN")
    elif joint > d:
        reason = "REJECT_JOINT_FEASIBILITY"
    elif oracles["p12"]["ret"] < Rfb + 0.10 * span:
        reason = "REJECT_TRIVIAL_RETURN"
    rec = dict(index=idx, family=cfg.family, name=cfg.name, params=cfg.params,
               budget=d, steps=steps, seed=seed, reason=reason,
               accepted=reason == "ACCEPT",
               own=dict(c1_pi1=c1_p1, c2_pi2=c2_p2),
               cross=dict(c2_pi1=oracles["p1"]["c"][1], c2_pi1_lo=cross_21_lo,
                          c1_pi2=oracles["p2"]["c"][0], c1_pi2_lo=cross_12_lo,
                          threshold=CROSS_MULT * d),
               joint=dict(max_cost=joint, c=oracles["p12"]["c"],
                          ret=oracles["p12"]["ret"]),
               ret=dict(unc=oracles["unc"]["ret"], fallback=Rfb,
                        span=span, needed=Rfb + 0.10 * span),
               fallback_cost=Cfb.mean(0).tolist())
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(os.path.join(OUT, f"cfg_{idx:02d}.json"), "w"), indent=1)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=-1)
    ap.add_argument("--steps", type=int, default=120_000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.probe:
        for i, cfg in enumerate(candidate_configs()):
            if i % 10 not in (0, 4, 9):
                continue
            print(f"[{i:2d}] {cfg.name:8s} {scripted_probe(cfg)}")
        return
    rec = run_config(a.index, a.steps, a.device)
    print(json.dumps({k: v for k, v in rec.items() if k != "params"}, indent=1))


if __name__ == "__main__":
    main()
