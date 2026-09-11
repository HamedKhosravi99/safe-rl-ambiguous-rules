"""WP-F.5/F.6: offline datasets from the prescreen oracles, with support gates.

Behaviour mixture (frozen, plan F.6):
    25% reward-seeking (unconstrained oracle)
    20% psi_1-constrained        -> covers D_{2\\1}
    20% psi_2-constrained        -> covers D_{1\\2}
    20% jointly constrained      -> covers the robust path
    15% broad exploratory

A dataset is admitted only if it satisfies the support criteria: at least 5%
of transitions in each disagreement region, at least 10% jointly safe, and no
single behaviour component above 60%. A task that passes the simulator
geometry but fails support is labelled REJECT_OFFLINE_SUPPORT and is neither
a CORSET success nor a CORSET failure.

Run: python -m corset_e2e.datasets.generate_offline --index 20 --size 300000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.environments.families import (  # noqa: E402
    HORIZON, candidate_configs, make, region_of)
from corset_e2e.learners.sac_lag import Actor  # noqa: E402

OUT = os.path.join(ROOT, "results/e2e", "datasets")
CKPT = os.path.join(ROOT, "results/e2e", "oracles")
MIX = (("unc", 0.25), ("p1", 0.20), ("p2", 0.20), ("p12", 0.20), ("rand", 0.15))
NOISE = 0.15


def load_actor(idx: int, tag: str, obs: int, act: int, device: str):
    p = os.path.join(CKPT, f"cfg{idx:02d}_{tag}.pt")
    if not os.path.exists(p):
        return None
    a = Actor(obs, act).to(device)
    a.load_state_dict(torch.load(p, map_location=device))
    a.eval()
    return a


@torch.no_grad()
def rollout(env, actor, device, noise: float, seed: int):
    s = env.reset(seed)
    rows, done = [], False
    while not done:
        if actor is None:
            a = np.random.uniform(-1, 1, env.act_dim)
        else:
            st = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            _, _, mu = actor(st)
            a = mu.squeeze(0).cpu().numpy()
            a = np.clip(a + noise * np.random.randn(*a.shape), -1, 1)
        s2, r, c, done, _ = env.step(a)
        rows.append((s.copy(), a.copy(), r, c.copy(), s2.copy(), float(done)))
        s = s2
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--size", type=int, default=300_000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = candidate_configs()[a.index]
    env = make(cfg, 0)
    actors = {t: load_actor(a.index, t, env.obs_dim, env.act_dim, a.device)
              for t, _ in MIX if t != "rand"}
    missing = [t for t, v in actors.items() if v is None]
    if missing:
        raise SystemExit(f"missing oracle checkpoints for {missing}")

    S, A, R, C, S2, D, SRC = [], [], [], [], [], [], []
    np.random.seed(a.seed)
    ep = 0
    for tag, frac in MIX:
        want = int(a.size * frac)
        got = 0
        while got < want:
            e = make(cfg, 200_000 + ep)
            rows = rollout(e, actors.get(tag), a.device,
                           NOISE if tag != "rand" else 0.0, 200_000 + ep)
            for (s, act, r, c, s2, d) in rows:
                S.append(s); A.append(act); R.append(r); C.append(c)
                S2.append(s2); D.append(d); SRC.append(tag)
            got += len(rows); ep += 1

    S = np.asarray(S, np.float32); A = np.asarray(A, np.float32)
    R = np.asarray(R, np.float32)[:, None]; C = np.asarray(C, np.float32)
    S2 = np.asarray(S2, np.float32); D = np.asarray(D, np.float32)[:, None]
    regions = np.array([region_of(c) for c in C])
    n = len(S)
    frac = {k: float((regions == k).mean())
            for k in ("D1_only", "D2_only", "both", "neither")}
    comp = {t: float((np.array(SRC) == t).mean()) for t, _ in MIX}
    gates = dict(
        d1_only=frac["D1_only"] >= 0.05, d2_only=frac["D2_only"] >= 0.05,
        jointly_safe=frac["neither"] >= 0.10,
        no_dominant_component=max(comp.values()) <= 0.60)
    ok = all(gates.values())
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"cfg{a.index:02d}_n{a.size}.npz"),
                        s=S, a=A, r=R, c=C, s2=S2, d=D)
    rec = dict(index=a.index, name=cfg.name, family=cfg.family, n=n,
               requested=a.size, region_fractions=frac, components=comp,
               gates=gates, accepted=ok,
               reason="ACCEPT" if ok else "REJECT_OFFLINE_SUPPORT")
    json.dump(rec, open(os.path.join(OUT, f"cfg{a.index:02d}_n{a.size}.json"), "w"),
              indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
