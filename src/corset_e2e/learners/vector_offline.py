"""WP-G + WP-H: vector-cost offline learners and the fixed-policy certificate.

One reward critic, ONE COST CRITIC AND ONE MULTIPLIER PER RETAINED READING,
one policy. Costs are never averaged into a scalar before the constraint is
applied (plan G.1). Three learners share the trainer and differ in how they
regularise toward the behaviour data:

  vector_cql   conservative Q penalty on out-of-distribution actions
  vector_fqi   Lagrangian fitted-Q with a behaviour-cloning trust term
  vector_bc    behaviour-regularised baseline (the published-learner stand-in)

Arms per task (plan's baseline list):
  single_1  defends only psi_1        single_2  defends only psi_2
  corset    defends every retained reading (the vector arm)
  union     one scalar critic on max_k c_k -- the conservative surrogate
  fallback  the environment's certified fallback

WP-H: every shipped policy is frozen, hashed, and certified on independent
episodes with a Hoeffding upper bound per maximal reading; nothing is pooled
across seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

_NT = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.environments.families import (  # noqa: E402
    HORIZON, candidate_configs, make)
from corset_e2e.learners.sac_lag import Actor, Critic  # noqa: E402

OUT = os.path.join(ROOT, "results/e2e", "offline")
DATA = os.path.join(ROOT, "results/e2e", "datasets")
DELTA_EV = 0.05
CERT_EPISODES = 1000


class VectorOffline:
    def __init__(self, obs, act, k, arm, learner, limit, device="cpu", seed=0,
                 cql_alpha=1.0, bc_alpha=0.5):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.dev = torch.device(device)
        self.k, self.arm, self.learner = k, arm, learner
        self.limit = limit
        # vector_fqi keeps a light trust term so the policy may improve on the
        # behaviour; vector_bc is the published-learner stand-in, where the
        # behaviour term dominates the objective. Sharing one weight made the
        # two learners identical at matched seeds.
        self.cql_alpha = cql_alpha
        self.bc_alpha = bc_alpha if learner == "vector_fqi" else 5.0 * bc_alpha
        self.actor = Actor(obs, act).to(self.dev)
        self.qr = Critic(obs, act).to(self.dev)
        self.qr_t = Critic(obs, act).to(self.dev)
        self.qr_t.load_state_dict(self.qr.state_dict())
        # one cost critic per enforced channel (union arm collapses to one)
        self.nc = 1 if arm == "union" else k
        self.qc = nn.ModuleList([Critic(obs, act) for _ in range(self.nc)]).to(self.dev)
        self.qc_t = nn.ModuleList([Critic(obs, act) for _ in range(self.nc)]).to(self.dev)
        self.qc_t.load_state_dict(self.qc.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), 3e-4)
        self.opt_r = torch.optim.Adam(self.qr.parameters(), 3e-4)
        self.opt_c = torch.optim.Adam(self.qc.parameters(), 3e-4)
        self.log_lam = torch.zeros(self.nc, requires_grad=True, device=self.dev)
        self.opt_lam = torch.optim.Adam([self.log_lam], 1e-4)
        if arm == "single_1":
            self.mask = np.array([True] + [False] * (k - 1))
        elif arm == "single_2":
            self.mask = np.array([False, True] + [False] * (k - 2))
        else:
            self.mask = np.ones(self.nc, dtype=bool)

    def _cost_targets(self, c):
        if self.arm == "union":
            return c.max(dim=1, keepdim=True)[0]
        return c

    def update(self, batch):
        s, a, r, c, s2, d = batch
        ct = self._cost_targets(c)
        with torch.no_grad():
            a2, logp2, _ = self.actor(s2)
            q1t, q2t = self.qr_t(s2, a2)
            yr = r + 0.99 * (1 - d) * torch.min(q1t, q2t)
            # undiscounted, matching the episodic budget d*H the multiplier
            # is compared against: a discounted critic understates the
            # violation count and settles above budget (the same scale error
            # that rejected nine of ten maintenance configs in prescreen v1)
            yc = [ct[:, i:i + 1] + 1.0 * (1 - d) * torch.max(*self.qc_t[i](s2, a2))
                  for i in range(self.nc)]
        q1, q2 = self.qr(s, a)
        loss_r = F.mse_loss(q1, yr) + F.mse_loss(q2, yr)
        if self.learner == "vector_cql":
            rnd = torch.empty_like(a).uniform_(-1, 1)
            qr_rand = torch.min(*self.qr(s, rnd))
            qr_data = torch.min(q1, q2)
            loss_r = loss_r + self.cql_alpha * (qr_rand.mean() - qr_data.mean())
        self.opt_r.zero_grad(set_to_none=True); loss_r.backward(); self.opt_r.step()

        lc = 0.0
        for i in range(self.nc):
            p1, p2 = self.qc[i](s, a)
            lc = lc + F.mse_loss(p1, yc[i]) + F.mse_loss(p2, yc[i])
        self.opt_c.zero_grad(set_to_none=True); lc.backward(); self.opt_c.step()

        an, logp, _ = self.actor(s)
        qa = torch.min(*self.qr(s, an))
        lam = self.log_lam.exp()
        pen = 0.0
        for i in range(self.nc):
            if self.mask[i]:
                pen = pen + lam[i] * torch.max(*self.qc[i](s, an))
        la = -(qa - (pen if torch.is_tensor(pen) else 0.0)).mean()
        if self.learner in ("vector_fqi", "vector_bc"):
            la = la + self.bc_alpha * F.mse_loss(an, a)     # stay near behaviour
        self.opt_a.zero_grad(set_to_none=True); la.backward(); self.opt_a.step()

        if self.mask.any():
            with torch.no_grad():
                an2, _, _ = self.actor(s)
                viol = torch.stack([torch.max(*self.qc[i](s, an2)).mean()
                                    - self.limit for i in range(self.nc)])
            m = torch.as_tensor(self.mask, device=self.dev)
            ll = -(self.log_lam.exp() * viol.detach() * m).sum()
            self.opt_lam.zero_grad(set_to_none=True); ll.backward(); self.opt_lam.step()
            with torch.no_grad():
                self.log_lam.clamp_(-8.0, 4.0)
        with torch.no_grad():
            for t, p in zip(self.qr_t.parameters(), self.qr.parameters()):
                t.mul_(0.995).add_(0.005 * p)
            for t, p in zip(self.qc_t.parameters(), self.qc.parameters()):
                t.mul_(0.995).add_(0.005 * p)

    @torch.no_grad()
    def act(self, s):
        st = torch.as_tensor(s, dtype=torch.float32, device=self.dev).unsqueeze(0)
        _, _, mu = self.actor(st)
        return mu.squeeze(0).cpu().numpy()


@torch.no_grad()
def certify(policy, cfg, episodes=CERT_EPISODES, seed0=700_000, fallback=False):
    """WP-H: frozen-artifact evaluation on independent episodes."""
    rs, cs = [], []
    for e in range(episodes):
        env = make(cfg, seed0 + e)
        s = env.reset(seed0 + e)
        R, C, done = 0.0, np.zeros(env.n_costs), False
        while not done:
            a = env.fallback_action(s) if fallback else policy.act(s)
            s, r, c, done, _ = env.step(a)
            R += r; C += c
        rs.append(R); cs.append(C / HORIZON)
    cs = np.asarray(cs)
    n, w = len(cs), cs.shape[1]
    # Hoeffding upper bound per maximal reading, union-corrected over w readings
    half = math.sqrt(math.log(2 * w / DELTA_EV) / (2 * n))
    ucb = cs.mean(0) + half
    viol_ep = (cs > cfg.budget).any(1).mean()
    return dict(ret=float(np.mean(rs)), cost=cs.mean(0).tolist(),
                ucb=ucb.tolist(), half_width=half,
                episode_violation_rate=float(viol_ep),
                ships=bool(ucb.max() <= cfg.budget), n_episodes=n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--learner", default="vector_cql")
    ap.add_argument("--size", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    if _NT:
        torch.set_num_threads(_NT)
    cfg = candidate_configs()[a.index]
    dev = torch.device(a.device)
    z = np.load(os.path.join(DATA, f"cfg{a.index:02d}_n{a.size}.npz"))
    T = {k: torch.as_tensor(z[k], device=dev) for k in ("s", "a", "r", "c", "s2", "d")}
    n = T["s"].shape[0]
    env = make(cfg, 0)

    if a.arm == "fallback":
        cert = certify(None, cfg, fallback=True)
        rec = dict(index=a.index, arm=a.arm, learner="none", seed=a.seed, cert=cert)
    else:
        ag = VectorOffline(env.obs_dim, env.act_dim, env.n_costs, a.arm, a.learner,
                           cfg.budget * HORIZON, device=a.device, seed=a.seed)
        for step in range(a.steps):
            j = torch.randint(0, n, (256,), device=dev)
            ag.update(tuple(T[k][j] for k in ("s", "a", "r", "c", "s2", "d")))
        sd = ag.actor.state_dict()
        blob = b"".join(v.detach().cpu().numpy().tobytes() for v in sd.values())
        cert = certify(ag, cfg)
        rec = dict(index=a.index, arm=a.arm, learner=a.learner, seed=a.seed,
                   size=a.size, steps=a.steps,
                   policy_sha256=hashlib.sha256(blob).hexdigest(), cert=cert)
    rec.update(name=cfg.name, family=cfg.family, budget=cfg.budget)
    os.makedirs(OUT, exist_ok=True)
    tag = f"cfg{a.index:02d}_{a.arm}_{a.learner}_n{a.size}_s{a.seed}"
    json.dump(rec, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
