"""Online SAC-Lagrangian with a VECTOR of cost channels.

Used only to build the benchmark: the prescreen oracles pi_1, pi_2, pi_12 and
the unconstrained maximiser (plan F.4). One cost critic and one nonnegative
multiplier per channel; costs are never averaged into a scalar before the
constraint is applied.

`constrain` is a boolean mask selecting which channels are enforced, so the
same trainer produces every oracle:
    pi_1  -> [True, False]      pi_2  -> [False, True]
    pi_12 -> [True, True]       pi_unc-> [False, False]
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(i, o, h=256, n=2):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, o)]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs, act):
        super().__init__()
        self.net = mlp(obs, 2 * act)
        self.act = act

    def forward(self, s):
        mu, log_std = self.net(s).chunk(2, -1)
        # Guard: a mis-built CUDA wheel (no kernels for the device's compute
        # capability) yields non-finite or degenerate activations rather than
        # failing loudly, which surfaces later as an unreadable distribution
        # constraint error. Fail fast and visibly instead.
        if not torch.isfinite(mu).all() or not torch.isfinite(log_std).all():
            raise RuntimeError(
                "non-finite actor output; check that the torch build has "
                f"kernels for this GPU (arch_list={torch.cuda.get_arch_list() if torch.cuda.is_available() else 'cpu'})")
        mu = torch.clamp(mu, -20.0, 20.0)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        d = torch.distributions.Normal(mu, std)
        x = d.rsample()
        a = torch.tanh(x)
        logp = (d.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return a, logp, torch.tanh(mu)


class Critic(nn.Module):
    def __init__(self, obs, act):
        super().__init__()
        self.q1, self.q2 = mlp(obs + act, 1), mlp(obs + act, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        return self.q1(x), self.q2(x)


class Buffer:
    def __init__(self, obs, act, k, cap=400_000):
        self.s = np.zeros((cap, obs), np.float32)
        self.a = np.zeros((cap, act), np.float32)
        self.r = np.zeros((cap, 1), np.float32)
        self.c = np.zeros((cap, k), np.float32)
        self.s2 = np.zeros((cap, obs), np.float32)
        self.d = np.zeros((cap, 1), np.float32)
        self.n, self.i, self.cap = 0, 0, cap

    def add(self, s, a, r, c, s2, d):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.c[i], self.s2[i], self.d[i] = \
            s, a, r, c, s2, float(d)
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, bs, dev):
        j = np.random.randint(0, self.n, bs)
        t = lambda x: torch.as_tensor(x[j], device=dev)
        return t(self.s), t(self.a), t(self.r), t(self.c), t(self.s2), t(self.d)


class SACLag:
    def __init__(self, obs, act, k, constrain: Sequence[bool], limits: Sequence[float],
                 device="cpu", gamma=0.99, gamma_c=1.0, lr=3e-4, lr_lam=1e-3,
                 seed=0):
        """gamma_c defaults to 1.0 on purpose.

        The constraint limit is an UNDISCOUNTED episodic budget d*H, so the
        cost critic must estimate the expected remaining violation COUNT on
        the same scale. With gamma_c=0.995 over H=100 the critic's effective
        horizon is 78.8, so a policy sitting exactly at the budget scores
        ~3.94 against a limit of 5: the learner sees slack that does not
        exist and converges at ~1.27x the budget. That bias is what rejected
        9 of 10 maintenance configurations in the v1 prescreen on
        own-reading feasibility, while a scripted policy achieved 0.000 on
        the same tasks.
        """
        torch.manual_seed(seed)
        self.dev = torch.device(device)
        self.k, self.gamma, self.gamma_c = k, gamma, gamma_c
        self.constrain = np.asarray(constrain, dtype=bool)
        self.limits = np.asarray(limits, dtype=np.float32)
        self.actor = Actor(obs, act).to(self.dev)
        self.qr = Critic(obs, act).to(self.dev)
        self.qr_t = Critic(obs, act).to(self.dev)
        self.qr_t.load_state_dict(self.qr.state_dict())
        self.qc = nn.ModuleList([Critic(obs, act) for _ in range(k)]).to(self.dev)
        self.qc_t = nn.ModuleList([Critic(obs, act) for _ in range(k)]).to(self.dev)
        self.qc_t.load_state_dict(self.qc.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_r = torch.optim.Adam(self.qr.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.qc.parameters(), lr=lr)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.dev)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_ent = -float(act)
        self.log_lam = torch.zeros(k, requires_grad=True, device=self.dev)
        self.opt_lam = torch.optim.Adam([self.log_lam], lr=lr_lam)
        self.buf = Buffer(obs, act, k)

    @torch.no_grad()
    def act(self, s, deterministic=False):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.dev).unsqueeze(0)
        a, _, mu = self.actor(s)
        return (mu if deterministic else a).squeeze(0).cpu().numpy()

    def update(self, bs=256):
        s, a, r, c, s2, d = self.buf.sample(bs, self.dev)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2, _ = self.actor(s2)
            q1t, q2t = self.qr_t(s2, a2)
            yr = r + self.gamma * (1 - d) * (torch.min(q1t, q2t) - alpha * logp2)
            yc = []
            for i in range(self.k):
                c1t, c2t = self.qc_t[i](s2, a2)
                yc.append(c[:, i:i + 1] + self.gamma_c * (1 - d) * torch.max(c1t, c2t))
        q1, q2 = self.qr(s, a)
        lr_ = F.mse_loss(q1, yr) + F.mse_loss(q2, yr)
        self.opt_r.zero_grad(set_to_none=True); lr_.backward(); self.opt_r.step()
        lc = 0.0
        for i in range(self.k):
            p1, p2 = self.qc[i](s, a)
            lc = lc + F.mse_loss(p1, yc[i]) + F.mse_loss(p2, yc[i])
        self.opt_c.zero_grad(set_to_none=True); lc.backward(); self.opt_c.step()

        an, logp, _ = self.actor(s)
        qa1, qa2 = self.qr(s, an)
        obj = torch.min(qa1, qa2) - alpha * logp
        lam = self.log_lam.exp()
        pen = 0.0
        for i in range(self.k):
            if self.constrain[i]:
                p1, p2 = self.qc[i](s, an)
                pen = pen + lam[i] * torch.max(p1, p2)
        la = -(obj - pen).mean() if torch.is_tensor(pen) else -obj.mean()
        self.opt_a.zero_grad(set_to_none=True); la.backward(); self.opt_a.step()

        lal = -(self.log_alpha.exp() * (logp.detach() + self.target_ent)).mean()
        self.opt_al.zero_grad(set_to_none=True); lal.backward(); self.opt_al.step()

        if self.constrain.any():
            with torch.no_grad():
                an2, _, _ = self.actor(s)
                viol = []
                for i in range(self.k):
                    p1, p2 = self.qc[i](s, an2)
                    viol.append((torch.max(p1, p2).mean() - float(self.limits[i])))
                viol = torch.stack(viol)
            mask = torch.as_tensor(self.constrain, device=self.dev)
            ll = -(self.log_lam.exp() * viol.detach() * mask).sum()
            self.opt_lam.zero_grad(set_to_none=True); ll.backward(); self.opt_lam.step()
            with torch.no_grad():
                self.log_lam.clamp_(-8.0, 4.0)

        with torch.no_grad():
            for t, p in zip(self.qr_t.parameters(), self.qr.parameters()):
                t.mul_(0.995).add_(0.005 * p)
            for t, p in zip(self.qc_t.parameters(), self.qc.parameters()):
                t.mul_(0.995).add_(0.005 * p)


def train_online(env_fn, constrain, limits, steps=120_000, start=2_000,
                 device="cpu", seed=0):
    env = env_fn(seed)
    ag = SACLag(env.obs_dim, env.act_dim, env.n_costs, constrain, limits,
                device=device, seed=seed)
    s = env.reset(seed)
    for t in range(steps):
        a = (np.random.uniform(-1, 1, env.act_dim) if t < start else ag.act(s))
        s2, r, c, done, _ = env.step(a)
        ag.buf.add(s, a, r, c, s2, done and env.t < 10_000)
        s = env.reset(seed + t) if done else s2
        if t >= start:
            ag.update()
    return ag


@torch.no_grad()
def evaluate(ag, env_fn, episodes=100, seed0=10_000, horizon_norm=True):
    """Monte-Carlo return and per-channel normalised cost."""
    from corset_e2e.environments.families import HORIZON
    rs, cs = [], []
    for e in range(episodes):
        env = env_fn(seed0 + e)
        s = env.reset(seed0 + e)
        R, C, n = 0.0, np.zeros(env.n_costs), 0
        done = False
        while not done:
            a = ag.act(s, deterministic=True) if ag is not None else env.fallback_action(s)
            s, r, c, done, _ = env.step(a)
            R += r; C += c; n += 1
        rs.append(R); cs.append(C / (HORIZON if horizon_norm else max(n, 1)))
    return (float(np.mean(rs)), np.asarray(cs).mean(0), np.asarray(cs),
            float(np.std(rs)))
