"""A neural offline-RL learner (discrete CQL + Lagrangian) for the SA-ORL claim.

saorl.offline_rl is a *tabular* batch-constrained FQI: it discretizes the state and
restricts bootstrapping with an explicit BCQ support mask. A reviewer will ask
whether the kill test is an artifact of that small, hand-tuned policy class. This
module answers with a genuinely different learner -- so the comparison in the
paper is "same SA-ORL principle, different function approximation," not the same
learner run twice:

  * Q is a neural network over the RAW continuous features (rul_hat, q05, anom) --
    no discretization, so the discretization-boundary concern disappears.
  * Offline pessimism comes from Conservative Q-Learning (Kumar et al. 2020):
    instead of an explicit support mask, a CQL regularizer pushes down the value
    of out-of-distribution actions relative to the logged action. This is a
    standard, citable offline-RL algorithm, not a bespoke rule.
  * The semantic constraint is enforced identically to the tabular learner:
    shaped reward r' = r - lam * 1{honored fires}*1{a = continue}, with lam raised
    by dual ascent until the offline honored worst-case cost meets eps; the
    smallest such lam is kept.

The interface mirrors learn_fqi_constrained (returns the same FQIResult) so the
benchmark can drop it in as a second model class. Small nets + small data => runs
on CPU/MPS in seconds; no GPU needed at this scale.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .dsl import Candidate
from .env import ACTIONS
from .offline import OfflineDataset, Policy, evaluate_return, worst_case_cost
from .offline_rl import FQIResult

NA = len(ACTIONS)
A_INDEX = {a: i for i, a in enumerate(ACTIONS)}
CONTINUE = A_INDEX["continue"]
_FEATS = ("rul_hat", "q05", "anom")


def _featurize(state) -> np.ndarray:
    return np.array([state[k] for k in _FEATS], dtype=np.float64)


class _NeuralBatch:
    """Logged transitions as flat feature arrays for neural training."""

    def __init__(self, data: OfflineDataset, honor: Sequence[Candidate]):
        s, a, r, sp, term, fired = [], [], [], [], [], []
        for traj, acts, rews in zip(data.trajectories, data.actions, data.rewards):
            for t in range(len(acts)):
                s.append(_featurize(traj[t]))
                a.append(A_INDEX[acts[t]])
                r.append(rews[t])
                if t + 1 < len(traj):
                    sp.append(_featurize(traj[t + 1]))
                    term.append(0.0)
                else:
                    sp.append(np.zeros(len(_FEATS)))
                    term.append(1.0)
                fired.append(any(c.fires(traj, t) for c in honor))
        self.s = np.asarray(s, dtype=np.float64)
        self.a = np.asarray(a, dtype=np.int64)
        self.r = np.asarray(r, dtype=np.float64)
        self.sp = np.asarray(sp, dtype=np.float64)
        self.term = np.asarray(term, dtype=np.float64)
        self.fired = np.asarray(fired, dtype=bool)
        # standardize features on the logged distribution (domain-agnostic scaling)
        self.mu = self.s.mean(axis=0)
        self.sd = self.s.std(axis=0) + 1e-6

    def norm(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sd


def _train_cql(
    b: _NeuralBatch,
    lam: float,
    gamma: float,
    n_steps: int,
    batch_size: int,
    cql_alpha: float,
    lr: float,
    hidden: int,
    seed: int,
    device: str,
):
    """Train a discrete-action CQL Q-network on the shaped logged rewards."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    dev = torch.device(device)

    S = torch.tensor(b.norm(b.s), dtype=torch.float32, device=dev)
    SP = torch.tensor(b.norm(b.sp), dtype=torch.float32, device=dev)
    A = torch.tensor(b.a, dtype=torch.long, device=dev)
    shaped = b.r - lam * (b.fired & (b.a == CONTINUE))
    R = torch.tensor(shaped, dtype=torch.float32, device=dev)
    NT = torch.tensor(1.0 - b.term, dtype=torch.float32, device=dev)  # non-terminal

    def mlp():
        return nn.Sequential(
            nn.Linear(len(_FEATS), hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, NA),
        ).to(dev)

    q, qt = mlp(), mlp()
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    n = S.shape[0]
    rng = np.random.default_rng(seed)

    for step in range(n_steps):
        idx = torch.tensor(rng.integers(0, n, size=batch_size), device=dev)
        s, sp, a, r, nt = S[idx], SP[idx], A[idx], R[idx], NT[idx]
        with torch.no_grad():
            tgt = r + gamma * nt * qt(sp).max(dim=1).values
        qsa_all = q(s)
        qsa = qsa_all.gather(1, a.unsqueeze(1)).squeeze(1)
        bellman = ((qsa - tgt) ** 2).mean()
        # discrete CQL: push down logsumexp_a Q(s,a) toward the logged action's Q
        cql = (torch.logsumexp(qsa_all, dim=1) - qsa).mean()
        loss = bellman + cql_alpha * cql
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            qt.load_state_dict(q.state_dict())

    q.eval()
    return q, dev


def _greedy_policy(b: _NeuralBatch, q, dev) -> Policy:
    import torch

    def pol(traj, t):
        x = b.norm(_featurize(traj[t]))[None, :]
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=dev)
            ai = int(q(xt).argmax(dim=1).item())
        return ACTIONS[ai]

    return pol


def learn_cql_constrained(
    data: OfflineDataset,
    mdp,
    honor: Sequence[Candidate],
    U_eval: Sequence[Candidate],
    eps: float,
    gamma: float = 0.99,
    n_steps: int = 4000,
    batch_size: int = 256,
    cql_alpha: float = 1.0,
    lr: float = 1e-3,
    hidden: int = 128,
    seed: int = 0,
    lam_grid: Sequence[float] = (0.0, 5.0, 20.0, 80.0, 160.0, 320.0),
    n_return_eps: int = 40,
    return_fn=None,
    device: Optional[str] = None,
) -> FQIResult:
    """Neural CQL + Lagrangian dual ascent on the honored semantic cost.

    Mirrors learn_fqi_constrained: returns the smallest-lam policy whose offline
    honored worst-case cost meets eps (preserving return); if none does, returns
    the largest-lam policy. Selection is fully offline; return is simulated only
    for reporting (return_fn overrides the default MDP rollout).
    """
    # CPU by default: the problem is tiny (3-d state, ~70k transitions) so CPU
    # trains in seconds and is fully deterministic. (torch 1.12 MPS also lacks
    # several indexing ops used here.)
    if device is None:
        device = "cpu"
    b = _NeuralBatch(data, honor)

    chosen: Optional[FQIResult] = None
    fallback: Optional[FQIResult] = None
    for lam in lam_grid:
        q, dev = _train_cql(
            b, lam, gamma, n_steps, batch_size, cql_alpha, lr, hidden, seed, device
        )
        pol = _greedy_policy(b, q, dev)
        honored = worst_case_cost(pol, data, honor, normalize="active")
        ret = (return_fn(pol) if return_fn is not None
               else evaluate_return(mdp, pol, n_episodes=n_return_eps))
        true_worst = worst_case_cost(pol, data, U_eval, normalize="active")
        res = FQIResult(ret, honored, true_worst, lam, pol)
        fallback = res
        if honored <= eps:
            chosen = res
            break
    return chosen if chosen is not None else fallback
