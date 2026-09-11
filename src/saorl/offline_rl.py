"""A real offline RL learner for the Algorithm-3 claim (CPU-scale, tabular).

The threshold grid-search in saorl.learn is a transparent stand-in: it enumerates
two thresholds and scores return on the *true* model. That is fine for isolating
the kill test, but a reviewer will (rightly) ask whether the result survives an
actual offline RL algorithm that learns only from the logged dataset. This module
answers that.

It is batch-constrained fitted-Q iteration (discrete BCQ) with a Lagrangian on the
honored semantic cost:

  * State is the discretized observation (rul_hat bin x anom bin) -- the same
    features the DSL predicates read; RUL_true is never used.
  * Q is learned by FQI over logged transitions only. To avoid the classic offline
    failure of bootstrapping off unsupported actions, the max over next actions is
    restricted to actions with enough empirical support at that state (BCQ).
  * The constraint "do not continue where the honored rule fires" is enforced by
    shaping the logged reward: r' = r - lambda * 1{honored fires}*1{a = continue}.
    lambda is raised by dual ascent until the offline honored worst-case cost
    meets the budget eps; the smallest such lambda is kept to preserve return.

The point: the kill test (a single-interpretation learner hides a worst-case
violation that honoring all of U_alpha removes) should reproduce here, with a real
learned value function rather than an enumerated threshold. The implementation is
vectorized over an integer-indexed state space so the whole sweep runs in seconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dsl import Candidate
from .env import ACTIONS, MaintenanceMDP
from .offline import OfflineDataset, Policy, evaluate_return, worst_case_cost

NA = len(ACTIONS)
A_INDEX = {a: i for i, a in enumerate(ACTIONS)}
CONTINUE = A_INDEX["continue"]


def discretize(state, rul_w: float = 4.0, anom_w: float = 0.1) -> Tuple[int, int]:
    """Map an observation to a (rul_hat bin, anom bin) cell."""
    r = int(min(state["rul_hat"], 120.0) // rul_w)
    a = int(min(state["anom"], 1.0) // anom_w)
    return (r, a)


@dataclass
class FQIResult:
    ret: float                # simulated return of the learned greedy policy
    honored_cost: float       # offline worst cost over the honored set (what it saw)
    true_worst: float         # offline worst cost over the full U_eval (ground truth)
    lam: float                # Lagrange multiplier selected
    policy: Policy


class _Batch:
    """Logged transitions in an integer-indexed discretized state space."""

    def __init__(self, data: OfflineDataset, honor: Sequence[Candidate]):
        ids: Dict[Tuple[int, int], int] = {}

        def sid(cell):
            return ids.setdefault(cell, len(ids))

        s, a, r, sp, fired = [], [], [], [], []
        for traj, acts, rews in zip(data.trajectories, data.actions, data.rewards):
            for t in range(len(acts)):
                s.append(sid(discretize(traj[t])))
                a.append(A_INDEX[acts[t]])
                r.append(rews[t])
                sp.append(sid(discretize(traj[t + 1])) if t + 1 < len(traj) else -1)
                fired.append(any(c.fires(traj, t) for c in honor))
        self.ids = ids
        self.n_states = len(ids)
        self.s = np.asarray(s, dtype=np.int64)
        self.a = np.asarray(a, dtype=np.int64)
        self.r = np.asarray(r, dtype=np.float64)
        self.sp = np.asarray(sp, dtype=np.int64)
        self.fired = np.asarray(fired, dtype=bool)
        # state-action visit counts
        self.counts = np.zeros((self.n_states, NA))
        np.add.at(self.counts, (self.s, self.a), 1.0)

    def allowed(self, tau: float, min_abs: int) -> np.ndarray:
        thr = np.maximum(float(min_abs), tau * self.counts.max(axis=1, keepdims=True))
        return self.counts >= thr


def _fit_q(
    b: _Batch, allowed: np.ndarray, lam: float, gamma: float, n_iter: int
) -> np.ndarray:
    """Vectorized batch FQI on shaped rewards; returns Q [n_states, NA]."""
    q = np.where(allowed, 0.0, -1e9)
    shaped = b.r - lam * (b.fired & (b.a == CONTINUE))
    terminal = b.sp < 0
    sa_flat = b.s * NA + b.a
    sum_buf = np.zeros(b.n_states * NA)
    cnt_buf = np.zeros(b.n_states * NA)
    np.add.at(cnt_buf, sa_flat, 1.0)
    nz = cnt_buf > 0
    q_masked = np.where(allowed, q, -np.inf)
    for _ in range(n_iter):
        nxt = q_masked.max(axis=1)
        nxt = np.where(np.isfinite(nxt), nxt, 0.0)
        boot = np.where(terminal, 0.0, gamma * nxt[np.where(terminal, 0, b.sp)])
        y = shaped + boot
        sum_buf[:] = 0.0
        np.add.at(sum_buf, sa_flat, y)
        flat = q.reshape(-1)
        flat[nz] = sum_buf[nz] / cnt_buf[nz]
        q = flat.reshape(b.n_states, NA)
        q_masked = np.where(allowed, q, -np.inf)
    return q


def _greedy_policy(b: _Batch, q: np.ndarray, allowed: np.ndarray) -> Policy:
    q_masked = np.where(allowed, q, -np.inf)
    has_support = allowed.any(axis=1)
    best = q_masked.argmax(axis=1)

    def pol(traj, t):
        cell = discretize(traj[t])
        sid = b.ids.get(cell)
        if sid is None or not has_support[sid]:
            return "continue"  # no offline support -> default to the data's prior
        return ACTIONS[int(best[sid])]
    return pol


def learn_fqi_constrained(
    data: OfflineDataset,
    mdp: MaintenanceMDP,
    honor: Sequence[Candidate],
    U_eval: Sequence[Candidate],
    eps: float,
    gamma: float = 0.99,
    n_iter: int = 80,
    bcq_tau: float = 0.1,
    bcq_min_abs: int = 1,
    lam_grid: Sequence[float] = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0),
    n_return_eps: int = 40,
    return_fn=None,
    feas_fn=None,
) -> FQIResult:
    """Offline BCQ-FQI + Lagrangian dual ascent on the honored semantic cost.

    Returns the smallest-lambda policy whose offline honored worst-case cost meets
    eps (so return is preserved); if none meets it, returns the largest-lambda
    policy. Return is simulated for reporting only; selection is fully offline.
    `return_fn(policy) -> float` overrides the default MDP rollout (e.g. to score
    a policy on a real-data replay environment).
    `feas_fn(policy, data, honor) -> float` overrides the dual-ascent feasibility
    quantity (default: per-interpretation worst-case cost over the active steps).
    Passing union_cost yields the 'union-cost' robustification baseline.
    """
    b = _Batch(data, honor)
    allowed = b.allowed(bcq_tau, bcq_min_abs)
    feas = feas_fn if feas_fn is not None else (
        lambda pol, d, h: worst_case_cost(pol, d, h, normalize="active"))

    chosen: Optional[FQIResult] = None
    fallback: Optional[FQIResult] = None
    for lam in lam_grid:
        q = _fit_q(b, allowed, lam, gamma, n_iter)
        pol = _greedy_policy(b, q, allowed)
        honored = feas(pol, data, honor)
        ret = (return_fn(pol) if return_fn is not None
               else evaluate_return(mdp, pol, n_episodes=n_return_eps))
        true_worst = worst_case_cost(pol, data, U_eval, normalize="active")
        res = FQIResult(ret, honored, true_worst, lam, pol)
        fallback = res
        if honored <= eps:
            chosen = res
            break
    return chosen if chosen is not None else fallback
