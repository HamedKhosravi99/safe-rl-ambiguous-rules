"""Offline-RL learner for Domain 3 (the budget-guardrail agent).

A compact sibling of the Domain-2 gridworld learner with the SAME algorithm
(offset-indexed tabular BCQ-FQI + Lagrangian dual ascent), specialized to the
budget agent's sufficient statistic and forbidden action:

  * state  = the running-bill bucket floor(spend / unit_cost), capped (this is
    the Markov-sufficient observable: every cap predicate spend>=theta and the
    constraint both depend on the bill alone);
  * actions = {buy, skip}; the forbidden action is `buy`;
  * constraint "do not buy where the honored cap fires" is enforced by reward
    shaping r' = r - lambda * 1{honored fires} * 1{a = buy}, with lambda raised by
    dual ascent until the offline honored worst-case cost meets eps.

Economics (the externality the guardrail prices). Each paid action earns task
value and adds to the bill; nothing else penalizes spending, so a return-greedy
agent buys at every step. Honoring only the loosest plausible cap (e.g. $60)
stops buying late and looks compliant on that reading, yet keeps buying through
the mid-bill band [30,60) that the tighter retained caps forbid -- hidden
overspend. Honoring all of U_alpha stops at the tightest retained cap and removes
those violations. This mirrors the Domains 1-2 kill tests on the budget axis.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .dsl import Candidate
from .offline import OfflineDataset, Policy, worst_case_cost
from .offline_rl import FQIResult

B_ACTIONS = ("buy", "skip")
_AIDX = {"buy": 0, "skip": 1}
BUY = 0
NA = 2
CAP = 12  # bill bucket 0..CAP; CAP absorbs every bill >= CAP*unit_cost


def spend_bucket(traj, t: int, unit_cost: float = 10.0, cap: int = CAP) -> int:
    """Running-bill bucket floor(spend / unit_cost), capped at `cap`.

    This is the Markov-sufficient observable: cap predicate spend>=theta fires at
    t iff bucket >= theta/unit_cost, and the optimal honoring decision depends on
    the bill alone."""
    return min(cap, int(traj[t]["spend"] // unit_cost))


class _BudgetBatch:
    """Logged transitions in the bill-bucket state space."""

    def __init__(self, data: OfflineDataset, honor: Sequence[Candidate],
                 unit_cost: float = 10.0):
        s, a, r, sp, fired = [], [], [], [], []
        for traj, acts, rews in zip(data.trajectories, data.actions, data.rewards):
            for t in range(len(acts)):
                s.append(spend_bucket(traj, t, unit_cost))
                a.append(_AIDX[acts[t]])
                r.append(rews[t])
                sp.append(spend_bucket(traj, t + 1, unit_cost)
                          if t + 1 < len(traj) else -1)
                fired.append(any(c.fires(traj, t) for c in honor))
        self.n_states = CAP + 1
        self.s = np.asarray(s, dtype=np.int64)
        self.a = np.asarray(a, dtype=np.int64)
        self.r = np.asarray(r, dtype=np.float64)
        self.sp = np.asarray(sp, dtype=np.int64)
        self.fired = np.asarray(fired, dtype=bool)
        self.counts = np.zeros((self.n_states, NA))
        np.add.at(self.counts, (self.s, self.a), 1.0)

    def allowed(self, tau: float, min_abs: int) -> np.ndarray:
        thr = np.maximum(float(min_abs), tau * self.counts.max(axis=1, keepdims=True))
        return self.counts >= thr


def _fit_q(b: _BudgetBatch, allowed: np.ndarray, lam: float, gamma: float,
           n_iter: int) -> np.ndarray:
    """Vectorized batch FQI on shaped rewards; returns Q [n_states, NA]."""
    q = np.where(allowed, 0.0, -1e9)
    shaped = b.r - lam * (b.fired & (b.a == BUY))
    terminal = b.sp < 0
    sa_flat = b.s * NA + b.a
    cnt_buf = np.zeros(b.n_states * NA)
    np.add.at(cnt_buf, sa_flat, 1.0)
    nz = cnt_buf > 0
    q_masked = np.where(allowed, q, -np.inf)
    sum_buf = np.zeros(b.n_states * NA)
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


def _greedy_policy(b: _BudgetBatch, q: np.ndarray, allowed: np.ndarray,
                   unit_cost: float) -> Policy:
    q_masked = np.where(allowed, q, -np.inf)
    has_support = allowed.any(axis=1)
    best = q_masked.argmax(axis=1)

    def pol(traj, t):
        o = spend_bucket(traj, t, unit_cost)
        if not has_support[o]:
            return "buy"  # no offline support -> the data's reward-greedy prior
        return B_ACTIONS[int(best[o])]
    return pol


def learn_budget_constrained(
    data: OfflineDataset,
    honor: Sequence[Candidate],
    U_eval: Sequence[Candidate],
    eps: float,
    return_fn,
    unit_cost: float = 10.0,
    gamma: float = 0.97,
    n_iter: int = 120,
    bcq_tau: float = 0.05,
    bcq_min_abs: int = 1,
    lam_grid: Sequence[float] = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0),
    feas_fn=None,
) -> FQIResult:
    """Offline FQI + Lagrangian dual ascent on the honored semantic cost.

    Returns the smallest-lambda policy whose offline honored worst-case cost meets
    eps (so return is preserved); if none meets it, the largest-lambda policy.
    Return is scored by return_fn(policy) for reporting only; selection is offline.
    `feas_fn(policy, data, honor) -> float` overrides the dual-ascent feasibility
    quantity (default: per-interpretation worst case); pass union_cost for the
    union-cost baseline."""
    b = _BudgetBatch(data, honor, unit_cost)
    allowed = b.allowed(bcq_tau, bcq_min_abs)
    feas = feas_fn if feas_fn is not None else (
        lambda pol, d, h: worst_case_cost(pol, d, h, normalize="active"))

    chosen = None
    fallback = None
    for lam in lam_grid:
        q = _fit_q(b, allowed, lam, gamma, n_iter)
        pol = _greedy_policy(b, q, allowed, unit_cost)
        honored = feas(pol, data, honor)
        true_worst = worst_case_cost(pol, data, U_eval, normalize="active")
        res = FQIResult(ret=0.0, honored_cost=honored, true_worst=true_worst,
                        lam=lam, policy=pol)
        fallback = res
        if honored <= eps:
            chosen = res
            break
    res = chosen if chosen is not None else fallback
    res.ret = return_fn(res.policy)
    return res


def evaluate_budget_return(env, policy: Policy, n_episodes: int = 60,
                           seed: int = 123) -> float:
    """Mean episodic return (task value earned) of a policy in the budget env."""
    rng = np.random.default_rng(seed)
    totals = []
    for _ in range(n_episodes):
        state = env.initial(rng)
        traj, total = [], 0.0
        for t in range(env.horizon):
            traj.append(env.observe(state))
            a = policy(traj, t)
            r, state = env.step(state, a)
            total += r
        totals.append(total)
    return float(np.mean(totals))
