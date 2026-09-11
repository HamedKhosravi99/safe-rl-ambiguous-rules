"""Offline-RL learner for Domain 2 (the temporal warning-window gridworld).

The Domain-1 BCQ-FQI learner (saorl.offline_rl) is hardwired to the maintenance
state (rul_hat/anom cells) and forbids `continue`. Domain 2 has a different
sufficient statistic and a different forbidden action, so this is a compact
sibling learner with the SAME algorithm:

  * state  = offset since the most recent warning (the observable statistic that
    determines every Within_W predicate and the latent danger probability),
    capped into a finite bucket set;
  * actions = {advance, wait}; the forbidden action is `advance`;
  * constraint "do not advance where the honored rule fires" is enforced by
    reward shaping r' = r - lambda * 1{honored fires}*1{a = advance}, with lambda
    raised by dual ascent until the offline honored worst-case cost meets eps.

Economics (the externality the rule prices). An accident is only moderately
costly, so a return-greedy learner already waits through the *core* of the danger
window (always unsafe) but is tempted to advance through its rare *tail* (unsafe
only on the longer, rarer danger windows) -- exactly the steps a short window
(W3) does not flag but a long one (W5) does. Honoring only W3 therefore looks
compliant yet hides those tail violations; honoring all of U_alpha removes them.
This mirrors the Domain-1 kill test with a genuinely temporal hidden gap.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .dsl import Candidate
from .offline import OfflineDataset, Policy, worst_case_cost
from .offline_rl import FQIResult

GW_ACTIONS = ("advance", "wait")
_AIDX = {"advance": 0, "wait": 1}
ADVANCE = 0
NA = 2
CAP = 8  # offset bucket 0..CAP; CAP = "no warning within CAP steps" (clear)


def offset_since_warning(traj, t: int, cap: int = CAP) -> int:
    """Steps since the most recent warning at or before t (cap if none in range).

    This is the Markov-sufficient observable: Within_W fires at t iff offset < W,
    and the latent danger probability is a function of the offset alone."""
    for k in range(cap):
        j = t - k
        if j >= 0 and traj[j].get("warning"):
            return k
    return cap


class _GWBatch:
    """Logged transitions in the offset-indexed state space."""

    def __init__(self, data: OfflineDataset, honor: Sequence[Candidate]):
        s, a, r, sp, fired = [], [], [], [], []
        for traj, acts, rews in zip(data.trajectories, data.actions, data.rewards):
            for t in range(len(acts)):
                s.append(offset_since_warning(traj, t))
                a.append(_AIDX[acts[t]])
                r.append(rews[t])
                sp.append(offset_since_warning(traj, t + 1) if t + 1 < len(traj) else -1)
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


def _fit_q(b: _GWBatch, allowed: np.ndarray, lam: float, gamma: float,
           n_iter: int) -> np.ndarray:
    """Vectorized batch FQI on shaped rewards; returns Q [n_states, NA]."""
    q = np.where(allowed, 0.0, -1e9)
    shaped = b.r - lam * (b.fired & (b.a == ADVANCE))
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


def _greedy_policy(b: _GWBatch, q: np.ndarray, allowed: np.ndarray) -> Policy:
    q_masked = np.where(allowed, q, -np.inf)
    has_support = allowed.any(axis=1)
    best = q_masked.argmax(axis=1)

    def pol(traj, t):
        o = offset_since_warning(traj, t)
        if not has_support[o]:
            return "advance"  # no offline support -> the data's reward-greedy prior
        return GW_ACTIONS[int(best[o])]
    return pol


def learn_gw_constrained(
    data: OfflineDataset,
    honor: Sequence[Candidate],
    U_eval: Sequence[Candidate],
    eps: float,
    return_fn,
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
    b = _GWBatch(data, honor)
    allowed = b.allowed(bcq_tau, bcq_min_abs)
    feas = feas_fn if feas_fn is not None else (
        lambda pol, d, h: worst_case_cost(pol, d, h, normalize="active"))

    chosen = None
    fallback = None
    for lam in lam_grid:
        q = _fit_q(b, allowed, lam, gamma, n_iter)
        pol = _greedy_policy(b, q, allowed)
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


def evaluate_gridworld_return(env, policy: Policy, n_episodes: int = 60,
                              seed: int = 123) -> float:
    """Mean episodic return of a policy rolled out in the gridworld env."""
    rng = np.random.default_rng(seed)
    totals = []
    for _ in range(n_episodes):
        state = env.initial(rng)
        traj, total = [], 0.0
        for t in range(env.horizon):
            latent, obs = env.signals(state, rng)
            traj.append(dict(pos=state["pos"], **obs))
            a = policy(traj, t)
            r, pos = env.step(state, a, obs)
            total += r
            state = dict(pos=pos, dleft=latent["dleft"], cool=latent["cool"])
        totals.append(total)
    return float(np.mean(totals))
