"""Additional SOTA constrained offline-RL inner learners for the kill test.

The kill test (single-translation vs \saorl) is a claim about the audited SET
U_alpha, not about the inner optimizer: honoring the set should beat honoring one
reading *whatever* constrained offline-RL learner sits underneath. Two learners
already show it under different function approximation --
``offline_rl.learn_fqi_constrained`` (BCQ-filtered tabular FQI) and
``neural_rl.learn_cql_constrained`` (neural CQL) -- but they share ONE constraint
mechanism: Lagrangian reward shaping (r' = r - lambda*cost, lambda by dual ascent).
A reviewer will rightly ask whether the result is an artifact of that mechanism.

This module adds learners whose constraint mechanism is GENUINELY DIFFERENT, so the
learner-agnostic comparison spans how offline *safe* RL actually enforces a cost:

  * ``learn_cpq_constrained`` -- a CPQ-style learner (Constraints Penalized
    Q-learning, Xu, Zhan & Zhu, AAAI 2022): rather than shaping the reward, it
    learns a COST critic Q_c and FORBIDS actions whose cost-to-go exceeds a limit
    d_lim (treating over-budget actions as out-of-distribution / unsafe), then runs
    reward FQI over the safe, in-support actions only -- CPQ's core "penalize
    unsafe/OOD actions, value only the safe ones" idea, in tabular form.

  * ``learn_pid_lagrangian_constrained`` -- a PID-Lagrangian learner (Stooke,
    Achiam & Abbeel, ICML 2020): same shaped-reward family as the base learner but
    the multiplier is driven by a PID controller on the constraint error rather
    than plain (integral-only) dual ascent -- a different, widely used way to set
    the penalty that damps the overshoot plain dual ascent is prone to.

FAIRNESS. Every learner is *selected* and *scored* by the IDENTICAL offline rule
the base learner uses -- the loosest setting whose offline honored worst-case cost
(``worst_case_cost(., honor, normalize='active')``) meets eps, evaluated by the
same return/worst-case functions. Only the policy-GENERATION mechanism changes
between learners; the honor set (single = most-plausible reading, \saorl = all of
U_alpha) is the experimental variable, identical across learners. All learners
return the shared ``FQIResult`` so benchmark code treats them interchangeably.

Run:  PYTHONPATH=. python3 -m saorl.sota_learners   # self-test on maintenance
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .dsl import Candidate
from .env import MaintenanceMDP
from .offline import OfflineDataset, Policy, evaluate_return, worst_case_cost
from .offline_rl import (
    CONTINUE,
    NA,
    FQIResult,
    _Batch,
    _fit_q,
    _greedy_policy,
)


# --------------------------------------------------------------------------- #
# CPQ-style: cost critic + safe-action restriction                            #
# --------------------------------------------------------------------------- #
def _fit_cost_q(
    b: _Batch, safe: np.ndarray, qr: np.ndarray, gamma: float, n_iter: int
) -> np.ndarray:
    """Tabular cost-to-go Q_c(s,a) under the reward-greedy-over-safe policy.

    Per-step cost is the honored semantic cost 1{honored fires and a=continue} --
    exactly the signal the Lagrangian learner shapes with. Q_c(s,a) is the expected
    discounted future honored cost of taking a at s then acting reward-greedily
    among safe actions, fit by averaging Bellman targets over logged (s,a) cells.
    """
    qr_safe = np.where(safe, qr, -np.inf)
    nxt_a = qr_safe.argmax(axis=1)                      # greedy-safe action per state
    cost_step = (b.fired & (b.a == CONTINUE)).astype(np.float64)
    terminal = b.sp < 0
    ns = np.where(terminal, 0, b.sp)                    # safe index for terminals
    sa_flat = b.s * NA + b.a
    cnt = np.zeros(b.n_states * NA)
    np.add.at(cnt, sa_flat, 1.0)
    nz = cnt > 0
    qc = np.zeros((b.n_states, NA))
    sum_buf = np.zeros(b.n_states * NA)
    for _ in range(n_iter):
        nxt_qc = qc[ns, nxt_a[ns]]
        boot = np.where(terminal, 0.0, gamma * nxt_qc)
        y = cost_step + boot
        sum_buf[:] = 0.0
        np.add.at(sum_buf, sa_flat, y)
        flat = qc.reshape(-1)
        flat[nz] = sum_buf[nz] / cnt[nz]
        qc = flat.reshape(b.n_states, NA)
    return qc


def learn_cpq_constrained(
    data: OfflineDataset,
    mdp: MaintenanceMDP,
    honor: Sequence[Candidate],
    U_eval: Sequence[Candidate],
    eps: float,
    gamma: float = 0.99,
    n_iter: int = 80,
    bcq_tau: float = 0.1,
    bcq_min_abs: int = 1,
    dlim_grid: Sequence[float] = (
        50.0, 30.0, 20.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.2, 0.8, 0.5, 0.3, 0.15, 0.05, 0.0
    ),
    n_fixed_point: int = 8,
    n_return_eps: int = 40,
    return_fn=None,
) -> FQIResult:
    """CPQ-style cost-critic + safe-action restriction (no reward shaping).

    For each cost limit d_lim (loose -> tight), alternate (i) reward FQI restricted
    to currently-safe actions and (ii) a cost critic Q_c, then mark an in-support
    action safe iff Q_c <= d_lim (always keeping >=1 action per supported state).
    The loosest d_lim whose offline honored worst-case cost meets eps is returned
    (preserving return) -- the same selection rule the Lagrangian learner uses, so
    only the mechanism differs. If no d_lim is offline-feasible (CPQ's cost-to-go
    critic controls a discounted *sum*, not the active-normalized *rate* eps bounds,
    so on a knife-edge seed the safest non-degenerate policy can sit just above eps),
    we fall back to the best-safety iterate -- exactly as the PID learner does -- not
    the return-destroying tightest one.
    """
    b = _Batch(data, honor)
    support = b.allowed(bcq_tau, bcq_min_abs)

    chosen: Optional[FQIResult] = None
    fallback: Optional[FQIResult] = None
    for dlim in dlim_grid:
        safe = support.copy()
        for _ in range(n_fixed_point):
            qr = _fit_q(b, safe, 0.0, gamma, n_iter)    # reward Q over safe actions
            qc = _fit_cost_q(b, safe, qr, gamma, n_iter)
            new_safe = support & (qc <= dlim)
            # never strand a supported state with no admissible action: keep its
            # least-cost supported action (CPQ likewise always leaves an action).
            stranded = support.any(axis=1) & ~new_safe.any(axis=1)
            if stranded.any():
                qc_sup = np.where(support, qc, np.inf)
                mincost = qc_sup.argmin(axis=1)
                new_safe[stranded, mincost[stranded]] = True
            if np.array_equal(new_safe, safe):
                break
            safe = new_safe
        qr = _fit_q(b, safe, 0.0, gamma, n_iter)
        pol = _greedy_policy(b, qr, safe)
        honored = worst_case_cost(pol, data, honor, normalize="active")
        ret = (return_fn(pol) if return_fn is not None
               else evaluate_return(mdp, pol, n_episodes=n_return_eps))
        true_worst = worst_case_cost(pol, data, U_eval, normalize="active")
        res = FQIResult(ret, honored, true_worst, dlim, pol)
        if fallback is None or honored < fallback.honored_cost:
            fallback = res                 # best-safety iterate, if none feasible
        if honored <= eps and chosen is None:
            chosen = res                   # loosest feasible (grid is loose->tight)
    return chosen if chosen is not None else fallback


# --------------------------------------------------------------------------- #
# PID-Lagrangian: shaped reward, multiplier set by a PID controller           #
# --------------------------------------------------------------------------- #
def learn_pid_lagrangian_constrained(
    data: OfflineDataset,
    mdp: MaintenanceMDP,
    honor: Sequence[Candidate],
    U_eval: Sequence[Candidate],
    eps: float,
    gamma: float = 0.99,
    n_iter: int = 80,
    bcq_tau: float = 0.1,
    bcq_min_abs: int = 1,
    kp: float = 80.0,
    ki: float = 60.0,
    kd: float = 40.0,
    n_dual_steps: int = 24,
    n_return_eps: int = 40,
    return_fn=None,
) -> FQIResult:
    """PID-Lagrangian (Stooke et al., ICML 2020) dual update on the same shaped
    reward as the base learner.

    The constraint error is e = honored_worst - eps; the multiplier is
    lambda = max(0, kp*e + ki*I + kd*de), with I the running integral of e. PID
    damps the limit-cycle overshoot plain (integral-only) dual ascent shows. We
    track the smallest-lambda iterate that is offline-feasible (honored <= eps),
    matching the base learner's "loosest feasible" selection.
    """
    b = _Batch(data, honor)
    allowed = b.allowed(bcq_tau, bcq_min_abs)

    integral = 0.0
    prev_e = 0.0
    lam = 0.0
    chosen: Optional[FQIResult] = None
    fallback: Optional[FQIResult] = None
    for _ in range(n_dual_steps):
        q = _fit_q(b, allowed, lam, gamma, n_iter)
        pol = _greedy_policy(b, q, allowed)
        honored = worst_case_cost(pol, data, honor, normalize="active")
        ret = (return_fn(pol) if return_fn is not None
               else evaluate_return(mdp, pol, n_episodes=n_return_eps))
        true_worst = worst_case_cost(pol, data, U_eval, normalize="active")
        res = FQIResult(ret, honored, true_worst, lam, pol)
        if fallback is None or honored < fallback.honored_cost:
            fallback = res                     # best-safety iterate, if none feasible
        if honored <= eps and (chosen is None or ret > chosen.ret):
            chosen = res                       # loosest feasible: best return at safe
        # PID update on the constraint error
        e = honored - eps
        integral = max(0.0, integral + e)      # anti-windup: clamp integral >= 0
        deriv = e - prev_e
        prev_e = e
        lam = max(0.0, kp * e + ki * integral + kd * deriv)
    return chosen if chosen is not None else fallback


# --------------------------------------------------------------------------- #
# registry + self-test                                                        #
# --------------------------------------------------------------------------- #
SOTA_LEARNERS = {
    "CPQ": learn_cpq_constrained,
    "PID-Lagrangian": learn_pid_lagrangian_constrained,
}


def _selftest(n_seeds: int = 5, eps: float = 0.05):
    """Sanity check: the kill test (single hides a gap, \saorl closes it) should
    reproduce under each new learner on the synthetic maintenance domain."""
    from .build_plaus_cache import RULE_SETS
    from .construct import construct_U_alpha
    from .experiments import most_plausible
    from .judge import attach_plausibility, load_llm_cache
    from .offline import make_offline_rl_dataset

    MAINT = MaintenanceMDP(r_op=4.0, c_fail=20.0, c_replace=40.0)
    raw = [c for c, _ in RULE_SETS["conservative"]["items"]]
    pool = attach_plausibility(raw, load_llm_cache("conservative"))
    print(f"CPQ/PID self-test ({n_seeds} seeds, eps={eps}) -- maintenance\n")
    print(f"  {'learner':16s} {'policy':8s} {'return':>8s} {'honored':>8s} "
          f"{'TRUE worst':>10s}")
    for lname, fn in SOTA_LEARNERS.items():
        agg = {"single": [], "SA-ORL": []}
        for seed in range(n_seeds):
            data = make_offline_rl_dataset(MAINT, seed=seed)
            U = construct_U_alpha(pool, data.to_semantic_dataset()).U_alpha
            if len(U) < 2:
                continue
            rf = lambda pol: evaluate_return(MAINT, pol, n_episodes=40)
            for name, honor in [("single", most_plausible(U)), ("SA-ORL", list(U))]:
                res = fn(data, MAINT, honor=honor, U_eval=U, eps=eps, return_fn=rf)
                agg[name].append((res.ret, res.honored_cost, res.true_worst))
        for name in ("single", "SA-ORL"):
            arr = np.array(agg[name])
            if arr.size:
                print(f"  {lname:16s} {name:8s} {arr[:,0].mean():8.1f} "
                      f"{arr[:,1].mean():8.3f} {arr[:,2].mean():10.3f}")
        print()


if __name__ == "__main__":
    _selftest()
