"""Offline constrained policy learning -- the Algorithm-3 hypothesis at CPU scale.

This is a small, transparent stand-in for the full behavior-regularized
constrained actor-critic. The policy class is a two-trigger threshold rule
(replace if RUL_hat <= tau_r OR anom >= tau_a, else operate); "learning" is a
grid search that maximizes model-based return subject to an offline semantic-cost
constraint. It is enough to test the paper's central learning claim:

    A policy optimized to satisfy ONE interpretation can look fully compliant on
    that interpretation yet carry a large HIDDEN worst-case violation under the
    other retained interpretations. Constraining over all of U_alpha (SA-ORL)
    removes the hidden violation, at a measurable -- but bounded -- return cost.

If a single-interpretation learner had no hidden violation, the robust machinery
would be pointless and the idea would be dead. So this is a genuine kill test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .dsl import Candidate
from .env import MaintenanceMDP
from .offline import (
    OfflineDataset,
    Policy,
    evaluate_return,
    policy_costs,
    worst_case_cost,
)

TAUS_R = tuple(range(0, 44, 4))                    # RUL_hat replace thresholds
TAUS_A = (0.55, 0.65, 0.75, 0.85, 0.95, 1.01)      # anom replace thresholds (1.01 = off)


def threshold_policy(tau_r: float, tau_a: float) -> Policy:
    def pol(traj, t):
        s = traj[t]
        if s["rul_hat"] <= tau_r or s["anom"] >= tau_a:
            return "replace"
        return "continue"
    return pol


@dataclass
class LearnedPolicy:
    tau_r: float
    tau_a: float
    ret: float
    honored_cost: float          # worst cost over the *honored* set (what the learner sees)
    true_worst: float            # worst cost over the *full* U_alpha (ground truth)
    policy: Policy


def learn_constrained(
    data: OfflineDataset,
    mdp: MaintenanceMDP,
    honor: Sequence[Candidate],
    U_eval: Sequence[Candidate],
    eps: float,
    n_return_eps: int = 40,
) -> Optional[LearnedPolicy]:
    """Maximize model return s.t. worst honored-interpretation cost <= eps.

    The constraint is evaluated *offline* on D; return is model-based. The full
    worst-case over U_eval is recorded afterwards as ground truth (not used for
    selection) so we can see what the learner missed.
    """
    best: Optional[LearnedPolicy] = None
    for tr in TAUS_R:
        for ta in TAUS_A:
            pol = threshold_policy(tr, ta)
            honored = max(
                policy_costs(pol, data, honor, normalize="active").values(), default=0.0
            )
            if honored <= eps:
                ret = evaluate_return(mdp, pol, n_episodes=n_return_eps)
                if best is None or ret > best.ret:
                    best = LearnedPolicy(tr, ta, ret, honored, 0.0, pol)
    if best is not None:
        best.true_worst = worst_case_cost(best.policy, data, U_eval, normalize="active")
    return best
