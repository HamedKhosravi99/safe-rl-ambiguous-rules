"""The gate: a guard that consults an offline-trained policy to allow/deny the
PAID tool, wrapping the agent loop (paper section 22.3, components 6-7).

Guard interface (the contract the agent loop calls):

    guard.allow_paid(features: dict) -> bool

Concrete guards:

  * Passthrough        -- always allow (logging runs / dry run).
  * TrainedGuard       -- wraps a saorl FQI policy learned on offline logs; the
                          policy gates the PAID tool exactly as the budget
                          domain gates `buy`.

Three trained modes (paper section 22.3, component 6), which differ only in the
set of cap/quota READINGS the learner is told to honor:

  * single           -- honor the single most-plausible reading.
  * corset           -- honor the whole conformal-retained set (worst-case over
                        all surviving readings); this is CORSET's robust policy.
  * post_clarify     -- honor the reading picked after the clarification, i.e.
                        the externally-anchored target reading.

We REUSE saorl's FQI unchanged (`learn_budget_constrained`), per the plan. That
learner buckets state on `traj[t]["spend"]` via `spend_bucket`, so to gate on a
chosen AXIS (real dollars, or paid-call count for a per-episode quota) we build
a saorl OfflineDataset whose "spend" field carries that axis and set unit_cost
accordingly. The cap/quota readings are saorl `Candidate`s over the same
"spend" field, so predicate firing and the learner's bucketing agree. The demo
does not modify anything under saorl/.

The numeric thresholds and which reading is "most plausible" / "anchored" come
from the policy text + conformal scoring, which are DEFERRED until the policy is
selected (see policy_source.json). This module is parametrized by the reading
sets; `demo_call_quota_readings()` provides placeholder readings so the wiring
is testable now.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from saorl.budget_rl import learn_budget_constrained, spend_bucket, CAP  # noqa: E402
from saorl.offline import OfflineDataset, worst_case_cost  # noqa: E402
from saorl.dsl import Atom, Candidate  # noqa: E402

from agent_loop import GuardLike, Passthrough  # noqa: E402,F401  (re-export)


# --------------------------------------------------------------------------- #
# Cap / quota readings (the frozen DSL pool for the demo rule)
# --------------------------------------------------------------------------- #
def make_reading(theta: float, axis: str = "calls",
                 plausibility: Sequence[float] = ()) -> Candidate:
    """A single '<axis> >= theta' cap/quota reading as a saorl Candidate.

    The predicate reads the field named 'spend' -- that is the field the saorl
    learner and evaluators bucket on -- but we populate it from `axis` when we
    build the dataset (see episodes_to_offline). `axis` is recorded in the
    candidate name so the two never drift."""
    name = f"psi_{axis}>={theta}"
    return Candidate(name, Atom("spend", "ge", float(theta)),
                     frozenset({"buy"}), tuple(plausibility))


def demo_call_quota_readings() -> Dict[str, List[Candidate]]:
    """PLACEHOLDER per-episode paid-call quota readings for wiring tests.

    A vague policy quantifier ("avoid excessive calls") admits several thresholds;
    these stand in until the real policy text pins the anchor. Nested tight->loose
    like the saorl budget caps."""
    tight = make_reading(5, "calls", plausibility=(0.8, 0.7, 0.75))
    mid = make_reading(8, "calls", plausibility=(0.6, 0.55, 0.5))
    loose = make_reading(10, "calls", plausibility=(0.3, 0.35, 0.4))
    return {
        "retained": [tight, mid, loose],   # conformal-surviving set (CORSET honors all)
        "single": [mid],                    # single most-plausible reading
        "anchored": [tight],                # post-clarification / anchored target
    }


# --------------------------------------------------------------------------- #
# Adapter: demo episode logs -> saorl OfflineDataset
# --------------------------------------------------------------------------- #
def episodes_to_offline(episodes: Sequence[Dict[str, Any]],
                        axis: str = "calls") -> OfflineDataset:
    """Map demo episodes (EpisodeResult.to_jsonable() dicts, or JSONL rows) to a
    saorl OfflineDataset whose 'spend' field carries `axis`.

    Each episode row must have 'trajectory' (list of state dicts with the chosen
    axis field), 'actions' ("buy"/"skip"), 'rewards'."""
    trajs, acts, rews = [], [], []
    for ep in episodes:
        traj = [{"spend": float(st[axis]), "step": int(st.get("step", i))}
                for i, st in enumerate(ep["trajectory"])]
        trajs.append(traj)
        acts.append(list(ep["actions"]))
        rews.append([float(r) for r in ep["rewards"]])
    return OfflineDataset(trajectories=trajs, actions=acts, rewards=rews)


def offline_return_proxy(dataset: OfflineDataset) -> Callable:
    """A REPORT-ONLY offline return estimator for learn_budget_constrained.

    Returns f(policy) = mean over episodes of the logged reward on transitions
    where the policy's action matches the logged action (a behavior-consistent
    slice). It is a crude proxy for ranking, NOT the headline number; the real
    per-condition return comes from LIVE eval later (section 22.4)."""
    def ret(policy) -> float:
        totals = []
        for traj, acts, rews in zip(dataset.trajectories, dataset.actions,
                                    dataset.rewards):
            tot = 0.0
            for t in range(len(acts)):
                if policy(traj, t) == acts[t]:
                    tot += rews[t]
            totals.append(tot)
        return float(np.mean(totals)) if totals else 0.0
    return ret


# --------------------------------------------------------------------------- #
# Trained guard
# --------------------------------------------------------------------------- #
@dataclass
class TrainedGuard(GuardLike):
    """Wraps a saorl policy learned by FQI; gates PAID iff the policy says 'buy'."""
    policy: Callable          # saorl Policy: (traj, t) -> "buy"/"skip"
    axis: str = "calls"
    unit_cost: float = 1.0
    mode: str = "trained"
    honored_cost: float = float("nan")
    true_worst: float = float("nan")
    lam: float = float("nan")

    def allow_paid(self, features: Dict[str, Any]) -> bool:
        x = float(features[self.axis])
        # one-element trajectory; the caps/spend_bucket read traj[0]["spend"] only
        traj = [{"spend": x, "step": int(features.get("step", 0))}]
        return self.policy(traj, 0) == "buy"


def train_guard(episodes: Sequence[Dict[str, Any]],
                honor: Sequence[Candidate],
                eval_readings: Sequence[Candidate],
                *, axis: str = "calls", unit_cost: float = 1.0,
                eps: float = 0.0, mode: str = "trained",
                gamma: float = 0.97) -> TrainedGuard:
    """Train ONE guard via saorl FQI + dual ascent on the honored semantic cost.

    `honor` is the reading-set this guard must respect (one reading for 'single'
    /'post_clarify', the whole retained set for 'corset'); `eval_readings` is the
    set violations are measured against (usually the retained set)."""
    data = episodes_to_offline(episodes, axis=axis)
    res = learn_budget_constrained(
        data, honor=list(honor), U_eval=list(eval_readings), eps=eps,
        return_fn=offline_return_proxy(data), unit_cost=unit_cost, gamma=gamma,
    )
    return TrainedGuard(policy=res.policy, axis=axis, unit_cost=unit_cost,
                        mode=mode, honored_cost=res.honored_cost,
                        true_worst=res.true_worst, lam=res.lam)


def train_three_guards(episodes: Sequence[Dict[str, Any]],
                       readings: Optional[Dict[str, List[Candidate]]] = None,
                       *, axis: str = "calls", unit_cost: float = 1.0,
                       eps: float = 0.0) -> Dict[str, TrainedGuard]:
    """Train the single / corset / post_clarify guards (section 22.3 component 6).

    `readings` must supply 'retained' (conformal set), 'single' (most-plausible),
    and 'anchored' (post-clarification target). Defaults to placeholder call-quota
    readings until the policy text is selected."""
    readings = readings or demo_call_quota_readings()
    retained = readings["retained"]
    return {
        "single": train_guard(episodes, readings["single"], retained,
                              axis=axis, unit_cost=unit_cost, eps=eps,
                              mode="single"),
        "corset": train_guard(episodes, retained, retained,
                              axis=axis, unit_cost=unit_cost, eps=eps,
                              mode="corset"),
        "post_clarify": train_guard(episodes, readings["anchored"], retained,
                                    axis=axis, unit_cost=unit_cost, eps=eps,
                                    mode="post_clarify"),
    }


if __name__ == "__main__":
    # Smoke test: fabricate a few episodes covering the call-quota buckets and
    # verify all three guards train and gate sensibly. No API cost.
    import numpy as np
    rng = np.random.default_rng(0)
    eps_list = []
    for e in range(40):
        traj, acts, rews = [], [], []
        calls = 0
        for t in range(12):
            traj.append({"calls": calls, "spend": 0.001 * calls, "step": t,
                         "progress": min(1.0, 0.1 * t)})
            a = "buy" if rng.random() < 0.6 else "skip"
            acts.append(a)
            rews.append(1.0 if a == "buy" else 0.3)
            if a == "buy":
                calls += 1
        eps_list.append({"trajectory": traj, "actions": acts, "rewards": rews})

    guards = train_three_guards(eps_list)
    for name, g in guards.items():
        allowed = [c for c in range(13) if g.allow_paid({"calls": c, "step": 0})]
        print(f"{name:13s} lam={g.lam:5.1f} honored_cost={g.honored_cost:.3f} "
              f"true_worst={g.true_worst:.3f} allows buy at calls={allowed}")
