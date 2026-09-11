"""Omitted-intended-semantics ablation (plan section 8.8.4): the certification is
membership-tied, not arbitrary.

\saorl{}'s guarantee is precise: it protects *every reading in* $U_\alpha$. To show
that guarantee is meaningful --- that you cannot drop the reading you care about and
remain protected --- we omit the \emph{intended} interpretation from the honored set
and check whether the learned policy still satisfies it.

Choice of intended reading (principled, pre-registerable). For a safety rule
(``do not operate when degradation is severe''; ``do not advance right after a
warning'') the operative reading is the most \emph{protective} retained one --- the
candidate with the widest firing region over the data, since a narrower reading is
implied by it but not vice versa. We take $\psi^\star=\arg\max_{\psi\in U_\alpha}$
(firing count over the dataset). Honoring $U_\alpha\setminus\{\psi^\star\}$ therefore
honors only readings that do NOT by themselves cover $\psi^\star$'s extra region.

For each seed with $|U_\alpha|\ge 2$ we train two SA-ORL policies --- one honoring
all of $U_\alpha$, one honoring $U_\alpha\setminus\{\psi^\star\}$ --- and report each
policy's true worst-case cost on the omitted reading $\psi^\star$ alone (active-state
normalized). The full policy drives it below $\eps$; the omit policy leaves a real
violation, exactly the gap a single-translation learner hides --- now shown to be a
property of \emph{which readings are retained}, not of the constraint count.

Run:  PYTHONPATH=. python3 -m saorl.omit_intended
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .build_plaus_cache import RULE_SETS
from .construct import construct_U_alpha
from .dsl import Candidate
from .env import MaintenanceMDP
from .judge import attach_plausibility, load_llm_cache
from .offline import (
    OfflineDataset,
    evaluate_return,
    make_offline_rl_dataset,
    worst_case_cost,
)
from .offline_rl import learn_fqi_constrained

N_SEEDS = 30
EPS = 0.05
MAINT = MaintenanceMDP(r_op=4.0, c_fail=20.0, c_replace=40.0)


@dataclass
class OmitRow:
    domain: str
    seed: int
    u_size: int
    intended: str
    cost_star_full: float   # true cost on psi* of the policy honoring all of U
    cost_star_omit: float   # true cost on psi* of the policy honoring U \ {psi*}
    worst_full: float       # true worst over U of the full policy
    worst_omit: float       # true worst over U of the omit policy


def _firing_count(cand: Candidate, data: OfflineDataset) -> int:
    return sum(cand.fires(traj, t)
               for traj in data.trajectories for t in range(len(traj)))


def _intended_idx(U: Sequence[Candidate], data: OfflineDataset) -> int:
    """Most protective retained reading = widest firing region over the data."""
    return int(np.argmax([_firing_count(c, data) for c in U]))


def probe_maintenance(rule_set: str) -> List[OmitRow]:
    raw = [c for c, _ in RULE_SETS[rule_set]["items"]]
    judge = load_llm_cache(rule_set)
    rows: List[OmitRow] = []
    for seed in range(N_SEEDS):
        data = make_offline_rl_dataset(MAINT, seed=seed)
        U = construct_U_alpha(attach_plausibility(raw, judge),
                              data.to_semantic_dataset()).U_alpha
        if len(U) < 2:
            continue
        si = _intended_idx(U, data)
        star = U[si]
        omit_set = [c for j, c in enumerate(U) if j != si]
        rf = lambda pol: evaluate_return(MAINT, pol, n_episodes=30)
        full = learn_fqi_constrained(data, MAINT, honor=U, U_eval=U, eps=EPS,
                                     return_fn=rf, bcq_tau=0.05)
        omit = learn_fqi_constrained(data, MAINT, honor=omit_set, U_eval=U, eps=EPS,
                                     return_fn=rf, bcq_tau=0.05)
        rows.append(OmitRow(
            "maintenance", seed, len(U), star.name,
            worst_case_cost(full.policy, data, [star], normalize="active"),
            worst_case_cost(omit.policy, data, [star], normalize="active"),
            full.true_worst, omit.true_worst))
    return rows


def probe_gridworld(rule_set: str = "window_ambiguous") -> List[OmitRow]:
    from .gridworld import (CrossingGridworld, RULE_SETS_T,
                            gridworld_semantic_dataset, make_gridworld_dataset)
    from .gridworld_rl import evaluate_gridworld_return, learn_gw_constrained
    cache = str(Path(__file__).with_name("plausibility_cache_temporal.json"))
    raw = [c for c, _ in RULE_SETS_T[rule_set]["items"]]
    judge = load_llm_cache(rule_set, cache)
    rows: List[OmitRow] = []
    for seed in range(N_SEEDS):
        env = CrossingGridworld(c_accident=2.0)
        data = make_gridworld_dataset(env, seed=seed)
        U = construct_U_alpha(attach_plausibility(raw, judge),
                              gridworld_semantic_dataset(data)).U_alpha
        if len(U) < 2:
            continue
        si = _intended_idx(U, data)
        star = U[si]
        omit_set = [c for j, c in enumerate(U) if j != si]
        rf = lambda pol: evaluate_gridworld_return(env, pol, n_episodes=30)
        full = learn_gw_constrained(data, honor=U, U_eval=U, eps=EPS, return_fn=rf)
        omit = learn_gw_constrained(data, honor=omit_set, U_eval=U, eps=EPS, return_fn=rf)
        rows.append(OmitRow(
            "gridworld", seed, len(U), star.name,
            worst_case_cost(full.policy, data, [star], normalize="active"),
            worst_case_cost(omit.policy, data, [star], normalize="active"),
            full.true_worst, omit.true_worst))
    return rows


def probe_budget(rule_set: str = "budget_ambiguous") -> List[OmitRow]:
    """Domain-3 (budget agent) analogue. The retained set is the nested cap pair
    {spend>=60, spend>=80}; the most protective (widest-firing) reading is the
    tighter cap psi*=$60. Honoring U\\{psi*}={$80} stops buying only once the bill
    hits $80, so it keeps buying through the [60,80) band psi* forbids --- a real
    residual violation on psi*, while honoring all of U drives it below eps."""
    from .budget import (BudgetAgentMDP, RULE_SETS_B,
                         budget_semantic_dataset, make_budget_dataset)
    from .budget_rl import evaluate_budget_return, learn_budget_constrained
    cache = str(Path(__file__).with_name("plausibility_cache_budget.json"))
    raw = [c for c, _ in RULE_SETS_B[rule_set]["items"]]
    judge = load_llm_cache(rule_set, cache)
    rows: List[OmitRow] = []
    for seed in range(N_SEEDS):
        env = BudgetAgentMDP()
        data = make_budget_dataset(env, seed=seed)
        U = construct_U_alpha(attach_plausibility(raw, judge),
                              budget_semantic_dataset(env, data)).U_alpha
        if len(U) < 2:
            continue
        si = _intended_idx(U, data)
        star = U[si]
        omit_set = [c for j, c in enumerate(U) if j != si]
        rf = lambda pol: evaluate_budget_return(env, pol, n_episodes=30)
        full = learn_budget_constrained(data, honor=U, U_eval=U, eps=EPS, return_fn=rf)
        omit = learn_budget_constrained(data, honor=omit_set, U_eval=U, eps=EPS,
                                        return_fn=rf)
        rows.append(OmitRow(
            "budget", seed, len(U), star.name,
            worst_case_cost(full.policy, data, [star], normalize="active"),
            worst_case_cost(omit.policy, data, [star], normalize="active"),
            full.true_worst, omit.true_worst))
    return rows


def _summ(rows: List[OmitRow], dom: str):
    sub = [r for r in rows if r.domain == dom]
    if not sub:
        print(f"  {dom}: no seeds with |U_alpha|>=2")
        return
    cf = np.array([r.cost_star_full for r in sub])
    co = np.array([r.cost_star_omit for r in sub])
    n_exposed = int((co > EPS).sum())
    n_prot = int((cf <= EPS).sum())
    names = sorted({r.intended for r in sub})
    print(f"  {dom:12s} n={len(sub):2d}  intended psi* in {names}")
    print(f"    cost on psi*:  full SA-ORL = {cf.mean():.3f}+/-{cf.std():.3f} "
          f"(protected {n_prot}/{len(sub)});  "
          f"omit-intended = {co.mean():.3f}+/-{co.std():.3f} "
          f"(exposed {n_exposed}/{len(sub)})")


def main():
    rows: List[OmitRow] = []
    rows += probe_maintenance("conservative")
    rows += probe_gridworld("window_ambiguous")
    rows += probe_budget("budget_ambiguous")
    print(f"Omitted-intended-semantics ablation ({N_SEEDS} seeds, eps={EPS})\n")
    _summ(rows, "maintenance")
    _summ(rows, "gridworld")
    _summ(rows, "budget")
    out = Path(__file__).with_name("omit_intended.json")
    out.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
