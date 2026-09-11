"""Multi-seed, baselined benchmark of the SA-ORL kill test (the headline figure).

The per-demo kill tests (rl_demo, cmapss_rl_demo) report a single seed. A reviewer
needs (a) error bars over seeds -- to rule out a lucky draw -- and (b) an explicit
baseline that SA-ORL is measured against. This module supplies both, on both
domains, with one shared protocol.

Three policies per seed, all learned by the SAME offline learner (BCQ-FQI +
Lagrangian, saorl.offline_rl); only the constraint set differs:

  * UNCONSTRAINED   greedy/operate -- ignores the rule entirely (return ceiling);
  * SINGLE-INTERP   the naive language->constraint pipeline: take the single most
    plausible interpretation and constrain on it (the real competitor);
  * SA-ORL          constrain on the worst case over the whole audited U_alpha.

The claim, now as a distribution over seeds: SINGLE-INTERP looks compliant on its
own reading yet leaves a large TRUE worst-case cost over U_alpha; SA-ORL drives
that true cost under eps, at a bounded, reported return price.

Run:  python3 -m saorl.benchmark            (synthetic always; real if caches built)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .build_plaus_cache import RULE_SETS
from .construct import construct_U_alpha
from .dsl import Candidate
from .env import MaintenanceMDP
from .judge import attach_plausibility, load_llm_cache
from .offline import (
    evaluate_return,
    greedy_continue_policy,
    make_offline_rl_dataset,
    worst_case_cost,
)
from .offline_rl import learn_fqi_constrained

TENSION = MaintenanceMDP(r_op=4.0, c_fail=20.0, c_replace=40.0)
EPS = 0.05
LAM_GRID = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0)
BCQ_TAU = 0.05


@dataclass
class SeedRow:
    seed: int
    greedy_ret: float
    greedy_true: float
    single_ret: float
    single_honored: float
    single_true: float
    single_lam: float
    robust_ret: float
    robust_honored: float
    robust_true: float
    robust_lam: float


def _mean_std(xs: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std())


def run_domain(
    make_data: Callable[[int], "OfflineDataset"],  # noqa: F821
    env,                                            # learner's MDP/replay env
    make_U: Callable[["OfflineDataset"], List[Candidate]],  # noqa: F821
    return_fn: Callable,                            # policy -> scalar return
    seeds: Sequence[int],
    eps: float = EPS,
    lam_grid: Sequence[float] = LAM_GRID,
    bcq_tau: float = BCQ_TAU,
) -> List[SeedRow]:
    """Run the three-policy protocol over `seeds`; one SeedRow per seed."""
    rows: List[SeedRow] = []
    for seed in seeds:
        data = make_data(seed)
        U = make_U(data)
        greedy = greedy_continue_policy()
        g_ret = return_fn(greedy)
        g_true = worst_case_cost(greedy, data, U, normalize="active")
        single = learn_fqi_constrained(
            data, env, honor=U[:1], U_eval=U, eps=eps, bcq_tau=bcq_tau,
            lam_grid=lam_grid, return_fn=return_fn,
        )
        robust = learn_fqi_constrained(
            data, env, honor=U, U_eval=U, eps=eps, bcq_tau=bcq_tau,
            lam_grid=lam_grid, return_fn=return_fn,
        )
        rows.append(SeedRow(
            seed, g_ret, g_true,
            single.ret, single.honored_cost, single.true_worst, single.lam,
            robust.ret, robust.honored_cost, robust.true_worst, robust.lam,
        ))
    return rows


def summarize(name: str, rows: List[SeedRow], eps: float = EPS) -> bool:
    """Print a mean +/- std table and return the across-seed kill-test verdict."""
    def col(attr):
        return _mean_std([getattr(r, attr) for r in rows])

    print(f"\n  === {name}  (n={len(rows)} seeds) ===")
    print(f"  {'policy':22s} {'return':>15s} {'honored cost':>16s} {'TRUE worst*':>16s}")
    gm, gs = col("greedy_ret"); gtm, gts = col("greedy_true")
    print(f"  {'unconstrained':22s} {gm:7.1f} +/-{gs:4.1f} {'--':>16s} "
          f"{gtm:8.3f} +/-{gts:5.3f}")
    sm, ss = col("single_ret"); shm, shs = col("single_honored"); stm, sts = col("single_true")
    print(f"  {'single-interp':22s} {sm:7.1f} +/-{ss:4.1f} {shm:8.3f} +/-{shs:5.3f} "
          f"{stm:8.3f} +/-{sts:5.3f}")
    rm, rs = col("robust_ret"); rhm, rhs = col("robust_honored"); rtm, rts = col("robust_true")
    print(f"  {'SA-ORL (ours)':22s} {rm:7.1f} +/-{rs:4.1f} {rhm:8.3f} +/-{rhs:5.3f} "
          f"{rtm:8.3f} +/-{rts:5.3f}")

    hidden = stm - shm
    price_m, price_s = _mean_std([r.single_ret - r.robust_ret for r in rows])
    print(f"  -> single-interp hidden violation (true - honored) = {hidden:+.3f}")
    print(f"  -> SA-ORL true worst {rtm:.3f} <= eps={eps}: {rtm <= eps}")
    print(f"  -> return price of robustness = {price_m:+.1f} +/- {price_s:.1f}")
    # across-seed verdict: EVERY seed must show the separation (not just the mean)
    all_single_violate = all(r.single_true > eps for r in rows)
    all_robust_safe = all(r.robust_true <= eps for r in rows)
    verdict = all_single_violate and all_robust_safe
    print(f"  -> KILL TEST holds on ALL {len(rows)} seeds: {verdict}")
    return verdict


# --- domain builders ---------------------------------------------------------
def _conservative_cands() -> List[Candidate]:
    raw = [c for c, _ in RULE_SETS["conservative"]["items"]]
    return attach_plausibility(raw, load_llm_cache("conservative"))


def synthetic_domain(seeds: Sequence[int]) -> List[SeedRow]:
    """Domain 1 on the synthetic TENSION MaintenanceMDP."""
    mdp = TENSION
    cands = _conservative_cands()

    def make_data(seed):
        return make_offline_rl_dataset(mdp, seed=seed)

    def make_U(data):
        return construct_U_alpha(cands, data.to_semantic_dataset()).U_alpha

    def return_fn(pol):
        return evaluate_return(mdp, pol, n_episodes=40)

    return run_domain(make_data, mdp, make_U, return_fn, seeds)


def real_cmapss_domain(seeds: Sequence[int]) -> Optional[List[SeedRow]]:
    """Domain 1 on the REAL C-MAPSS replay MDP (None if caches are not built)."""
    try:
        from .cmapss import load_features, to_trajectories
        from .cmapss_env import ReplayMDP, evaluate_replay_return, make_replay_rl_dataset
        from .dataset import SemanticDataset
        trajs = to_trajectories(load_features())
    except (FileNotFoundError, ImportError):
        return None

    env = ReplayMDP(trajs)
    cands = _conservative_cands()
    U_fixed = construct_U_alpha(cands, SemanticDataset(trajs)).U_alpha  # data-seed-independent

    def make_data(seed):
        return make_replay_rl_dataset(env, seed=seed)

    def make_U(_data):
        return U_fixed

    def return_fn(pol):
        return evaluate_replay_return(env, pol, n_episodes=40)

    return run_domain(make_data, env, make_U, return_fn, seeds)


def main(synth_seeds=(0, 1, 2, 3, 4), real_seeds=(0, 1, 2)):
    print("SA-ORL kill-test benchmark (multi-seed, baselined)")
    verdicts = []

    syn = synthetic_domain(synth_seeds)
    verdicts.append(summarize("SYNTHETIC MaintenanceMDP (Domain 1)", syn))

    real = real_cmapss_domain(real_seeds)
    if real is None:
        print("\n  (real C-MAPSS domain skipped: feature/plausibility caches not built)")
    else:
        verdicts.append(summarize("REAL C-MAPSS replay MDP (Domain 1, grounded)", real))

    ok = all(verdicts)
    print(f"\n  ALL DOMAINS PASS ACROSS SEEDS: {ok}")
    return ok


if __name__ == "__main__":
    main()
