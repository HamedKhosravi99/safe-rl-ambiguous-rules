"""The offline-RL kill test on REAL C-MAPSS degradation (Domain 1, Phase B).

saorl.rl_demo runs the kill test with a real BCQ-FQI + Lagrangian learner, but on
the synthetic MaintenanceMDP. This re-runs the identical learner on a replay MDP
whose every transition is replayed from real NASA turbofan run-to-failure data
(saorl.cmapss_env). Both the construction of U_alpha and the control problem are
now grounded in real degradation; nothing about the learner changes.

Interpretations come from the real `claude` LLM ensemble (cached), constructed on
the real engine trajectories (the same |U|=2 pair cmapss_demo retains). Return is
scored on the replay MDP via return_fn; policy selection is fully offline.

Expected (and asserted): the learner constrained on only the liberal
interpretation looks compliant on it yet hides a worst-case violation over the
full U_alpha; constraining on all of U_alpha (SA-ORL) removes it.

Run (needs the feature + plausibility caches):
    python3 -m saorl.cmapss && python3 -m saorl.cmapss_rl_demo
"""
from __future__ import annotations

from .build_plaus_cache import RULE_SETS
from .cmapss import load_features, to_trajectories
from .cmapss_env import ReplayMDP, evaluate_replay_return, make_replay_rl_dataset
from .construct import construct_U_alpha
from .dataset import SemanticDataset
from .judge import attach_plausibility, load_llm_cache
from .offline import greedy_continue_policy, worst_case_cost
from .offline_rl import learn_fqi_constrained

EPS = 0.05
LAM_GRID = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0)


def main():
    trajs = to_trajectories(load_features())
    env = ReplayMDP(trajs)
    data = make_replay_rl_dataset(env, seed=0)

    raw = [c for c, _ in RULE_SETS["conservative"]["items"]]
    U = construct_U_alpha(
        attach_plausibility(raw, load_llm_cache("conservative")),
        SemanticDataset(trajs),
    ).U_alpha
    print(f"REAL C-MAPSS replay MDP: {env.n_engines()} engines")
    print(f"offline-RL dataset: {len(data.trajectories)} eps, {data.n_steps()} steps")
    print(f"U_alpha (real LLM, real data) = {[c.name for c in U]}  (|U|={len(U)}),  eps={EPS}")

    def ret_fn(pol):
        return evaluate_replay_return(env, pol, n_episodes=40)

    greedy = greedy_continue_policy()
    g_ret = ret_fn(greedy)
    g_worst = worst_case_cost(greedy, data, U, normalize="active")

    single = learn_fqi_constrained(
        data, env, honor=U[:1], U_eval=U, eps=EPS, bcq_tau=0.05,
        lam_grid=LAM_GRID, return_fn=ret_fn,
    )
    robust = learn_fqi_constrained(
        data, env, honor=U, U_eval=U, eps=EPS, bcq_tau=0.05,
        lam_grid=LAM_GRID, return_fn=ret_fn,
    )

    print(f"\n  {'policy':30s} {'lambda':>6s} {'return':>7s} {'honored':>8s} "
          f"{'TRUE worst*':>12s}")
    print(f"  {'greedy (unconstrained)':30s} {'-':>6s} {g_ret:7.1f} {'-':>8s} "
          f"{g_worst:12.3f}")
    print(f"  {'FQI: honor liberal only':30s} {single.lam:6.0f} {single.ret:7.1f} "
          f"{single.honored_cost:8.3f} {single.true_worst:12.3f}")
    print(f"  {'FQI: honor all (SA-ORL)':30s} {robust.lam:6.0f} {robust.ret:7.1f} "
          f"{robust.honored_cost:8.3f} {robust.true_worst:12.3f}")

    hidden = single.true_worst - single.honored_cost
    price = single.ret - robust.ret
    print(f"\n  hidden violation of single-interp learner = {hidden:.3f} "
          f"(true worst {single.true_worst:.3f} >> honored {single.honored_cost:.3f})")
    print(f"  SA-ORL removes it: robust true worst = {robust.true_worst:.3f} "
          f"(<= eps={EPS})")
    print(f"  return price of robustness (single - robust) = {price:+.1f}")

    verdict = single.true_worst > EPS and robust.true_worst <= EPS
    print(f"\n  KILL TEST on REAL C-MAPSS degradation -- single-interp learner hides "
          f"a real violation that SA-ORL fixes: {verdict}")
    return verdict


if __name__ == "__main__":
    main()
