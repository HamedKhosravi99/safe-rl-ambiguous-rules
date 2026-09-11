"""The Algorithm-3 kill test with a REAL offline RL learner (not a threshold grid).

saorl.learn_demo runs the kill test with a 2-parameter threshold policy chosen by
grid search -- transparent, but a reviewer will ask whether the result is an
artifact of that tiny policy class. This re-runs the same test with an actual
offline value-based learner: batch-constrained fitted-Q iteration (discrete BCQ)
with a Lagrangian on the honored semantic cost, learning ONLY from logged
transitions (saorl.offline_rl). Return is simulated for reporting; policy
selection is fully offline.

Interpretations come from the real `claude` LLM ensemble (cached). The dataset is
heterogeneous so the logging policy actually demonstrates the honoring action
across the rule's firing region -- the offline-coverage condition under which
honoring is even learnable (see make_offline_rl_dataset).

Result we expect (and assert): a learner constrained on only the liberal
interpretation looks compliant on it yet hides a large worst-case violation over
the full U_alpha; constraining over all of U_alpha (SA-ORL) removes it, at a
bounded return price.

Run:  python3 -m saorl.rl_demo
"""
from __future__ import annotations

from .build_plaus_cache import RULE_SETS
from .construct import construct_U_alpha
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


def main():
    mdp = TENSION
    data = make_offline_rl_dataset(mdp, seed=0)
    raw = [c for c, _ in RULE_SETS["conservative"]["items"]]
    U = construct_U_alpha(
        attach_plausibility(raw, load_llm_cache("conservative")),
        data.to_semantic_dataset(),
    ).U_alpha
    print(f"offline-RL dataset: {len(data.trajectories)} eps, {data.n_steps()} steps")
    print(f"U_alpha (real LLM) = {[c.name for c in U]}  (|U|={len(U)}),  eps={EPS}")

    greedy = greedy_continue_policy()
    g_ret = evaluate_return(mdp, greedy)
    g_worst = worst_case_cost(greedy, data, U, normalize="active")

    single = learn_fqi_constrained(
        data, mdp, honor=U[:1], U_eval=U, eps=EPS, bcq_tau=0.05, lam_grid=LAM_GRID
    )
    robust = learn_fqi_constrained(
        data, mdp, honor=U, U_eval=U, eps=EPS, bcq_tau=0.05, lam_grid=LAM_GRID
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
    print(f"\n  KILL TEST (real offline RL learner) -- single-interp learner hides "
          f"a real violation that SA-ORL fixes: {verdict}")
    return verdict


if __name__ == "__main__":
    main()
