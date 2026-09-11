"""The kill test for SA-ORL: does a single-interpretation learner hide violations?

Builds the offline dataset, constructs U_alpha for the ambiguous rule, then learns
two constrained threshold policies from the data:

  single  -- constrained to satisfy only the first retained interpretation;
  robust  -- constrained to satisfy every interpretation in U_alpha (SA-ORL).

Both are then scored by their TRUE worst-case semantic cost over the full U_alpha.
The single-interpretation learner should look compliant on its own constraint yet
carry a hidden worst-case violation; the robust learner should eliminate it, at a
bounded return cost. We run this in the TENSION regime, where honoring the rule
actually costs return (otherwise the trade-off is vacuous).

Run:  python3 -m saorl.learn_demo
"""
from __future__ import annotations

from .construct import construct_U_alpha
from .demo import conservative_rule_candidates
from .env import MaintenanceMDP
from .learn import learn_constrained
from .offline import (
    epsilon_threshold_behavior,
    evaluate_return,
    generate_offline_dataset,
    greedy_continue_policy,
    worst_case_cost,
)

TENSION = MaintenanceMDP(r_op=4.0, c_fail=20.0, c_replace=40.0)
EPS = 0.05  # semantic-cost budget the learner must meet on its honored set


def main():
    mdp = TENSION
    data = generate_offline_dataset(mdp, epsilon_threshold_behavior(), n_episodes=200)
    U = construct_U_alpha(conservative_rule_candidates(), data.to_semantic_dataset()).U_alpha
    print(f"U_alpha = {[c.name for c in U]}  (|U|={len(U)}),  budget eps={EPS}")

    greedy = greedy_continue_policy()
    g_ret = evaluate_return(mdp, greedy)
    g_worst = worst_case_cost(greedy, data, U, normalize="active")

    single = learn_constrained(data, mdp, honor=U[:1], U_eval=U, eps=EPS)
    robust = learn_constrained(data, mdp, honor=U, U_eval=U, eps=EPS)

    print(f"\n  {'policy':28s} {'return':>7s} {'honored':>8s} {'TRUE worst*':>12s}")
    print(f"  {'greedy (unconstrained)':28s} {g_ret:7.1f} {'-':>8s} {g_worst:12.3f}")
    print(f"  {'learned: honor psi1 only':28s} {single.ret:7.1f} "
          f"{single.honored_cost:8.3f} {single.true_worst:12.3f}")
    print(f"  {'learned: honor all (SA-ORL)':28s} {robust.ret:7.1f} "
          f"{robust.honored_cost:8.3f} {robust.true_worst:12.3f}")

    hidden = single.true_worst - single.honored_cost
    price = single.ret - robust.ret
    print(f"\n  hidden violation of single-interp learner = {hidden:.3f} "
          f"(true worst {single.true_worst:.3f} >> honored {single.honored_cost:.3f})")
    print(f"  SA-ORL removes it: robust true worst = {robust.true_worst:.3f} (<= eps={EPS})")
    print(f"  return price of robustness (single - robust) = {price:+.1f}")

    verdict = (
        single.true_worst > EPS            # single-interp learner is secretly unsafe
        and robust.true_worst <= EPS       # robust learner is actually safe
    )
    print(f"\n  KILL TEST -- single-interp learner hides a real violation that "
          f"SA-ORL fixes: {verdict}")
    return verdict


if __name__ == "__main__":
    main()
