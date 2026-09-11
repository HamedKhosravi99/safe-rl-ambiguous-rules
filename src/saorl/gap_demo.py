"""Policy-level ambiguity tracking + the price of safety (plan v12 H4).

Two things are shown here, across two reward regimes:

1. The hidden semantic gap. Worst-case semantic cost over U_alpha among
   rule-relevant states, for three target policies:
       greedy        -- always operate (ignores semantics)
       respect-first -- honor only the first retained interpretation
       respect-all   -- honor every retained interpretation (the robust target)
   The gap = worst-case(respect-first) - worst-case(respect-all) is positive
   only when the rule is ambiguous (|U_alpha| > 1) -- the policy-level H4.

2. The price of safety. Two reward regimes are run:
       ALIGNED  -- failures are expensive; maintaining the asset also improves
                   return, so honoring the rule is essentially free.
       TENSION  -- operating revenue is high and failures cheap, so a
                   return-greedy policy *wants* to operate into the degraded band
                   and honoring the rule costs return. This is the regime that
                   actually motivates constrained robust RL.

Run:  python3 -m saorl.gap_demo
"""
from __future__ import annotations

from .construct import construct_U_alpha
from .demo import ambiguous_rule_candidates, sharp_rule_candidates
from .env import MaintenanceMDP
from .offline import (
    epsilon_threshold_behavior,
    evaluate_return,
    generate_offline_dataset,
    greedy_continue_policy,
    intervention_rate,
    respect_policy,
    semantic_gap,
    worst_case_cost,
)

ALIGNED = MaintenanceMDP()  # failures expensive -> maintenance pays for itself
# operating is profitable and a precautionary replace wastes good life (c_replace
# > c_fail), so honoring the rule -- replacing early -- costs return.
TENSION = MaintenanceMDP(r_op=4.0, c_fail=20.0, c_replace=40.0)


def _eval_rule(title, cands, mdp, data):
    dsem = data.to_semantic_dataset()
    res = construct_U_alpha(cands, dsem)
    U = res.U_alpha
    print(f"\n  --- {title}: U_alpha = {res.names()} (|U|={res.size()}) ---")

    policies = [
        ("greedy (operate)", greedy_continue_policy()),
        ("respect-first", respect_policy(U[:1])),
        ("respect-all (robust)", respect_policy(U)),
    ]
    print(f"  {'policy':24s} {'interv':>7s} {'worst*':>7s} {'return':>8s}")
    ret = {}
    for name, pol in policies:
        wc = worst_case_cost(pol, data, U, normalize="active")
        iv = intervention_rate(pol, data)
        ret[name] = evaluate_return(mdp, pol)
        print(f"  {name:24s} {iv:7.3f} {wc:7.3f} {ret[name]:8.1f}")
    gap = semantic_gap(data, U, normalize="active")
    price = ret["greedy (operate)"] - ret["respect-all (robust)"]
    print(f"  semantic gap (first - all) = {gap:.3f}   "
          f"price of safety (greedy - robust return) = {price:+.1f}")
    return gap, price


def run_regime(label, mdp):
    print(f"\n{'='*64}\n=== REGIME: {label} "
          f"(r_op={mdp.r_op}, c_fail={mdp.c_fail}, c_replace={mdp.c_replace})\n{'='*64}")
    data = generate_offline_dataset(mdp, epsilon_threshold_behavior(), n_episodes=200)
    print(f"  offline data: {len(data.trajectories)} eps, {data.n_steps()} steps, "
          f"behavior return = {data.mean_return():.1f}")
    gap_amb, price_amb = _eval_rule("Rule L1 (ambiguous)", ambiguous_rule_candidates(), mdp, data)
    gap_sharp, _ = _eval_rule("Rule L_sharp (sharp)", sharp_rule_candidates(), mdp, data)
    ok = gap_amb > 0.0 and gap_sharp == 0.0
    print(f"\n  gap tracks ambiguity: {ok}  (amb={gap_amb:.3f}>0, sharp={gap_sharp:.3f}=0)")
    return dict(label=label, gap_amb=gap_amb, gap_sharp=gap_sharp, price_amb=price_amb, ok=ok)


def main():
    aligned = run_regime("ALIGNED", ALIGNED)
    tension = run_regime("TENSION", TENSION)

    print(f"\n{'='*64}\n=== VERDICT\n{'='*64}")
    for r in (aligned, tension):
        print(f"  {r['label']:8s}: gap(amb)={r['gap_amb']:.3f}  gap(sharp)={r['gap_sharp']:.3f}  "
              f"price-of-safety={r['price_amb']:+.1f}  H4 holds={r['ok']}")
    motivated = tension["price_amb"] > 0
    print(f"\n  H4 holds in both regimes: {aligned['ok'] and tension['ok']}")
    print(f"  TENSION regime makes safety cost return (method motivated): {motivated}")
    return aligned["ok"] and tension["ok"]


if __name__ == "__main__":
    main()
