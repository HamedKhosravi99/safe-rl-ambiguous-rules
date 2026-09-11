"""The offline-RL kill test on Domain 2: the temporal warning-window gridworld.

saorl.gridworld_demo establishes the construction-level claims (H4 + the oracle
respect-policy semantic gap). This runs a real offline-RL learner (tabular FQI +
Lagrangian dual ascent, saorl.gridworld_rl) end to end and shows the SAME kill
test as Domain 1, now with a genuinely TEMPORAL hidden gap:

  * the learner constrained on only the SHORT plausible window (W3) looks
    compliant on its own interpretation (honored cost ~ 0) yet keeps advancing
    through the rare TAIL of longer danger windows -- a worst-case violation over
    the full U_alpha that it hides;
  * constraining on all of U_alpha (SA-ORL, honoring the long window W5) removes
    the hidden violation at a small return price.

Interpretations come from the real `claude` LLM ensemble (cached); the
conservatively-ordered short/long pair (W3, W5) is the same pair gridworld_demo
retains. Economics: an accident is only moderately costly, so a return-greedy
learner already waits through the always-unsafe core of the danger window but is
tempted into its rare tail -- precisely the steps W3 misses and W5 covers.

Run (needs the temporal plausibility cache):
    python3 -m saorl.build_plaus_cache_temporal && python3 -m saorl.gridworld_rl_demo
"""
from __future__ import annotations

from pathlib import Path

from .construct import construct_U_alpha
from .gridworld import (
    CrossingGridworld,
    RULE_SETS_T,
    gridworld_semantic_dataset,
    make_gridworld_dataset,
)
from .gridworld_rl import evaluate_gridworld_return, learn_gw_constrained
from .judge import attach_plausibility, load_llm_cache
from .offline import worst_case_cost

EPS = 0.05
LAM_GRID = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
_CACHE = str(Path(__file__).with_name("plausibility_cache_temporal.json"))


def main():
    # Moderate accident cost: advancing through the rare tail of a long danger
    # window is return-tempting (the externality the language rule prices), while
    # the always-unsafe core is avoided by any return-greedy learner.
    env = CrossingGridworld(c_accident=2.0)
    data = make_gridworld_dataset(env, seed=0)
    dsem = gridworld_semantic_dataset(data)

    raw = [c for c, _ in RULE_SETS_T["window_conservative"]["items"]]
    U = construct_U_alpha(
        attach_plausibility(raw, load_llm_cache("window_conservative", _CACHE)),
        dsem,
    ).U_alpha
    print(f"TEMPORAL GRIDWORLD: {len(data.trajectories)} eps, {data.n_steps()} steps")
    print(f"U_alpha (real LLM ensemble) = {[c.name for c in U]}  (|U|={len(U)}),  "
          f"eps={EPS}")

    def ret_fn(pol):
        return evaluate_gridworld_return(env, pol, n_episodes=60)

    greedy = learn_gw_constrained(data, honor=[], U_eval=U, eps=EPS,
                                  return_fn=ret_fn, lam_grid=(0.0,))
    single = learn_gw_constrained(data, honor=U[:1], U_eval=U, eps=EPS,
                                  return_fn=ret_fn, lam_grid=LAM_GRID)
    robust = learn_gw_constrained(data, honor=U, U_eval=U, eps=EPS,
                                  return_fn=ret_fn, lam_grid=LAM_GRID)

    print(f"\n  {'policy':30s} {'lambda':>6s} {'return':>7s} {'honored':>8s} "
          f"{'TRUE worst*':>12s}")
    print(f"  {'greedy (unconstrained)':30s} {'-':>6s} {greedy.ret:7.1f} {'-':>8s} "
          f"{greedy.true_worst:12.3f}")
    print(f"  {'FQI: honor short W3 only':30s} {single.lam:6.0f} {single.ret:7.1f} "
          f"{single.honored_cost:8.3f} {single.true_worst:12.3f}")
    print(f"  {'FQI: honor all (SA-ORL)':30s} {robust.lam:6.0f} {robust.ret:7.1f} "
          f"{robust.honored_cost:8.3f} {robust.true_worst:12.3f}")

    hidden = single.true_worst - single.honored_cost
    price = single.ret - robust.ret
    print(f"\n  hidden violation of short-window learner = {hidden:.3f} "
          f"(true worst {single.true_worst:.3f} >> honored {single.honored_cost:.3f})")
    print(f"  SA-ORL removes it: robust true worst = {robust.true_worst:.3f} "
          f"(<= eps={EPS})")
    print(f"  return price of robustness (single - robust) = {price:+.1f}")

    verdict = single.true_worst > EPS and robust.true_worst <= EPS
    print(f"\n  TEMPORAL KILL TEST -- short-window learner hides a temporal "
          f"violation that SA-ORL fixes: {verdict}")
    return verdict


if __name__ == "__main__":
    main()
