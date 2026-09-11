"""Lock the SA-ORL construction claims on Domain 2: the temporal warning-window
gridworld. The ambiguity is a TEMPORAL window length (Within_W), not a scalar
threshold. Uses the REAL `claude` LLM plausibility ensemble (cached offline);
skips cleanly if the temporal cache has not been built.
Run: python3 -m tests.test_gridworld
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "saorl" / "plausibility_cache_temporal.json"


def _ready():
    return CACHE.exists()


def _setup():
    from saorl.construct import construct_U_alpha
    from saorl.gridworld import (
        CrossingGridworld,
        RULE_SETS_T,
        gridworld_semantic_dataset,
        make_gridworld_dataset,
    )
    from saorl.judge import attach_plausibility, load_llm_cache

    env = CrossingGridworld()
    data = make_gridworld_dataset(env, seed=0)
    dsem = gridworld_semantic_dataset(data)

    def U(name):
        raw = [c for c, _ in RULE_SETS_T[name]["items"]]
        judge = load_llm_cache(name, str(CACHE))
        return construct_U_alpha(attach_plausibility(raw, judge), dsem)

    return data, U


def test_temporal_h4_separation():
    if not _ready():
        print("  (skip test_temporal_h4_separation: temporal cache not built)")
        return
    _, U = _setup()
    assert U("window_ambiguous").size() > 1, \
        "vague 'a few steps' rule must stay temporally ambiguous (|U|>1)"
    assert U("window_sharp").size() == 1, \
        "sharp 'exactly four steps' rule must be unambiguous (|U|==1)"


def test_temporal_conservative_pair_and_gap():
    if not _ready():
        print("  (skip test_temporal_conservative_pair_and_gap: temporal cache not built)")
        return
    data, U = _setup()
    from saorl.gridworld import gw_respect_policy
    from saorl.offline import worst_case_cost

    amb = U("window_ambiguous").U_alpha
    # respecting only the shortest retained window hides a positive worst-case
    # cost in the tail of longer danger windows; respecting all of U closes it.
    first = worst_case_cost(gw_respect_policy(amb[:1]), data, amb, normalize="active")
    allc = worst_case_cost(gw_respect_policy(amb), data, amb, normalize="active")
    assert first - allc > 0.0, "ambiguous window rule must leave a positive hidden gap"
    assert allc == 0.0, "respecting all of U_alpha must remove the worst-case cost"

    consv = U("window_conservative")
    assert consv.size() == 2, \
        "conservatively-ordered short/long window pair must survive"


def test_temporal_offline_rl_kill_test():
    """A real offline-RL learner (FQI + Lagrangian) honoring only the short
    window W3 looks compliant on its own reading (honored cost ~ 0) yet keeps a
    positive TRUE worst-case violation over all of U_alpha; honoring all of
    U_alpha (SA-ORL) removes it. Mirrors the Domain-1 RL kill test."""
    if not _ready():
        print("  (skip test_temporal_offline_rl_kill_test: temporal cache not built)")
        return
    from saorl.construct import construct_U_alpha
    from saorl.gridworld import (
        CrossingGridworld,
        RULE_SETS_T,
        gridworld_semantic_dataset,
        make_gridworld_dataset,
    )
    from saorl.gridworld_rl import evaluate_gridworld_return, learn_gw_constrained
    from saorl.judge import attach_plausibility, load_llm_cache

    eps = 0.05
    env = CrossingGridworld(c_accident=2.0)
    data = make_gridworld_dataset(env, seed=0)
    dsem = gridworld_semantic_dataset(data)
    raw = [c for c, _ in RULE_SETS_T["window_conservative"]["items"]]
    U = construct_U_alpha(
        attach_plausibility(raw, load_llm_cache("window_conservative", str(CACHE))),
        dsem,
    ).U_alpha
    assert len(U) == 2, "kill test needs the retained short/long window pair"

    def ret_fn(pol):
        return evaluate_gridworld_return(env, pol, n_episodes=60)

    lam_grid = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
    single = learn_gw_constrained(data, honor=U[:1], U_eval=U, eps=eps,
                                  return_fn=ret_fn, lam_grid=lam_grid)
    robust = learn_gw_constrained(data, honor=U, U_eval=U, eps=eps,
                                  return_fn=ret_fn, lam_grid=lam_grid)
    assert single.honored_cost <= eps, "short-window learner must look compliant"
    assert single.true_worst > eps, \
        "short-window learner must hide a true worst-case temporal violation"
    assert robust.true_worst <= eps, "SA-ORL must remove the hidden violation"


def _run():
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")


if __name__ == "__main__":
    _run()
