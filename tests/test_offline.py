"""Lock the offline-dataset + policy-level semantic-gap behavior (Gate-B).
Run: python3 -m tests.test_offline
"""
from saorl.construct import construct_U_alpha
from saorl.demo import (
    ambiguous_rule_candidates,
    conservative_rule_candidates,
    sharp_rule_candidates,
)
from saorl.env import ACTIONS, MaintenanceMDP
from saorl.learn import learn_constrained
from saorl.offline import (
    epsilon_threshold_behavior,
    generate_offline_dataset,
    respect_policy,
    semantic_gap,
    worst_case_cost,
)

TENSION = MaintenanceMDP(r_op=4.0, c_fail=20.0, c_replace=40.0)


def _dataset(n=150, seed=0):
    mdp = MaintenanceMDP()
    data = generate_offline_dataset(
        mdp, epsilon_threshold_behavior(), n_episodes=n, seed=seed
    )
    return mdp, data


def test_dataset_covers_both_regimes():
    # The construction needs both incident (RUL<=20) and normal (RUL>=80) states.
    _, data = _dataset()
    dsem = data.to_semantic_dataset()
    assert len(dsem.incident_points()) > 0, "no incident coverage in offline data"
    assert len(dsem.normal_points()) > 0, "no normal coverage in offline data"


def test_construction_on_offline_data():
    _, data = _dataset()
    dsem = data.to_semantic_dataset()
    amb = construct_U_alpha(ambiguous_rule_candidates(), dsem)
    sharp = construct_U_alpha(sharp_rule_candidates(), dsem)
    assert amb.size() > 1
    assert sharp.size() == 1


def test_respect_all_eliminates_worst_case():
    _, data = _dataset()
    dsem = data.to_semantic_dataset()
    U = construct_U_alpha(ambiguous_rule_candidates(), dsem).U_alpha
    wc = worst_case_cost(respect_policy(U), data, U, normalize="active")
    assert wc == 0.0, "respecting all interpretations must zero the worst case"


def test_gap_tracks_ambiguity():
    # Policy-level H4: gap > 0 only when the rule is ambiguous (|U_alpha| > 1).
    _, data = _dataset()
    dsem = data.to_semantic_dataset()
    U_amb = construct_U_alpha(ambiguous_rule_candidates(), dsem).U_alpha
    U_sharp = construct_U_alpha(sharp_rule_candidates(), dsem).U_alpha
    gap_amb = semantic_gap(data, U_amb, normalize="active")
    gap_sharp = semantic_gap(data, U_sharp, normalize="active")
    assert gap_amb > 0.0, "ambiguous rule must leave a hidden gap"
    assert gap_sharp == 0.0, "sharp rule must have no gap"


def test_single_interp_learner_hides_violation():
    # Algorithm-3 kill test: a learner constrained on only the liberal
    # interpretation looks compliant on it (honored <= eps) yet carries a hidden
    # worst-case violation over the full U_alpha; honoring all of U removes it.
    eps = 0.05
    data = generate_offline_dataset(
        TENSION, epsilon_threshold_behavior(), n_episodes=200, seed=0
    )
    U = construct_U_alpha(
        conservative_rule_candidates(), data.to_semantic_dataset()
    ).U_alpha
    assert len(U) == 2, "kill test needs a conservatively-ordered pair"
    single = learn_constrained(data, TENSION, honor=U[:1], U_eval=U, eps=eps)
    robust = learn_constrained(data, TENSION, honor=U, U_eval=U, eps=eps)
    assert single.honored_cost <= eps, "single learner must look compliant"
    assert single.true_worst > eps, "single learner must hide a real violation"
    assert robust.true_worst <= eps, "robust learner must be actually safe"
    assert single.ret >= robust.ret, "robustness should cost (not gain) return"


def test_real_llm_ensemble_reproduces_claims():
    # Non-circularity: the *real* claude-CLI ensemble (serialized cache) must
    # reproduce BOTH claims -- H4 separation and the kill test -- with no
    # hand-set scores. Skips only if the cache has not been built.
    from pathlib import Path

    cache = Path(__file__).resolve().parents[1] / "saorl" / "plausibility_cache.json"
    if not cache.exists():
        print("  (skip test_real_llm_ensemble_reproduces_claims: no cache)")
        return
    from saorl.build_plaus_cache import RULE_SETS
    from saorl.judge import attach_plausibility, load_llm_cache

    def cands(name):
        raw = [c for c, _ in RULE_SETS[name]["items"]]
        return attach_plausibility(raw, load_llm_cache(name))

    data = generate_offline_dataset(
        TENSION, epsilon_threshold_behavior(), n_episodes=200, seed=0
    )
    dsem = data.to_semantic_dataset()
    amb = construct_U_alpha(cands("ambiguous"), dsem)
    sharp = construct_U_alpha(cands("sharp"), dsem)
    assert amb.size() > 1, "real ensemble must keep H4 ambiguity (|U|>1)"
    assert sharp.size() == 1, "real ensemble must make the sharp rule unambiguous"

    U = construct_U_alpha(cands("conservative"), dsem).U_alpha
    assert len(U) == 2, "real ensemble must retain the conservatively-ordered pair"
    eps = 0.05
    single = learn_constrained(data, TENSION, honor=U[:1], U_eval=U, eps=eps)
    robust = learn_constrained(data, TENSION, honor=U, U_eval=U, eps=eps)
    assert single.true_worst > eps and robust.true_worst <= eps, (
        "kill test must hold under real LLM judgments"
    )


def test_offline_rl_learner_kill_test():
    # The kill test must survive a REAL offline RL learner (BCQ-FQI + Lagrangian),
    # not just the threshold grid search: single-interp learner looks compliant
    # yet hides a worst-case violation that honoring all of U_alpha removes.
    from pathlib import Path

    cache = Path(__file__).resolve().parents[1] / "saorl" / "plausibility_cache.json"
    if not cache.exists():
        print("  (skip test_offline_rl_learner_kill_test: no cache)")
        return
    from saorl.build_plaus_cache import RULE_SETS
    from saorl.judge import attach_plausibility, load_llm_cache
    from saorl.offline import make_offline_rl_dataset
    from saorl.offline_rl import learn_fqi_constrained

    eps = 0.05
    lg = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0)
    data = make_offline_rl_dataset(TENSION, seed=0)
    raw = [c for c, _ in RULE_SETS["conservative"]["items"]]
    U = construct_U_alpha(
        attach_plausibility(raw, load_llm_cache("conservative")),
        data.to_semantic_dataset(),
    ).U_alpha
    assert len(U) == 2
    single = learn_fqi_constrained(
        data, TENSION, honor=U[:1], U_eval=U, eps=eps, bcq_tau=0.05, lam_grid=lg
    )
    robust = learn_fqi_constrained(
        data, TENSION, honor=U, U_eval=U, eps=eps, bcq_tau=0.05, lam_grid=lg
    )
    assert single.honored_cost <= eps, "single FQI learner must look compliant"
    assert single.true_worst > eps, "single FQI learner must hide a violation"
    assert robust.true_worst <= eps, "robust FQI learner must be actually safe"
    assert single.ret >= robust.ret, "robustness should cost (not gain) return"


def test_cql_learner_kill_test():
    # The kill test must be model-class-agnostic: a genuinely different learner
    # (neural CQL on raw continuous features -- not the tabular BCQ-FQI) must also
    # show the single-interp learner hiding a violation that SA-ORL removes.
    # Skips if torch or the plausibility cache is unavailable.
    from pathlib import Path

    try:
        import torch  # noqa: F401
    except ImportError:
        print("  (skip test_cql_learner_kill_test: torch not installed)")
        return
    cache = Path(__file__).resolve().parents[1] / "saorl" / "plausibility_cache.json"
    if not cache.exists():
        print("  (skip test_cql_learner_kill_test: no cache)")
        return
    from saorl.build_plaus_cache import RULE_SETS
    from saorl.judge import attach_plausibility, load_llm_cache
    from saorl.offline import make_offline_rl_dataset
    from saorl.neural_rl import learn_cql_constrained

    eps = 0.05
    lg = (0.0, 10.0, 40.0, 160.0)
    data = make_offline_rl_dataset(TENSION, seed=0)
    raw = [c for c, _ in RULE_SETS["conservative"]["items"]]
    U = construct_U_alpha(
        attach_plausibility(raw, load_llm_cache("conservative")),
        data.to_semantic_dataset(),
    ).U_alpha
    assert len(U) == 2
    kw = dict(eps=eps, n_steps=2500, lam_grid=lg, seed=0, return_fn=lambda p: 0.0)
    single = learn_cql_constrained(data, TENSION, honor=U[:1], U_eval=U, **kw)
    robust = learn_cql_constrained(data, TENSION, honor=U, U_eval=U, **kw)
    assert single.true_worst > eps, "neural CQL single-interp must hide a violation"
    assert robust.true_worst <= eps, "neural CQL SA-ORL must be actually safe"


def test_kill_test_robust_across_seeds():
    # The kill test must not be a lucky single draw: across several dataset seeds,
    # EVERY seed's single-interp learner hides a violation (true > eps) and EVERY
    # seed's SA-ORL learner is actually safe (true <= eps). Synthetic only (fast).
    from saorl.benchmark import synthetic_domain

    rows = synthetic_domain(seeds=(0, 1, 2))
    assert all(r.single_true > 0.05 for r in rows), "every seed must hide a violation"
    assert all(r.robust_true <= 0.05 for r in rows), "every seed's SA-ORL must be safe"
    assert all(r.single_ret >= r.robust_ret for r in rows), "robustness should cost return"


def test_mdp_step_semantics():
    mdp = MaintenanceMDP()
    # operating a dead asset fails and resets; replace resets at a fixed cost.
    r, nxt, failed = mdp.step(0.0, "continue")
    assert failed and r == -mdp.c_fail and nxt == mdp.max_life
    r, nxt, failed = mdp.step(50.0, "replace")
    assert not failed and r == -mdp.c_replace and nxt == mdp.max_life
    r, nxt, failed = mdp.step(50.0, "continue")
    assert not failed and r == mdp.r_op and nxt == 49.0
    assert set(ACTIONS) == {"continue", "minor_repair", "replace"}


def _run():
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")


if __name__ == "__main__":
    _run()
