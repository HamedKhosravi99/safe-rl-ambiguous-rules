"""Lock the SA-ORL construction claims on REAL C-MAPSS FD001 data.
Skips cleanly if the feature/plausibility caches have not been built.
Run: python3 -m tests.test_cmapss
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "cmapss" / "features_FD001.csv"
CACHE = ROOT / "src" / "saorl" / "plausibility_cache.json"


def _ready():
    return FEATURES.exists() and CACHE.exists()


def _setup():
    from saorl.build_plaus_cache import RULE_SETS
    from saorl.cmapss import load_features, to_trajectories
    from saorl.dataset import SemanticDataset
    from saorl.judge import attach_plausibility, load_llm_cache

    trajs = to_trajectories(load_features())
    dsem = SemanticDataset(trajs)

    def U(name):
        from saorl.construct import construct_U_alpha
        raw = [c for c, _ in RULE_SETS[name]["items"]]
        return construct_U_alpha(attach_plausibility(raw, load_llm_cache(name)), dsem)

    return trajs, U


def test_h4_separation_on_real_data():
    if not _ready():
        print("  (skip test_h4_separation_on_real_data: caches not built)")
        return
    _, U = _setup()
    assert U("ambiguous").size() > 1, "ambiguous rule must stay ambiguous on real data"
    assert U("sharp").size() == 1, "sharp rule must be unambiguous on real data"


def test_conservative_pair_and_gap_on_real_data():
    if not _ready():
        print("  (skip test_conservative_pair_and_gap_on_real_data: caches not built)")
        return
    trajs, U = _setup()
    from saorl.offline import OfflineDataset, respect_policy, semantic_gap, worst_case_cost

    amb = U("ambiguous").U_alpha
    dummy = OfflineDataset(trajs, [["continue"] * len(t) for t in trajs],
                           [[0.0] * len(t) for t in trajs])
    gap = semantic_gap(dummy, amb, normalize="active")
    assert gap > 0.0, "ambiguous rule must leave a positive hidden gap on real data"
    assert worst_case_cost(respect_policy(amb), dummy, amb, normalize="active") == 0.0

    consv = U("conservative")
    assert consv.size() == 2, "conservatively-ordered pair must survive on real data"


def test_offline_rl_kill_test_on_real_degradation():
    """The BCQ-FQI + Lagrangian learner, run on a replay MDP whose every
    transition is real C-MAPSS degradation, reproduces the kill test: honoring
    only the liberal interpretation hides a worst-case violation over U_alpha
    that honoring all of U_alpha removes."""
    if not _ready():
        print("  (skip test_offline_rl_kill_test_on_real_degradation: caches not built)")
        return
    from saorl.build_plaus_cache import RULE_SETS
    from saorl.cmapss import load_features, to_trajectories
    from saorl.cmapss_env import ReplayMDP, make_replay_rl_dataset
    from saorl.construct import construct_U_alpha
    from saorl.dataset import SemanticDataset
    from saorl.judge import attach_plausibility, load_llm_cache
    from saorl.offline_rl import learn_fqi_constrained

    trajs = to_trajectories(load_features())
    env = ReplayMDP(trajs)
    data = make_replay_rl_dataset(env, seed=0)
    raw = [c for c, _ in RULE_SETS["conservative"]["items"]]
    U = construct_U_alpha(
        attach_plausibility(raw, load_llm_cache("conservative")),
        SemanticDataset(trajs),
    ).U_alpha
    assert len(U) == 2, "real-data conservative U_alpha must keep its disagreeing pair"

    lam = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0)
    # return_fn=const skips the (irrelevant-here) sim return for test speed.
    single = learn_fqi_constrained(
        data, env, honor=U[:1], U_eval=U, eps=0.05, bcq_tau=0.05,
        lam_grid=lam, return_fn=lambda p: 0.0,
    )
    robust = learn_fqi_constrained(
        data, env, honor=U, U_eval=U, eps=0.05, bcq_tau=0.05,
        lam_grid=lam, return_fn=lambda p: 0.0,
    )
    assert single.true_worst > 0.05, "single-interp learner must hide a real violation"
    assert robust.true_worst <= 0.05, "SA-ORL must remove the hidden violation"


def _run():
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")


if __name__ == "__main__":
    _run()
