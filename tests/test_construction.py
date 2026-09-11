"""Lock the SA-ORL semantic-core behavior. Run: python3 -m tests.test_construction"""
from saorl import (
    Atom,
    CachedEnsembleJudge,
    Candidate,
    DataDrivenJudge,
    Not,
    Persist,
    SemanticDataset,
    attach_plausibility,
    construct_U_alpha,
)
from saorl.demo import (
    ambiguous_rule_candidates,
    make_trajectories,
    sharp_rule_candidates,
)

FORBID = frozenset({"continue"})


def test_ambiguity_tracking():
    dsem = SemanticDataset(make_trajectories())
    amb = construct_U_alpha(ambiguous_rule_candidates(), dsem)
    sharp = construct_U_alpha(sharp_rule_candidates(), dsem)
    assert amb.size() > 1, "ambiguous rule must retain multiple interpretations"
    assert sharp.size() == 1, "sharp rule must collapse to one interpretation"


def test_empty_set_diagnostic():
    # An all-zero language ensemble -> below the plausibility gate -> empty set.
    dsem = SemanticDataset(make_trajectories())
    cands = [
        Candidate("c", Atom("rul_hat", "le", 20), FORBID, (0.0, 0.0, 0.0)),
    ]
    res = construct_U_alpha(cands, dsem)
    assert res.empty_diagnostic and res.size() == 0


def test_plausibility_gate():
    # Mean < TAU_PLAUS rejects; mean >= TAU_PLAUS passes the language gate.
    dsem = SemanticDataset(make_trajectories())
    weak = Candidate("weak", Atom("rul_hat", "le", 15), FORBID, (0.4, 0.4, 0.5))
    strong = Candidate("strong", Atom("rul_hat", "le", 15), FORBID, (0.9, 0.9, 0.9))
    assert construct_U_alpha([weak], dsem).size() == 0
    assert construct_U_alpha([strong], dsem).size() == 1


def test_judge_attach():
    # attach_plausibility populates the ensemble; DataDrivenJudge is data-only.
    dsem = SemanticDataset(make_trajectories())
    raw = Candidate("c", Atom("rul_hat", "le", 15), FORBID)
    assert not raw.has_plausibility()
    cached = attach_plausibility(
        [raw], CachedEnsembleJudge({"c": (0.8, 0.9, 0.7)})
    )[0]
    assert cached.has_plausibility()
    assert abs(cached.plaus_mean() - 0.8) < 1e-9
    dd = attach_plausibility([raw], DataDrivenJudge(dsem, n_members=5))[0]
    assert len(dd.plausibility) == 5
    assert 0.0 <= dd.plaus_mean() <= 1.0


def test_cost_semantics():
    traj = [dict(rul_hat=10.0, q05=5.0, anom=0.9, rul_true=10.0)]
    c = Candidate("c", Atom("rul_hat", "le", 20), FORBID, (1.0, 1.0, 1.0))
    assert c.cost(traj, 0, "continue") == 1  # predicate fires, action forbidden
    assert c.cost(traj, 0, "replace") == 0   # action not forbidden
    c2 = Candidate("c2", Atom("rul_hat", "le", 5), FORBID, (1.0, 1.0, 1.0))
    assert c2.cost(traj, 0, "continue") == 0  # predicate does not fire


def test_persist_and_not():
    traj = [dict(rul_hat=v, rul_true=v, q05=v, anom=0.0) for v in (30, 24, 23, 22)]
    p = Persist(3, Atom("rul_hat", "le", 25))
    assert not p.holds(traj, 1)  # window includes index 0 (rul=30, fails)
    assert p.holds(traj, 3)      # indices 1,2,3 all <= 25
    assert Not(Atom("rul_hat", "le", 25)).holds(traj, 0)


def _run():
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")


if __name__ == "__main__":
    _run()
