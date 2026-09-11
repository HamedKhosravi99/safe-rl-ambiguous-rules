"""SA-ORL construction on Domain 2: the temporal warning-window gridworld.

Re-runs the headline construction claims on a genuinely *temporal* ambiguity --
"after a warning, hold for the first few steps", where the ambiguous quantity is
the WINDOW LENGTH (a past-time Within_W operator), not a scalar feature threshold.
It checks, with the REAL `claude` LLM plausibility ensemble (cached offline):

  1. H4 ambiguity separation: the vague "a few steps" rule -> |U_alpha| > 1, the
     sharp "exactly four steps" rule -> |U_alpha| == 1;
  2. the policy-level semantic gap (respect-shortest vs respect-all) is positive
     for the ambiguous rule -- the hidden tail-of-the-danger-window disagreement
     SA-ORL must close -- and the conservatively-ordered short/long pair is
     retained.

The ambiguity is temporal in its logic structure: the retained readings differ in
Within_W window length, and they are conservatively ordered (Within_3 subset
Within_4 subset Within_5) so honoring only the short window hides violations in
the tail of longer danger windows.

Run (needs the temporal plausibility cache):
    python3 -m saorl.build_plaus_cache_temporal && python3 -m saorl.gridworld_demo
"""
from __future__ import annotations

from pathlib import Path

from .construct import construct_U_alpha
from .gridworld import (
    CrossingGridworld,
    RULE_SETS_T,
    gridworld_semantic_dataset,
    gw_respect_policy,
    make_gridworld_dataset,
)
from .judge import attach_plausibility, load_llm_cache
from .offline import worst_case_cost

_CACHE = str(Path(__file__).with_name("plausibility_cache_temporal.json"))


def _cands(set_name):
    raw = [c for c, _ in RULE_SETS_T[set_name]["items"]]
    return attach_plausibility(raw, load_llm_cache(set_name, _CACHE))


def _construct(set_name, dsem):
    res = construct_U_alpha(_cands(set_name), dsem)
    print(f"\n  --- {set_name}: \"{RULE_SETS_T[set_name]['rule_text']}\" ---")
    print(f"  {'candidate':28s} {'Lsem':>6s} {'pinc':>5s} {'pnorm':>6s} "
          f"{'plaus':>5s}  status")
    for r in res.reports:
        status = "KEPT" if r.retained else f"rej: {r.reason}"
        print(f"  {r.name:28s} {r.l_sem:6.3f} {r.p_inc:5.2f} {r.p_norm:6.2f} "
              f"{r.plaus:5.2f}  {status}")
    print(f"  -> U_alpha = {res.names()} (|U|={res.size()})")
    return res


def main():
    env = CrossingGridworld()
    data = make_gridworld_dataset(env, seed=0)
    dsem = gridworld_semantic_dataset(data)
    inc = dsem.incident_points()
    print(f"TEMPORAL GRIDWORLD: {len(data.trajectories)} episodes, "
          f"{data.n_steps()} steps, incident-rate {len(inc)/data.n_steps():.3f}")

    amb = _construct("window_ambiguous", dsem)
    sharp = _construct("window_sharp", dsem)
    consv = _construct("window_conservative", dsem)

    h4 = amb.size() > 1 and sharp.size() == 1
    U = amb.U_alpha
    first = worst_case_cost(gw_respect_policy(U[:1]), data, U, normalize="active")
    allc = worst_case_cost(gw_respect_policy(U), data, U, normalize="active")
    gap = first - allc

    print("\n  === temporal-domain verdict ===")
    print(f"  H4 separation: ambiguous |U|={amb.size()} (>1), "
          f"sharp |U|={sharp.size()} (==1)  -> {h4}")
    print(f"  semantic gap (ambiguous, respect-shortest {first:.3f} - "
          f"respect-all {allc:.3f}) = {gap:.3f}")
    print(f"  conservatively-ordered short/long pair retained: {consv.names()} "
          f"(|U|={consv.size()})")
    ok = h4 and gap > 0.0 and consv.size() == 2
    print(f"  ALL TEMPORAL-DOMAIN CHECKS PASS: {ok}")
    return ok


if __name__ == "__main__":
    main()
