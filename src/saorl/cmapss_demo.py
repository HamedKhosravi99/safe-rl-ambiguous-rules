"""SA-ORL construction on REAL C-MAPSS FD001 degradation (grounded Domain 1).

Re-runs the headline construction claims on real turbofan prognostic features
(saorl.cmapss) instead of the synthetic MDP, using the real `claude` LLM
plausibility ensemble (cached). It checks both:

  1. H4 ambiguity separation: ambiguous rule -> |U_alpha| > 1, sharp -> == 1;
  2. the policy-level semantic gap (respect-first vs respect-all) is positive for
     the ambiguous rule -- the hidden disagreement SA-ORL must close -- and the
     conservatively-ordered rule retains its liberal/conservative pair.

Nothing about the construction changes for real data: the DSL predicates read the
same feature keys (rul_hat/q05/anom), and the LLM scored predicate *meanings*,
which transfer to C-MAPSS's cycle-scale RUL and [0,1] anomaly.

Run (needs the feature + plausibility caches):
    python3 -m saorl.cmapss && python3 -m saorl.cmapss_demo
"""
from __future__ import annotations

from .build_plaus_cache import RULE_SETS
from .cmapss import load_features, to_trajectories
from .construct import construct_U_alpha
from .dataset import SemanticDataset
from .judge import attach_plausibility, load_llm_cache
from .offline import OfflineDataset, semantic_gap, worst_case_cost, respect_policy


def _cands(set_name):
    raw = [c for c, _ in RULE_SETS[set_name]["items"]]
    return attach_plausibility(raw, load_llm_cache(set_name))


def _construct(set_name, dsem):
    res = construct_U_alpha(_cands(set_name), dsem)
    print(f"\n  --- {set_name}: \"{RULE_SETS[set_name]['rule_text']}\" ---")
    print(f"  {'candidate':24s} {'Lsem':>6s} {'pinc':>5s} {'pnorm':>6s} "
          f"{'plaus':>5s}  status")
    for r in res.reports:
        status = "KEPT" if r.retained else f"rej: {r.reason}"
        print(f"  {r.name:24s} {r.l_sem:6.3f} {r.p_inc:5.2f} {r.p_norm:6.2f} "
              f"{r.plaus:5.2f}  {status}")
    print(f"  -> U_alpha = {res.names()} (|U|={res.size()})")
    return res


def main():
    feat = load_features()
    trajs = to_trajectories(feat)
    dsem = SemanticDataset(trajs)
    # wrap trajectories as an OfflineDataset (dummy actions) to reuse cost tooling
    dummy = OfflineDataset(
        trajs,
        [["continue"] * len(t) for t in trajs],
        [[0.0] * len(t) for t in trajs],
    )
    print(f"REAL C-MAPSS FD001: {len(trajs)} engines, {dummy.n_steps()} cycles")

    amb = _construct("ambiguous", dsem)
    sharp = _construct("sharp", dsem)
    consv = _construct("conservative", dsem)

    h4 = amb.size() > 1 and sharp.size() == 1
    gap = semantic_gap(dummy, amb.U_alpha, normalize="active")
    first = worst_case_cost(respect_policy(amb.U_alpha[:1]), dummy, amb.U_alpha,
                            normalize="active")
    allc = worst_case_cost(respect_policy(amb.U_alpha), dummy, amb.U_alpha,
                           normalize="active")

    print("\n  === real-data verdict ===")
    print(f"  H4 separation: ambiguous |U|={amb.size()} (>1), "
          f"sharp |U|={sharp.size()} (==1)  -> {h4}")
    print(f"  semantic gap (ambiguous, respect-first {first:.3f} - "
          f"respect-all {allc:.3f}) = {gap:.3f}")
    print(f"  conservatively-ordered pair retained: {consv.names()} "
          f"(|U|={consv.size()})")
    ok = h4 and gap > 0.0 and consv.size() == 2
    print(f"  ALL REAL-DATA CHECKS PASS: {ok}")
    return ok


if __name__ == "__main__":
    main()
