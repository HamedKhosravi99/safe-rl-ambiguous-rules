"""Re-run the SA-ORL claims against the REAL LLM-ensemble plausibility scores.

This is the non-circularity stress test. Everything here uses the serialized
real-`claude` ensemble (saorl/plausibility_cache.json, produced by
saorl.build_plaus_cache) instead of hand-set numbers. It asks two questions:

  1. Does the H4 ambiguity separation survive real judgments?
     ambiguous rule -> |U_alpha| > 1, sharp rule -> |U_alpha| == 1.
  2. Does the Algorithm-3 kill test survive? i.e. with the conservatively-ordered
     rule, does a single-interpretation learner still hide a worst-case violation
     that SA-ORL removes -- using interpretations a real LLM actually endorsed?

It prints, honestly, whatever the real scores imply -- including the case where
the conservative reading is judged too implausible to retain.

Run (needs the cache):  python3 -m saorl.build_plaus_cache && python3 -m saorl.llm_demo
"""
from __future__ import annotations

from .build_plaus_cache import RULE_SETS
from .construct import construct_U_alpha
from .env import MaintenanceMDP
from .judge import attach_plausibility, load_llm_cache
from .learn import learn_constrained
from .offline import (
    epsilon_threshold_behavior,
    evaluate_return,
    generate_offline_dataset,
    worst_case_cost,
)

TENSION = MaintenanceMDP(r_op=4.0, c_fail=20.0, c_replace=40.0)
EPS = 0.05


def _candidates(set_name):
    raw = [c for c, _ in RULE_SETS[set_name]["items"]]
    return attach_plausibility(raw, load_llm_cache(set_name))


def _construct(set_name, data):
    res = construct_U_alpha(_candidates(set_name), data.to_semantic_dataset())
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
    data = generate_offline_dataset(
        TENSION, epsilon_threshold_behavior(), n_episodes=200, seed=0
    )

    print("=" * 64)
    print("=== (1) H4 ambiguity separation under REAL LLM plausibility")
    print("=" * 64)
    amb = _construct("ambiguous", data)
    sharp = _construct("sharp", data)
    h4 = amb.size() > 1 and sharp.size() == 1
    print(f"\n  H4 separation holds (amb |U|={amb.size()}>1, "
          f"sharp |U|={sharp.size()}==1): {h4}")

    print("\n" + "=" * 64)
    print("=== (2) Algorithm-3 kill test under REAL LLM plausibility")
    print("=" * 64)
    consv = _construct("conservative", data)
    U = consv.U_alpha
    if len(U) < 2:
        print(f"\n  Conservative rule retained |U|={len(U)} < 2 under real "
              f"judgments -- the conservatively-ordered substrate did NOT "
              f"survive. Kill test cannot fire; see analysis.")
        kill = False
    else:
        single = learn_constrained(data, TENSION, honor=U[:1], U_eval=U, eps=EPS)
        robust = learn_constrained(data, TENSION, honor=U, U_eval=U, eps=EPS)
        print(f"\n  {'policy':28s} {'return':>7s} {'honored':>8s} {'TRUE worst*':>12s}")
        print(f"  {'learned: honor first only':28s} {single.ret:7.1f} "
              f"{single.honored_cost:8.3f} {single.true_worst:12.3f}")
        print(f"  {'learned: honor all (SA-ORL)':28s} {robust.ret:7.1f} "
              f"{robust.honored_cost:8.3f} {robust.true_worst:12.3f}")
        kill = single.true_worst > EPS and robust.true_worst <= EPS
        print(f"  hidden violation = {single.true_worst - single.honored_cost:.3f}, "
              f"return price = {single.ret - robust.ret:+.1f}")
        print(f"\n  kill test passes (hidden violation that SA-ORL fixes): {kill}")

    print("\n" + "=" * 64)
    print(f"  REAL-LLM VERDICT: H4 holds={h4}, kill-test holds={kill}")
    print("=" * 64)
    return h4, kill


if __name__ == "__main__":
    main()
