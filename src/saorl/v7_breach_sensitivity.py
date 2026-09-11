"""Post-hoc V7 sensitivity #2: the return-matched breach multiple for the
budget domain under the enlarged deployed set (review #23, question 5).

The paper's headline row (Table alt) reports, per domain, the minimal
budget multiple t such that a policy honoring EVERY retained reading at
t*d matches the most-plausible singleton's return: 2.58d for the budget
domain under the deployed set {$60,$80}. REGISTRATION_V7's recalibration
enlarges that set to {$30,$45,$60,$80} and reorders plausibility within
it (the old most-plausible $60 is now the least plausible retained cap).
This diagnostic -- labeled post-hoc, run because of the V7 outcome --
recomputes t under the enlarged set for both conventions: the V7
scorer's most-plausible retained cap, and the originally deployed $60.

Run:  PYTHONPATH=. python3 -m saorl.v7_breach_sensitivity
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from .alt_defense import (
    GREEDY_ACTION,
    _bisect_t,
    _budget_pool,
    _solve,
    build_budget,
    fixed_psi_Z,
)

OUT = Path(__file__).parent.parent.parent / "results/e2e" / "v7_breach_sensitivity.json"
V7_SET = ["psi_$30: spend>=30", "psi_$45: spend>=45",
          "psi_$60: spend>=60", "psi_$80: spend>=80"]
D = 0.05


def main() -> None:
    from .experiments import _domain_specs

    specs = _domain_specs(["budget"], ["fqi"])
    _data, env, _U_old, _ret, _pool = specs["budget"].build(0)
    pool_cands = _budget_pool()
    m = build_budget(env, pool_cands)
    pool = [c.name for c in pool_cands]
    a_greedy = m.action_names.index(GREEDY_ACTION["budget"])
    Z = fixed_psi_Z(m, pool, a_greedy)

    v7 = json.load(open(Path(__file__).parent / "plausibility_cache_v7.json"))
    plaus = {k: st.mean(v) for k, v in
             v7["units"]["orig:budget_ambiguous"]["scores"].items()}

    retained = V7_SET
    most_pl_v7 = max(retained, key=lambda k: plaus[k])
    out = dict(registration="REGISTRATION_V7 post-hoc sensitivity #2 "
                            "(not pre-registered; review #23 Q5)",
               retained=retained, budget=D,
               plaus_v7={k: round(plaus[k], 4) for k in retained},
               rows={})
    for label, single in (("v7_most_plausible", most_pl_v7),
                          ("deployed_most_plausible", "psi_$60: spend>=60")):
        v_single = _solve(m, [single], D, pool, Z).value
        t = _bisect_t(m, retained, pool, Z, D, v_single)
        out["rows"][label] = dict(singleton=single,
                                  v_single=round(v_single, 6),
                                  t_at_single=round(t, 4) if t is not None else None)
        print(f"{label}: singleton={single}  v={v_single:.3f}  "
              f"t_at_single={t}")
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
