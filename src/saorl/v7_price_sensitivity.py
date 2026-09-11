"""Post-hoc V7 sensitivity: the certified budget-domain price under the
enlarged deployed set.

REGISTRATION_V7's primary endpoint found budget-ambiguous enlarging from
{$60,$80} to {$30,$45,$60,$80} under the expanded corpus + current scorer
(gold retained; the set got more conservative). This follow-on diagnostic
-- not itself pre-registered, run after and because of that outcome --
recomputes the exact subset-LP price for the budget domain with the
enlarged set, holding everything else identical to saorl/shadow_price.py
(same builder, same greedy-action normalizer, same d).

Run:  PYTHONPATH=. python3 -m saorl.v7_price_sensitivity
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict

from .shadow_price import GREEDY_ACTION, _solve, build_budget, dp_optimal, fixed_psi_Z

OUT = Path(__file__).parent.parent.parent / "results/e2e" / "v7_price_sensitivity.json"
V7_SET = ["psi_$30: spend>=30", "psi_$45: spend>=45",
          "psi_$60: spend>=60", "psi_$80: spend>=80"]
D = 0.05


def main() -> None:
    from .experiments import _domain_specs
    from .budget import RULE_SETS_B

    specs = _domain_specs(["budget"], ["fqi"])
    _data, env, U_old, _ret, _pool = specs["budget"].build(0)
    by = {c.name: c for c, _g in RULE_SETS_B["budget_ambiguous"]["items"]}
    missing = [n for n in V7_SET if n not in by]
    assert not missing, f"pool lacks {missing}"
    U = [by[n] for n in V7_SET]

    m = build_budget(env, U)
    names = [c.name for c in U]
    M = len(names)
    Z = fixed_psi_Z(m, names, m.action_names.index(GREEDY_ACTION["budget"]))

    print(f"[budget/V7] M={M}; solving {2 ** M - 1} constrained programs ...")
    V: Dict[frozenset, float] = {frozenset(): dp_optimal(m)}
    for r in range(1, M + 1):
        for T in itertools.combinations(names, r):
            V[frozenset(T)] = _solve(m, list(T), D, names, Z).value

    singles = {n: V[frozenset([n])] for n in names}
    v_max = max(singles.values())
    v_full = V[frozenset(names)]
    poa = v_max - v_full
    rel = 100.0 * poa / v_max

    old_names = [c.name for c in U_old]
    v_max_old = max(singles[n] for n in old_names)
    v_old_full = V[frozenset(old_names)]

    out = dict(
        registration="REGISTRATION_V7 post-hoc sensitivity (not pre-registered)",
        deployed_old=old_names, deployed_v7=names, budget=D,
        V_by_subset={"+".join(sorted(T)) or "(none)": round(v, 6)
                     for T, v in V.items()},
        v_max_single=round(v_max, 6), v_full_v7=round(v_full, 6),
        poa_v7=round(poa, 6), rel_price_v7_pct=round(rel, 3),
        v_full_old_subset=round(v_old_full, 6),
        rel_price_old_pct=round(100.0 * (v_max_old - v_old_full) / v_max_old, 3),
    )
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"old set {old_names}: rel price {out['rel_price_old_pct']}%")
    print(f"V7 set  {names}: rel price {out['rel_price_v7_pct']}%")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
