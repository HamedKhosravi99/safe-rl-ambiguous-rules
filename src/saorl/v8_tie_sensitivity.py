"""Does the C-MAPSS headline survive dropping the reading retained on a tie?

Review #21 Q1 and review #28 W9 ask the same thing: the maintenance set
retains RUL<40 because its ensemble mean landed at exactly 0.500 = qhat and
the frozen convention includes ties, and the 18.94x breach multiple and the
28.35 certified price are both computed over that set. This recomputes both
with RUL<40 removed -- the answer to "what if the tie had gone the other
way" -- holding every other convention fixed.

Post-hoc and labeled as such; not pre-registered.

Run:  SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.v8_tie_sensitivity
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict

from .alt_defense import GREEDY_ACTION, _bisect_t, _solve, build_real, fixed_psi_Z
from .shadow_price import dp_optimal

OUT = Path(__file__).parent.parent.parent / "results/e2e" / "v8_tie_sensitivity.json"
TIE = "vconsv: RUL<40"
D = 0.05


def main() -> None:
    from .experiments import _domain_specs

    specs = _domain_specs(["real"], ["fqi"])
    _data, env, U, _ret, _pool = specs["real"].build(0)
    full = [c.name for c in U]
    assert TIE in full, f"{TIE} not in deployed set {full}"
    kept = [c for c in U if c.name != TIE]
    print(f"deployed set {full}\nwithout the tie  {[c.name for c in kept]}")

    out: Dict[str, dict] = {}
    for tag, cands in (("deployed", U), ("tie_dropped", kept)):
        m = build_real(env, cands)
        names = [c.name for c in cands]
        Z = fixed_psi_Z(m, names, m.action_names.index(GREEDY_ACTION["real"]))
        V = {frozenset(): dp_optimal(m)}
        for r in range(1, len(names) + 1):
            for T in itertools.combinations(names, r):
                V[frozenset(T)] = _solve(m, list(T), D, names, Z).value
        singles = {n: V[frozenset([n])] for n in names}
        v_max = max(singles.values())
        v_full = V[frozenset(names)]
        # the paper's headline: budget multiple at which honoring EVERY reading
        # matches the most-plausible singleton's return
        most_pl = max(cands, key=lambda c: c.plaus_mean()).name
        t = _bisect_t(m, names, names, Z, D, singles[most_pl])
        out[tag] = dict(
            readings=names, most_plausible=most_pl,
            v_max_single=round(v_max, 4), v_full=round(v_full, 4),
            poa=round(v_max - v_full, 4),
            rel_price_pct=round(100.0 * (v_max - v_full) / v_max, 3),
            breach_multiple=round(t, 4) if t is not None else None)
        print(f"[{tag}] price {out[tag]['poa']}  rel {out[tag]['rel_price_pct']}%  "
              f"breach x{out[tag]['breach_multiple']}  (most plausible {most_pl})")

    json.dump(dict(note="post-hoc tie sensitivity (review #21 Q1, #28 W9); "
                        "not pre-registered", tie_reading=TIE, budget=D, **out),
              open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
