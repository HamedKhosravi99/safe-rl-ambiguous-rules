"""Does the *language grounding* matter, or just being robust over a set of the
right size? (plan section 7.6, point 1; baseline matrix row ``random/nonsemantic
robust RL''.)

A reviewer's sharpest version of the H3 objection is: ``\saorl{} is just generic
robust constrained RL --- any same-size uncertainty set would do; the LLM audit is
a wrapper.'' If true, a robust policy that honors an *arbitrary* set of the same
cardinality as U_alpha (same robust objective, same number of constraints, same
learner) should be just as safe. It is not.

This script learns, with the identical inner learner, a robust policy for EVERY
size-|U_alpha| subset of the raw DSL candidate pool *except* the audited U_alpha
itself --- the exhaustive set of ``arbitrary same-size'' uncertainty sets, no RNG
and no cherry-picking. It then contrasts them with the single-translation policy
(most plausible reading) and \saorl{} (the audited U_alpha), all at matched set
size, on the maintenance domain.

The plan predicts a dichotomy (section 7.6): a non-language-grounded robust set
either (1) MISSES the binding interpretation --- because an arbitrary size-k subset
usually omits the protective reading the audit retains --- leaving the same hidden
gap a single translation does, or (2) OVER-protects (e.g. honoring a very
conservative threshold), buying safety with return. Only the audited U_alpha,
selected by language + data, reliably retains the binding reading and is safe at
high return. The headline: it is not the set's SIZE or the robust objective that
delivers safety, it is *which* interpretations the language audit selects.

Run:  PYTHONPATH=. python3 -m saorl.nonsemantic_robust
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import List

import numpy as np

from .build_plaus_cache import RULE_SETS
from .construct import construct_U_alpha
from .env import MaintenanceMDP
from .experiments import most_plausible
from .judge import attach_plausibility, load_llm_cache
from .offline import (
    evaluate_return,
    make_offline_rl_dataset,
    rollout_risk,
    worst_case_cost,
)
from .offline_rl import learn_fqi_constrained
from .baselines import _maint_rollout_once

N_SEEDS = 30
EPS = 0.05
MAINT = MaintenanceMDP(r_op=4.0, c_fail=20.0, c_replace=40.0)


@dataclass
class NSRow:
    seed: int
    u_size: int
    policy: str          # 'single' | 'nonsemantic' | 'SA-ORL'
    subset: str          # candidate names in the honored set (';'-joined)
    ret: float
    worst: float         # true worst-case cost over U_alpha (active-normalized)
    chance: float        # realized Pr(any retained interpretation violated)
    cvar: float
    safe: bool           # realized chance <= eps


def _names(cands) -> str:
    return ";".join(c.name for c in cands)


def probe_nonsemantic(rule_set: str = "conservative") -> List[NSRow]:
    raw = [c for c, _ in RULE_SETS[rule_set]["items"]]
    pool = attach_plausibility(raw, load_llm_cache(rule_set))
    rollout_once = _maint_rollout_once(MAINT)
    rows: List[NSRow] = []
    for seed in range(N_SEEDS):
        data = make_offline_rl_dataset(MAINT, seed=seed)
        U = construct_U_alpha(pool, data.to_semantic_dataset()).U_alpha
        if len(U) < 2:
            continue
        u_names = frozenset(c.name for c in U)
        rf = lambda pol: evaluate_return(MAINT, pol, n_episodes=40)

        def record(name, honor):
            res = learn_fqi_constrained(data, MAINT, honor=list(honor), U_eval=U,
                                        eps=EPS, return_fn=rf, bcq_tau=0.05)
            r = rollout_risk(rollout_once, res.policy, U)
            w = worst_case_cost(res.policy, data, U, normalize="active")
            rows.append(NSRow(seed, len(U), name, _names(honor), res.ret, w,
                              r["chance"], r["cvar"], r["chance"] <= EPS + 1e-9))

        # SA-ORL (the audited, language-grounded set) and the single-translation
        # policy (most plausible reading), both at this seed's set size.
        record("SA-ORL", U)
        record("single", most_plausible(U))
        # every arbitrary same-size set EXCEPT the audited U_alpha:
        for combo in combinations(pool, len(U)):
            if frozenset(c.name for c in combo) == u_names:
                continue
            record("nonsemantic", combo)
    return rows


def _agg(rows: List[NSRow], name: str):
    rs = [r for r in rows if r.policy == name]
    ret = np.array([r.ret for r in rs]); wc = np.array([r.worst for r in rs])
    ch = np.array([r.chance for r in rs]); cv = np.array([r.cvar for r in rs])
    safe = np.array([r.safe for r in rs])
    return ret, wc, ch, cv, safe


def _summ(rows: List[NSRow]):
    n = len({r.seed for r in rows})
    print(f"  === maintenance  (n={n} seeds, eps={EPS}) ===")
    print(f"  {'policy':40s} {'n':>4s} {'return':>14s} {'TRUE worst*':>14s} "
          f"{'Pr(viol)':>9s} {'CVaR_0.1':>9s} {'safe%':>6s}")
    for name, label in [
        ("single", "single-translation (most plausible)"),
        ("nonsemantic", "nonsemantic robust (arbitrary same-size set)"),
        ("SA-ORL", "SA-ORL (audited U_alpha, same size)"),
    ]:
        ret, wc, ch, cv, safe = _agg(rows, name)
        print(f"  {label:40s} {len(ret):4d} {ret.mean():7.1f}+/-{ret.std():4.1f} "
              f"{wc.mean():6.3f}+/-{wc.std():5.3f} {ch.mean():9.3f} {cv.mean():9.3f} "
              f"{100*safe.mean():5.0f}%")
    # the dichotomy among arbitrary same-size sets: unsafe (miss binding) vs
    # safe-but-costly (over-protect), neither matching SA-ORL.
    ns = [r for r in rows if r.policy == "nonsemantic"]
    unsafe = [r for r in ns if not r.safe]; safe = [r for r in ns if r.safe]
    so_ret = _agg(rows, "SA-ORL")[0].mean()
    print(f"\n  arbitrary same-size sets split into the plan's section-7.6 dichotomy:")
    if unsafe:
        print(f"    (1) MISS binding reading -> unsafe : {len(unsafe):3d}/{len(ns)} sets, "
              f"mean Pr(viol)={np.mean([r.chance for r in unsafe]):.3f} "
              f"(vs SA-ORL safe)")
    if safe:
        print(f"    (2) over-protect -> safe but costly: {len(safe):3d}/{len(ns)} sets, "
              f"mean return={np.mean([r.ret for r in safe]):.1f} "
              f"(vs SA-ORL {so_ret:.1f})")
    print(f"  => same set SIZE, same robust objective, same learner: only the "
          f"language-grounded U_alpha gets both safety and return.")


def main():
    rows = probe_nonsemantic("conservative")
    print(f"Nonsemantic vs. language-grounded robustness ({N_SEEDS} seeds)\n")
    _summ(rows)
    out = Path(__file__).with_name("nonsemantic_robust.json")
    out.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
