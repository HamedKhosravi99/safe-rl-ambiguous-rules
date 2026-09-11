"""Robustness probe for H1: is the hidden gap an artifact of *which* single reading
the single-translation learner happens to honor, or is it intrinsic to honoring
only one reading of a genuinely ambiguous rule?

A reviewer can object that the paper's single-translation baseline honors the
first retained reading (U[:1]) and that some *other* single reading would already
cover the set, making the gap an artifact of an unlucky pick. This script settles
it on the conservative maintenance domain: for each seed it builds U_alpha, then
for EACH retained reading i computes the surrogate worst-case (active-normalized)
and the realized chance-of-violation of a policy that respects ONLY reading i, and
compares to respecting all of U. The result (best-single gap ~= 0 vs. U[:1] gap
large, with the widest/most-plausible reading being the only safe single choice)
is what justifies fixing the single baseline to the most plausible interpretation
in experiments.py: even the most faithful single translation hides the gap.

Run:  PYTHONPATH=. python3 -m saorl.probe_single_reading
"""
from __future__ import annotations

import numpy as np

from .build_plaus_cache import RULE_SETS
from .construct import construct_U_alpha
from .env import MaintenanceMDP
from .judge import attach_plausibility, load_llm_cache
from .offline import (
    make_offline_rl_dataset, respect_policy, worst_case_cost, rollout_risk,
)
from .baselines import _maint_rollout_once

N_SEEDS = 30
MAINT = MaintenanceMDP(r_op=4.0, c_fail=20.0, c_replace=40.0)


def main():
    raw = [c for c, _ in RULE_SETS["conservative"]["items"]]
    judge = load_llm_cache("conservative")
    rollout_once = _maint_rollout_once(MAINT)

    # accumulate per-reading: surrogate worst-case, realized chance, fires count
    per_reading = {}   # name -> list of (worst_only, chance_only)
    gap_first = []     # gap if you pick U[:1]   (the paper's "single")
    gap_best_single = []   # gap of the BEST achievable single reading (min worst)
    all_worst = []
    sizes = []

    for seed in range(N_SEEDS):
        data = make_offline_rl_dataset(MAINT, seed=seed)
        U = construct_U_alpha(attach_plausibility(raw, judge),
                              data.to_semantic_dataset()).U_alpha
        if len(U) < 2:
            continue
        sizes.append(len(U))
        w_all = worst_case_cost(respect_policy(U), data, U, normalize="active")
        all_worst.append(w_all)

        singles = []
        for i, c in enumerate(U):
            w_i = worst_case_cost(respect_policy([U[i]]), data, U, normalize="active")
            r_i = rollout_risk(rollout_once, respect_policy([U[i]]), U)
            per_reading.setdefault(c.name, []).append((w_i, r_i["chance"]))
            singles.append(w_i)
        gap_first.append(singles[0] - w_all)
        gap_best_single.append(min(singles) - w_all)

    n = len(sizes)
    print(f"conservative maintenance: {n} seeds with |U|>=2 (mean |U|={np.mean(sizes):.2f})")
    print(f"  honor-all surrogate worst (active): {np.mean(all_worst):.3f}")
    print(f"  gap of U[:1] (paper 'single'):      {np.mean(gap_first):.3f} "
          f"+/- {np.std(gap_first):.3f}")
    print(f"  gap of BEST single reading:         {np.mean(gap_best_single):.3f} "
          f"+/- {np.std(gap_best_single):.3f}")
    print()
    print("  per-reading (respect ONLY this reading):")
    print(f"    {'reading':28s} {'n':>3s} {'surr worst':>11s} {'realized chance':>16s}")
    for name, vals in per_reading.items():
        w = np.array([v[0] for v in vals]); ch = np.array([v[1] for v in vals])
        print(f"    {name:28s} {len(vals):3d} {w.mean():11.3f} {ch.mean():16.3f}")


if __name__ == "__main__":
    main()
