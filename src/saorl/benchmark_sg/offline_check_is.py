"""Is the barrier specific to the MODEL-BASED route? (control for section 10)

The incumbent CHECK is model-free: empirical Bernstein on samples drawn from
pi_hat's own occupancy. Its offline analogue is importance weighting of the
held-out log, with per-(s,a) ratio w = x_pihat / x_behaviour. That estimator is
usable only if w is bounded and the effective sample size is non-trivial.

This script computes the ratio EXACTLY in the true model (experimental scoring
only) so the answer does not depend on any estimator. If sup w = inf, then no
reweighting of the behaviour log can evaluate pi_hat, and the negative result
is a property of the DATA, not of the robust-Bellman construction.
"""
from __future__ import annotations

import json
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import numpy as np

from .control_suite import _eligible, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool
from .safe_face_offline import (BUDGET, SEL, Log, behaviour_policy, sample_log,
                                score, occupancy)
from .offline_check_run import learn_all

_OUT = Path(__file__).resolve().parents[3] / "results" / "offline_check_experiment"
SEEDS = tuple(range(5))
N_TRAIN = 20000


def main():
    sel = json.load(open(SEL))
    mid = [r for r in sel["rows"] if r["regime"] == "optimizer_resolvable"
           and abs(r["budget"] - BUDGET) < 1e-12]
    pt, _ = parse_prometheus(); bank = prom_threshold_bank(pt)
    comp = {}
    for ti, t in enumerate(pt):
        ok, _w, rd = _eligible(build_prom_pool(t, bank))
        if ok:
            comp[f'{getattr(t, "name", "?")}#{ti}'] = compile_instance(rd)

    rows = []
    for row in mid:
        m = comp[row["rule_id"]]
        P, mu0, r, C = m["P"], m["mu0"], m["r"], m["C"]
        pi_b = behaviour_policy(m, row["anchor"], BUDGET)
        xb = occupancy(P, mu0, pi_b)
        for seed in SEEDS:
            rng = np.random.default_rng(
                abs(hash((row["rule_id"], "train", seed))) % (2 ** 32))
            pols = learn_all(Log(sample_log(m, pi_b, N_TRAIN, rng)),
                             row["anchor"], BUDGET)
            for name, pol in pols.items():
                xp = occupancy(P, mu0, pol)
                sup = xp > 1e-12
                zero_b = (xb <= 1e-12) & sup
                if zero_b.any():
                    wmax = float("inf")
                    ess = 0.0
                else:
                    w = np.zeros_like(xp)
                    w[sup] = xp[sup] / xb[sup]
                    wmax = float(w.max())
                    # ESS of self-normalised IS under the behaviour occupancy
                    ess = float(1.0 / ((xb * (w ** 2)).sum()))
                rows.append(dict(
                    rule_id=row["rule_id"], seed=seed, arm=name,
                    true_safe=bool(score(P, mu0, r, C, pol, BUDGET)["safe"]),
                    max_ratio=wmax, ess_fraction=ess,
                    mass_unreachable=float(xp[zero_b].sum()),
                    has_zero_support=bool(zero_b.any())))
        print(f"  {row['rule_id']:38s} done", flush=True)

    n = len(rows)
    inf_rows = [r for r in rows if r["has_zero_support"]]
    fin = [r for r in rows if not r["has_zero_support"]]
    out = dict(
        n_runs=n,
        frac_with_UNBOUNDED_importance_ratio=float(len(inf_rows) / n),
        median_pihat_mass_on_unreachable_pairs=float(np.median(
            [r["mass_unreachable"] for r in inf_rows])) if inf_rows else None,
        among_finite=dict(
            n=len(fin),
            median_max_ratio=float(np.median([r["max_ratio"] for r in fin])) if fin else None,
            median_ess_fraction=float(np.median([r["ess_fraction"] for r in fin])) if fin else None),
        verdict=("Unbounded importance ratio means NO reweighting of the "
                 "behaviour log identifies pi_hat's cost: the barrier is the "
                 "data, not the robust-Bellman construction."))
    (_OUT / "importance_ratio_control.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
