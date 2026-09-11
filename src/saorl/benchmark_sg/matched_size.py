"""Is conformal calibration better than a TUNED constant at matched conservatism?

The tau=0.5 comparison in calib_vs_fixed.py is structurally one-sided: every
leave-one-out threshold lands below the neutral tie value, so the fixed set is
contained in the calibrated one and coverage can only go up. The honest test
holds SET SIZE fixed and asks whether the data-chosen threshold covers more
than the best constant achieving the same mean size.

For each family and level we sweep tau over the observed score grid, pick the
tau whose mean retained-set size is closest to the conformal mean, and compare
coverage there. Reported either way.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from saorl.benchmark_sg.calib_vs_fixed import build_records, per_rule_rows  # noqa: E402

OUT = os.path.join("results/conformal", "benchmark_sg", "matched_size.json")
DELTAS = (0.05, 0.10, 0.15, 0.20)


def sweep(recs, target_size):
    """Best constant threshold at (closest to) a target mean set size."""
    grid = sorted({s for r in recs for s in r.class_scores})
    best = None
    for tau in grid:
        sizes, cov = [], []
        for r in recs:
            keep = [k for k, s in enumerate(r.class_scores) if s >= tau]
            sizes.append(len(keep))
            cov.append(1.0 if r.gold_idx in keep else 0.0)
        mean_size = sum(sizes) / len(sizes)
        d = abs(mean_size - target_size)
        if best is None or d < best["size_gap"]:
            best = dict(tau=float(tau), mean_size=mean_size,
                        coverage=sum(cov) / len(cov), size_gap=d)
    return best


def main() -> None:
    recs = build_records()
    out = []
    hdr = (f"{'family':<11} {'delta':>5} {'conf size':>9} {'conf cov':>8} "
           f"{'tuned tau':>9} {'tuned size':>10} {'tuned cov':>9} {'delta cov':>9}")
    print(hdr); print("-" * len(hdr))
    for family, rs in recs.items():
        for dl in DELTAS:
            rows = per_rule_rows(rs, dl)
            conf_size = sum(r["n_conf"] for r in rows) / len(rows)
            conf_cov = sum(1.0 if r["gold_in_conf"] else 0.0 for r in rows) / len(rows)
            t = sweep(rs, conf_size)
            rec = dict(family=family, delta=dl, conf_mean_size=conf_size,
                       conf_coverage=conf_cov, tuned_tau=t["tau"],
                       tuned_mean_size=t["mean_size"], tuned_coverage=t["coverage"],
                       size_gap=t["size_gap"], coverage_delta=conf_cov - t["coverage"])
            out.append(rec)
            print(f"{family:<11} {dl:>5.2f} {conf_size:>9.2f} {conf_cov:>8.3f} "
                  f"{t['tau']:>9.3f} {t['mean_size']:>10.2f} {t['coverage']:>9.3f} "
                  f"{conf_cov - t['coverage']:>+9.3f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
