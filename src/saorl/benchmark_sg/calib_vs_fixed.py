"""Does split-conformal calibration ever differ from a hand-set tau=0.5 gate?

A reviewer objected that on the tiny n=11 semantic corpus the conformal
threshold pins to q-hat=0.500 for every practical delta, making the namesake
mechanism indistinguishable from a fixed tau=0.5 plausibility gate.  This
script answers the same question on the *source-grounded* benchmark (88
Prometheus alert rules, 44 Kyverno policies), where n is an order of magnitude
larger and the score is the frozen deterministic s(l, psi) of `score.py`.

For each family and each delta in {0.05, 0.10, 0.15, 0.20} it computes, per
held-out rule, the leave-one-out conformal threshold q-hat (reusing
`saorl.conformal.conformal_threshold` -- the quantile logic is NOT
reimplemented here) and compares two retained sets over the same merged
candidate pool:

    U_conf = {psi : s(l, psi) >= q-hat}        (calibrated)
    U_fix  = {psi : s(l, psi) >= 0.5}          (hand-set gate)

The inclusion direction (>=, ties retained) is copied verbatim from
`run_benchmark._loo_eval`.  Pools, scores and gold indices come from
`run_benchmark._record`, so this is the real pipeline, not a re-derivation.

Run:  PYTHONPATH=. python3 -m saorl.benchmark_sg.calib_vs_fixed
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..conformal import conformal_threshold
from .parse import parse_kyverno, parse_prometheus, prom_threshold_bank
from .evaluate import build_kyv_pool, build_prom_pool
from .run_benchmark import Rec, _kyv_skeleton, _prom_skeleton, _record

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT = _ROOT / "results/conformal" / "benchmark_sg" / "calib_vs_fixed.json"

DELTAS = (0.05, 0.10, 0.15, 0.20)
TAU_FIXED = 0.5          # the hand-set gate the reviewer says conformal reduces to


# --------------------------------------------------------------------------
def build_records() -> Dict[str, List[Rec]]:
    """The exact records `run_benchmark.build()` scores -- same parse, same
    pools, same frozen score."""
    pt, _ = parse_prometheus()
    kt, _ = parse_kyverno()
    bank = prom_threshold_bank(pt)
    return {
        "prometheus": [_record(build_prom_pool(t, bank), t, _prom_skeleton(t)) for t in pt],
        "kyverno": [_record(build_kyv_pool(t), t, _kyv_skeleton(t)) for t in kt],
    }


def per_rule_rows(recs: Sequence[Rec], delta: float) -> List[dict]:
    """Leave-one-out q-hat per rule, then the two retained sets."""
    gs = [r.gold_score for r in recs]
    rows = []
    for i, r in enumerate(recs):
        others = [gs[j] for j in range(len(recs)) if j != i]
        q = conformal_threshold(others, delta)                 # reused, not reimplemented
        # inclusion direction copied from run_benchmark._loo_eval:
        #     keep = [k for k, s in enumerate(r.class_scores) if s >= q]
        u_conf = frozenset(k for k, s in enumerate(r.class_scores) if s >= q)
        u_fix = frozenset(k for k, s in enumerate(r.class_scores) if s >= TAU_FIXED)
        rows.append(dict(
            name=r.name,
            family=r.family,
            sub_source=r.sub_source,
            n_classes=len(r.class_scores),
            qhat=(None if not math.isfinite(q) else round(float(q), 6)),
            gold_score=round(float(r.gold_score), 6),
            n_conf=len(u_conf),
            n_fix=len(u_fix),
            differ=bool(u_conf != u_fix),
            only_conf=sorted(u_conf - u_fix),
            only_fix=sorted(u_fix - u_conf),
            gold_in_conf=bool(r.gold_score >= q),
            gold_in_fix=bool(r.gold_score >= TAU_FIXED),
        ))
    return rows


def aggregate(rows: List[dict], family: str, delta: float) -> dict:
    n = len(rows)
    qs = [r["qhat"] for r in rows]
    distinct = sorted({q for q in qs if q is not None})
    n_q_not_half = sum(1 for q in qs if q is None or abs(q - TAU_FIXED) > 1e-12)
    differ = [r["differ"] for r in rows]
    saves = [r for r in rows if r["gold_in_conf"] and not r["gold_in_fix"]]
    losses = [r for r in rows if r["gold_in_fix"] and not r["gold_in_conf"]]
    return dict(
        family=family, delta=delta, n_rules=n,
        n_distinct_qhat=len(distinct),
        distinct_qhat=distinct if len(distinct) <= 12 else
                      [distinct[0], distinct[len(distinct) // 2], distinct[-1]],
        qhat_min=min(distinct) if distinct else None,
        qhat_max=max(distinct) if distinct else None,
        n_qhat_ne_half=n_q_not_half,
        frac_qhat_ne_half=round(n_q_not_half / n, 4) if n else None,
        n_sets_differ=int(sum(differ)),
        pct_sets_differ=round(100.0 * float(np.mean(differ)), 2) if n else None,
        saves=len(saves), save_rules=[r["name"] for r in saves][:20],
        losses=len(losses), loss_rules=[r["name"] for r in losses][:20],
        mean_n_conf=round(float(np.mean([r["n_conf"] for r in rows])), 3),
        mean_n_fix=round(float(np.mean([r["n_fix"] for r in rows])), 3),
        mean_pool_size=round(float(np.mean([r["n_classes"] for r in rows])), 3),
        empty_rate_conf=round(float(np.mean([r["n_conf"] == 0 for r in rows])), 4),
        empty_rate_fix=round(float(np.mean([r["n_fix"] == 0 for r in rows])), 4),
        coverage_conf=round(float(np.mean([r["gold_in_conf"] for r in rows])), 4),
        coverage_fix=round(float(np.mean([r["gold_in_fix"] for r in rows])), 4),
        target_coverage=round(1 - delta, 4),
    )


# --------------------------------------------------------------------------
def _fmt_q(a: dict) -> str:
    if a["n_distinct_qhat"] == 1:
        return f"{a['qhat_min']:.4f} (1)"
    return f"{a['qhat_min']:.4f}-{a['qhat_max']:.4f} ({a['n_distinct_qhat']})"


def print_table(aggs: List[dict]) -> None:
    hdr = (f"{'family':<11}{'delta':>6}{'n':>5}{'qhat range (distinct)':>26}"
           f"{'q!=.5':>7}{'%U differ':>11}{'saves':>7}{'loss':>6}"
           f"{'|Uconf|':>9}{'|Ufix|':>8}{'cov_c':>7}{'cov_f':>7}")
    print(hdr)
    print("-" * len(hdr))
    for a in aggs:
        print(f"{a['family']:<11}{a['delta']:>6.2f}{a['n_rules']:>5}{_fmt_q(a):>26}"
              f"{a['n_qhat_ne_half']:>7}{a['pct_sets_differ']:>10.1f}%"
              f"{a['saves']:>7}{a['losses']:>6}"
              f"{a['mean_n_conf']:>9.2f}{a['mean_n_fix']:>8.2f}"
              f"{a['coverage_conf']:>7.3f}{a['coverage_fix']:>7.3f}")


def main() -> None:
    recs = build_records()
    aggs: List[dict] = []
    detail: Dict[str, Dict[str, List[dict]]] = {}
    for family in ("prometheus", "kyverno"):
        detail[family] = {}
        for delta in DELTAS:
            rows = per_rule_rows(recs[family], delta)
            detail[family][f"delta_{delta:.2f}"] = rows
            aggs.append(aggregate(rows, family, delta))

    print_table(aggs)
    print()
    for family in ("prometheus", "kyverno"):
        rows = detail[family][f"delta_{0.10:.2f}"]
        n_fix_empty = sum(1 for r in rows if r["n_fix"] == 0)
        print(f"[{family}] at the fixed tau=0.5 gate: {n_fix_empty}/{len(rows)} rules retain "
              f"NOTHING; mean |U_fix|={np.mean([r['n_fix'] for r in rows]):.2f} "
              f"vs mean pool {np.mean([r['n_classes'] for r in rows]):.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(tau_fixed=TAU_FIXED, deltas=list(DELTAS),
                   aggregates=aggs, per_rule=detail),
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
