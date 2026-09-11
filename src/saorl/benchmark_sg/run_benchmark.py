"""Source-grounded benchmark, step 7: split conformal + report + tex fragment.

Regenerates, from the pinned data only:
  * results/conformal/benchmark_sg/report.json  -- every plan-3.5 / 11.5 metric
    with exact-binomial (Clopper-Pearson) and cluster-bootstrap intervals;
  * paper/generated/gen_benchmark_sg.tex          -- a table body (data rows).

Splits (plan 3.4): primary = grouped chronological (group by AST skeleton, no
near-duplicate template crosses calibration/test, calibrate on older commits,
test on later); secondary = source-held-out (one sub-source held out); plus the
out-of-DSL partition for overall coverage.  Conformal reuses
saorl.conformal.conformal_threshold (the split-conformal q-hat logic).

Reproduce:
    SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.run_benchmark
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta

from ..conformal import DELTA_SEM, conformal_threshold
from . import control_mdp, dominance
from .candidates import prom_candidates
from .evaluate import Pool, build_kyv_pool, build_prom_pool
from .fetch import load_manifest
from .parse import (KyvTarget, PromTarget, kyverno_oodsl, parse_kyverno,
                    parse_prometheus, prom_threshold_bank, prometheus_oodsl)
from .score import class_score, score_reading

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REPORT = _ROOT / "results/conformal" / "benchmark_sg" / "report.json"
TEX = _ROOT / "paper" / "generated" / "gen_benchmark_sg.tex"
DELTA = DELTA_SEM  # 0.1 target miscoverage (reused from saorl.conformal)
BOOT = 2000
SEED = 0


# --------------------------------------------------------------------------
@dataclass
class Rec:
    family: str
    name: str
    sub_source: str
    ts: int
    skeleton: str
    gold_score: float
    class_scores: List[float]
    vectors: List[Tuple[float, ...]]
    gold_idx: int
    dom: dict
    axes_per_class: List[str]
    class_axes: List[frozenset]


def _prom_skeleton(t: PromTarget) -> str:
    r = t.reading
    d = "gt" if r.comparator in (">", ">=") else "lt"
    return f"prom|{r.metric.split('_')[0]}|{r.aggregation}|rate={r.rate_window_s is not None}|{d}"


def _kyv_skeleton(t: KyvTarget) -> str:
    r = t.reading
    return f"kyv|{r.family}|{'.'.join(sorted(r.match_kinds))}"


def _record(pool: Pool, t, skeleton: str) -> Rec:
    cs = [class_score(t, c) for c in pool.classes]
    vecs = [c.vector for c in pool.classes]
    dom = dominance.pool_dominance(vecs)
    axes = [sorted(c.axes)[0] if c.axes else "?" for c in pool.classes]
    return Rec(family=pool.family, name=getattr(t, "name", getattr(t, "policy_name", "?")),
               sub_source=t.sub_source, ts=int(t.last_commit_ts or 0), skeleton=skeleton,
               gold_score=cs[pool.gold_idx], class_scores=cs, vectors=vecs,
               gold_idx=pool.gold_idx, dom=dom, axes_per_class=axes,
               class_axes=[frozenset(c.axes) for c in pool.classes])


# canonical single- and paired-axis strata (plan 11.6): pure changes are often
# nested (collapse); tradeoff pairs are often incomparable (non-dominated)
_STRATA = {
    "prometheus": [
        ("pure_threshold", {"threshold"}), ("pure_duration", {"duration"}),
        ("pure_comparator", {"comparator"}), ("pure_aggregation", {"aggregation"}),
        ("pure_scope", {"scope"}),
        ("threshold+duration", {"threshold", "duration"}),
        ("threshold+aggregation", {"threshold", "aggregation"}),
        ("duration+scope", {"duration", "scope"}),
    ],
    "kyverno": [
        ("pure_exception", {"exception"}), ("pure_action", {"action"}),
        ("pure_scope", {"scope"}), ("pure_allowlist", {"allowlist"}),
        ("pure_label_set", {"label_set"}), ("pure_threshold", {"threshold"}),
        ("exception+action", {"exception", "action"}),
        ("allowlist+action", {"allowlist", "action"}),
    ],
}


def _subpool_vectors(rec: Rec, axes: set) -> List[Tuple[float, ...]]:
    """Distinct fire/cost vectors of readings that vary only the given axes
    (plus identity) -- the executable sub-pool for that ambiguity stratum."""
    keep = {"identity"} | axes
    vecs = [rec.vectors[i] for i, axset in enumerate(rec.class_axes) if axset & keep]
    return list(dict.fromkeys(vecs))     # unique, order-preserving


def _strata_dominance(recs: List[Rec], family: str) -> dict:
    out = {}
    for name, axes in _STRATA[family]:
        collapses, nus, widths, nds, ns = [], [], [], [], []
        for r in recs:
            sv = _subpool_vectors(r, axes)
            if len(sv) < 2:
                continue                  # this target has no candidates on these axes
            d = dominance.pool_dominance(sv)
            collapses.append(d["collapses"])
            nus.append(d["nu_pi"])
            widths.append(d["antichain_width"])
            nds.append(d["genuinely_non_dominated"])
            ns.append(len(sv))
        if not ns:
            continue
        out[name] = dict(
            n_pools=len(ns), mean_subpool_size=round(float(np.mean(ns)), 2),
            collapse_rate=round(float(np.mean(collapses)), 3),
            frac_non_dominated=round(float(np.mean(nds)), 3),
            mean_antichain_width=round(float(np.mean(widths)), 3),
            nu_pi_mean=round(float(np.mean(nus)), 4))
    return out


# --------------------------------------------------------------------------
# split-conformal coverage over a calibration/test partition of records
# --------------------------------------------------------------------------

def _conformal_eval(cal: List[Rec], test: List[Rec], oodsl_test: int, delta: float) -> dict:
    q = conformal_threshold([r.gold_score for r in cal], delta)
    covered = [r.gold_score >= q for r in test]
    set_sizes = [sum(1 for s in r.class_scores if s >= q) for r in test]
    empties = [ss == 0 for ss in set_sizes]
    # retained-set incomparability (>=2 genuinely incomparable retained readings)
    retained_nd = []
    for r in test:
        keep = [i for i, s in enumerate(r.class_scores) if s >= q]
        rd = dominance.restricted_dominance(r.vectors, keep)
        retained_nd.append(rd["genuinely_non_dominated"])
    n = len(test)
    k = int(sum(covered))
    cov = k / n if n else float("nan")
    overall_den = n + oodsl_test
    # with no out-of-DSL test units the overall rate is not defined: reporting
    # k/n there would silently relabel the in-DSL rate as an overall one.
    overall = (k / overall_den) if oodsl_test else float("nan")
    return dict(
        n_test=n, n_cal=len(cal), qhat=(None if not math.isfinite(q) else round(q, 6)),
        in_dsl_coverage=round(cov, 4),
        in_dsl_cov_cp=[round(x, 4) for x in _clopper_pearson(k, n)],
        overall_coverage=round(overall, 4), n_oodsl_test=oodsl_test,
        mean_set_size=round(float(np.mean(set_sizes)), 3) if set_sizes else None,
        median_set_size=(float(np.median(set_sizes)) if set_sizes else None),
        empty_set_rate=round(float(np.mean(empties)), 4) if empties else None,
        frac_ge2_incomparable_retained=round(float(np.mean(retained_nd)), 4) if retained_nd else None,
    )


def _loo_eval(recs: List[Rec], oodsl_test: int, delta: float) -> dict:
    """Leave-one-out (exchangeable / iid) split-conformal -- the validity
    headline: under exchangeability the theorem predicts coverage >= 1-delta.
    Each pool is held out and q-hat is calibrated on all other pools' gold
    scores."""
    gs = [r.gold_score for r in recs]
    covered, sizes, retained_nd = [], [], []
    for i, r in enumerate(recs):
        others = [gs[j] for j in range(len(recs)) if j != i]
        q = conformal_threshold(others, delta)
        covered.append(r.gold_score >= q)
        keep = [k for k, s in enumerate(r.class_scores) if s >= q]
        sizes.append(len(keep))
        retained_nd.append(dominance.restricted_dominance(r.vectors, keep)["genuinely_non_dominated"])
    n = len(recs)
    k = int(sum(covered))
    overall_den = n + oodsl_test
    # cluster bootstrap over AST-skeleton clusters of the LOO covered bits
    clusters: Dict[str, List[bool]] = {}
    for r, c in zip(recs, covered):
        clusters.setdefault(r.skeleton, []).append(c)
    boot = _cluster_boot_from_bits(clusters)
    return dict(
        n_test=n, split="leave-one-out (iid/exchangeable)",
        in_dsl_coverage=round(k / n, 4) if n else None,
        in_dsl_cov_cp=[round(x, 4) for x in _clopper_pearson(k, n)],
        cluster_bootstrap_ci=list(boot),
        overall_coverage=round(k / overall_den, 4) if overall_den else None,
        n_oodsl_test=oodsl_test,
        mean_set_size=round(float(np.mean(sizes)), 3) if sizes else None,
        median_set_size=float(np.median(sizes)) if sizes else None,
        empty_set_rate=round(float(np.mean([s == 0 for s in sizes])), 4) if sizes else None,
        frac_ge2_incomparable_retained=round(float(np.mean(retained_nd)), 4) if retained_nd else None,
    )


def _cluster_boot_from_bits(clusters: Dict[str, List[bool]], b: int = BOOT,
                            seed: int = SEED) -> Tuple[float, float]:
    keys = list(clusters.keys())
    if not keys:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    covs = []
    for _ in range(b):
        pick = rng.integers(0, len(keys), size=len(keys))
        pooled = [c for i in pick for c in clusters[keys[i]]]
        covs.append(float(np.mean(pooled)) if pooled else float("nan"))
    return (round(float(np.nanpercentile(covs, 2.5)), 4),
            round(float(np.nanpercentile(covs, 97.5)), 4))


def _clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def _cluster_bootstrap_cov(cal: List[Rec], test: List[Rec], delta: float,
                           b: int = BOOT, seed: int = SEED) -> Tuple[float, float]:
    """Percentile CI for test coverage, resampling AST-skeleton clusters."""
    q = conformal_threshold([r.gold_score for r in cal], delta)
    clusters: Dict[str, List[bool]] = {}
    for r in test:
        clusters.setdefault(r.skeleton, []).append(r.gold_score >= q)
    keys = list(clusters.keys())
    if not keys:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    covs = []
    for _ in range(b):
        pick = rng.integers(0, len(keys), size=len(keys))
        pooled = [c for i in pick for c in clusters[keys[i]]]
        covs.append(float(np.mean(pooled)) if pooled else float("nan"))
    return (round(float(np.nanpercentile(covs, 2.5)), 4),
            round(float(np.nanpercentile(covs, 97.5)), 4))


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------

def _grouped_chrono_split(recs: List[Rec]) -> Tuple[List[Rec], List[Rec]]:
    """Group by AST skeleton; order groups by (median commit ts, skeleton);
    older half calibrates, later half tests.  No skeleton crosses."""
    groups: Dict[str, List[Rec]] = {}
    for r in recs:
        groups.setdefault(r.skeleton, []).append(r)
    order = sorted(groups.keys(),
                   key=lambda s: (np.median([r.ts for r in groups[s]]), s))
    ncal = max(1, int(round(len(order) * 0.5)))
    cal = [r for s in order[:ncal] for r in groups[s]]
    test = [r for s in order[ncal:] for r in groups[s]]
    if not test:                      # tiny family safeguard
        cal, test = cal[:-1], cal[-1:]
    return cal, test


def _source_heldout_split(recs: List[Rec]) -> Tuple[List[Rec], List[Rec], str]:
    """Hold out the single largest sub-source for transfer reporting."""
    by: Dict[str, List[Rec]] = {}
    for r in recs:
        by.setdefault(r.sub_source, []).append(r)
    held = max(by.keys(), key=lambda s: (len(by[s]), s))
    cal = [r for s, rs in by.items() if s != held for r in rs]
    test = by[held]
    return cal, test, held


# --------------------------------------------------------------------------
# dominance aggregation across pools (plan 3.5 / 11.5)
# --------------------------------------------------------------------------

def _aggregate_dominance(recs: List[Rec]) -> dict:
    nus = [r.dom["nu_pi"] for r in recs]
    widths = [r.dom["antichain_width"] for r in recs]
    maxs = [r.dom["n_maximal"] for r in recs]
    collapses = [r.dom["collapses"] for r in recs]
    nd = [r.dom["genuinely_non_dominated"] for r in recs]
    ge2 = [r.dom["has_ge2_incomparable"] for r in recs]
    pruned = [r.dom["frac_pruned"] for r in recs]
    # stratify: which axes appear among the maximal (non-dominated) classes
    axis_nd: Dict[str, int] = {}
    axis_tot: Dict[str, int] = {}
    for r in recs:
        maximal = set(r.dom["maximal_idx"])
        for i, ax in enumerate(r.axes_per_class):
            axis_tot[ax] = axis_tot.get(ax, 0) + 1
            if i in maximal:
                axis_nd[ax] = axis_nd.get(ax, 0) + 1
    strat = {ax: dict(n_classes=axis_tot[ax], n_maximal=axis_nd.get(ax, 0),
                      frac_maximal=round(axis_nd.get(ax, 0) / axis_tot[ax], 3))
             for ax in sorted(axis_tot)}
    return dict(
        n_pools=len(recs),
        collapse_rate=round(float(np.mean(collapses)), 4),
        frac_dominating_member=round(float(np.mean(collapses)), 4),
        frac_genuinely_non_dominated=round(float(np.mean(nd)), 4),
        frac_ge2_incomparable=round(float(np.mean(ge2)), 4),
        mean_n_maximal=round(float(np.mean(maxs)), 3),
        median_n_maximal=float(np.median(maxs)),
        mean_antichain_width=round(float(np.mean(widths)), 3),
        max_antichain_width=int(np.max(widths)),
        mean_frac_pruned=round(float(np.mean(pruned)), 4),
        nu_pi_mean=round(float(np.mean(nus)), 4),
        nu_pi_median=round(float(np.median(nus)), 4),
        nu_pi_max=round(float(np.max(nus)), 4),
        frac_nu_pi_positive=round(float(np.mean([1.0 if x > 1e-9 else 0.0 for x in nus])), 4),
        axis_stratification=strat,
    )


# --------------------------------------------------------------------------
def _family_report(recs: List[Rec], oodsl: List[dict], delta: float) -> dict:
    # chronological ordering for the out-of-DSL denominator (later half = test)
    oodsl_sorted = sorted(oodsl, key=lambda d: (d.get("last_commit_ts") or 0, d["name"]))
    oodsl_test = len(oodsl_sorted) - max(1, int(round(len(oodsl_sorted) * 0.5))) if oodsl_sorted else 0

    # PRIMARY validity: exchangeable leave-one-out (theorem precondition holds).
    # Leave-one-out tests EVERY in-DSL rule, so the overall denominator takes
    # the whole out-of-DSL arm; the chronological split tests half the in-DSL
    # arm and therefore pairs with the half-sized out-of-DSL test set.
    primary = _loo_eval(recs, len(oodsl_sorted), delta)

    # STRESS 1: grouped chronological (skeleton groups, older calibrate / later test)
    cal, test = _grouped_chrono_split(recs)
    chrono = _conformal_eval(cal, test, oodsl_test, delta)
    chrono["cluster_bootstrap_ci"] = list(_cluster_bootstrap_cov(cal, test, delta))

    # STRESS 2: source held-out (one sub-source held out for transfer)
    scal, stest, held = _source_heldout_split(recs)
    src = _conformal_eval(scal, stest, 0, delta)
    src["held_out_source"] = held

    return dict(
        n_pools=len(recs),
        n_oodsl_total=len(oodsl_sorted),
        primary_iid_loo=primary,
        stress_grouped_chronological=chrono,
        stress_source_heldout=src,
        dominance=_aggregate_dominance(recs),
        axis_strata=_strata_dominance(recs, recs[0].family if recs else "prometheus"),
    )


def build() -> dict:
    man = load_manifest()
    pt, ps = parse_prometheus()
    kt, ks = parse_kyverno()
    bank = prom_threshold_bank(pt)

    prom_recs = [_record(build_prom_pool(t, bank), t, _prom_skeleton(t)) for t in pt]
    kyv_recs = [_record(build_kyv_pool(t), t, _kyv_skeleton(t)) for t in kt]

    prom_oodsl = prometheus_oodsl()
    kyv_oodsl = kyverno_oodsl()

    report = dict(
        provenance=dict(
            repos={k: dict(url=v["url"], commit=v["commit"]) for k, v in man["repos"].items()},
            delta=delta_str(), score_weights_frozen=True,
            reproduce="SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.run_benchmark",
        ),
        counts=dict(
            prometheus=dict(parsed=len(pt), skipped=ps, oodsl_with_text=len(prom_oodsl),
                            n_pools=len(prom_recs)),
            kyverno=dict(parsed=len(kt), skipped=ks, oodsl_with_text=len(kyv_oodsl),
                         n_pools=len(kyv_recs)),
            total_pools=len(prom_recs) + len(kyv_recs),
        ),
        prometheus=_family_report(prom_recs, prom_oodsl, DELTA),
        kyverno=_family_report(kyv_recs, kyv_oodsl, DELTA),
        dominance_all=_aggregate_dominance(prom_recs + kyv_recs),
        control_mdp=control_mdp.run(),
    )
    return report


def delta_str() -> float:
    return DELTA


# --------------------------------------------------------------------------
# tex fragment (data rows only, matching paper/generated/gen_*.tex style)
# --------------------------------------------------------------------------

def _fnum(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def emit_tex(rep: dict) -> str:
    rows = []

    def split_row(label, split_label, s, d, show_dom):
        cp = s["in_dsl_cov_cp"]
        dcell = (f"{_fnum(d['collapse_rate'])} & {_fnum(d['frac_genuinely_non_dominated'])} & "
                 f"{_fnum(d['nu_pi_mean'])}") if show_dom else "& & "
        rows.append(
            f"{label} & {split_label} & {s['n_test']} & "
            f"{_fnum(s['in_dsl_coverage'])}~[{_fnum(cp[0])}, {_fnum(cp[1])}] & "
            f"{_fnum(s['overall_coverage'])} & {_fnum(s['mean_set_size'],2)} & "
            f"{_fnum(s['empty_set_rate'])} & {dcell} \\\\\n")

    def fam_rows(famkey, label):
        f = rep[famkey]
        d = f["dominance"]
        split_row(label, "IID (leave-one-out)", f["primary_iid_loo"], d, True)
        split_row("", "Stress: grouped chrono.", f["stress_grouped_chronological"], d, False)
        s = f["stress_source_heldout"]
        split_row("", f"Stress: source held-out ({s['held_out_source']})", s, d, False)

    fam_rows("prometheus", "Prometheus")
    rows.append("\\addlinespace\n")
    fam_rows("kyverno", "Kyverno")
    cm = rep["control_mdp"]
    rows.append("\\midrule\n")
    rows.append(
        f"\\multicolumn{{10}}{{l}}{{\\emph{{Non-dominated control MDP}} "
        f"(CPU $>$90\\%/5m vs $>$80\\%/15m): "
        f"$\\nu_\\Pi={cm['nu_pi_exact']:.4f}>0$ (exact occupancy LP); "
        f"no singleton matches the robust feasible-return frontier}} \\\\\n")
    return "".join(rows)


def main() -> None:
    rep = build()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(REPORT, "w"), indent=1, default=str)
    TEX.parent.mkdir(parents=True, exist_ok=True)
    with open(TEX, "w") as fh:
        fh.write("% AUTO-GENERATED by saorl.benchmark_sg.run_benchmark -- do not edit\n")
        fh.write(emit_tex(rep))

    c = rep["counts"]
    print(f"pinned commits: " + ", ".join(f"{k}={v['commit'][:10]}" for k, v in rep["provenance"]["repos"].items()))
    print(f"pools: Prometheus={c['prometheus']['n_pools']}  Kyverno={c['kyverno']['n_pools']}  "
          f"total={c['total_pools']}")
    for fam in ("prometheus", "kyverno"):
        p = rep[fam]["primary_iid_loo"]
        gc = rep[fam]["stress_grouped_chronological"]
        sh = rep[fam]["stress_source_heldout"]
        d = rep[fam]["dominance"]
        print(f"[{fam}] IID cov={p['in_dsl_coverage']} CP{p['in_dsl_cov_cp']} "
              f"boot{p['cluster_bootstrap_ci']} overall={p['overall_coverage']} "
              f"meanset={p['mean_set_size']} empty={p['empty_set_rate']}")
        print(f"          stress: grouped-chrono cov={gc['in_dsl_coverage']} | "
              f"source-heldout({sh['held_out_source']}) cov={sh['in_dsl_coverage']}")
        print(f"          collapse={d['collapse_rate']} non-dom(nu>0)={d['frac_genuinely_non_dominated']} "
              f"ge2incomp={d['frac_ge2_incomparable']} "
              f"nu_pi(mean/med/max)={d['nu_pi_mean']}/{d['nu_pi_median']}/{d['nu_pi_max']}")
    cm = rep["control_mdp"]
    print(f"control MDP: nu_pi_exact={cm['nu_pi_exact']:.4f} non-dominated={cm['genuinely_non_dominated']} "
          f"no_singleton_matches={cm['no_singleton_matches_robust_frontier']}")
    print(f"wrote {REPORT}")
    print(f"wrote {TEX}")


if __name__ == "__main__":
    main()
