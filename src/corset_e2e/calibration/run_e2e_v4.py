"""WP-C/D/E: fit on dev, calibrate on cal, execute the frozen test once.

Stages, in order, with the hidden store opened only where marked:
  1. build the public metric catalog          [dev hidden store only]
  2. license slots for every unit             [visible only]
  3. fit score weights                        [dev hidden store only]
  4. mechanical eligibility of each target    [hidden store, AFTER generation]
  5. calibrate qhat and the OOD threshold     [cal hidden store]
  6. one frozen test pass                     [test hidden store, once]

Run: PYTHONPATH=. python3 corset_e2e/calibration/run_e2e.py [--stage all]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.dsl.schema import (  # noqa: E402
    IN_DSL, PARSER_FAILURE, classify_target, manifest as dsl_manifest,
    target_reading)
from corset_e2e.generator.generate import (  # noqa: E402
    Catalog, extract_slots, in_pool, pool_size)
from corset_e2e.source_grounded.build_catalog import build_catalog  # noqa: E402
from corset_e2e.source_grounded.build_catalog_v2 import RepoCatalogs  # noqa: E402
from corset_e2e.source_grounded.build_catalog_v3 import RepoCatalogsV3  # noqa: E402
from corset_e2e.scoring.deterministic import DEFAULT_WEIGHTS, Scorer  # noqa: E402

OUT = os.path.join(ROOT, "results/e2e")
VIS = os.path.join(OUT, "visible_store")
HID = os.path.join(OUT, "hidden_store")
CATALOG = os.path.join(OUT, "metric_catalog.json")
CATALOG_V2 = os.path.join(OUT, "metric_catalog_v2.json")
CATALOG_V3 = os.path.join(OUT, "metric_catalog_v3.json")
CATALOG_VERSION = os.environ.get("CORSET_CATALOG", "v1")
DELTA_SEM = 0.10
OOD_TARGET_RECALL = 0.80


def load_split(split: str) -> List[dict]:
    led = json.load(open(os.path.join(OUT, "cluster_ledger.json")))
    return [e for e in led if e["split"] == split]


def vis(rid: str) -> dict:
    return json.load(open(os.path.join(VIS, f"{rid}.json")))


def hid(rid: str) -> dict:
    return json.load(open(os.path.join(HID, f"{rid}.target.json")))


def unit_view(entry: dict, catalog, weights, mode: str = "complete") -> dict:
    """Everything computable from the VISIBLE side alone.

    `catalog` is either a global Catalog (v1) or a RepoCatalogs server (v2),
    in which case the unit gets its deployment's own metric namespace with its
    own source-file cluster removed.
    """
    v = vis(entry["rule_id"])
    cat = catalog
    if isinstance(catalog, (RepoCatalogs, RepoCatalogsV3)):
        cat = Catalog(catalog.metrics_for(entry["cluster_id"]))
    slots = extract_slots(v, cat, mode=mode)
    sc = Scorer(v, slots, weights)
    return dict(rid=entry["rule_id"], cluster=entry["cluster_id"],
                repo=entry["repo"], visible=v, slots=slots, scorer=sc)


def gold_info(u: dict) -> Tuple[str, float, int]:
    """Eligibility label, gold score, and #candidates scoring at least gold.

    Label is the grammar-reach class. For IN_DSL units the generator either
    put the target in its pool (rho_gen success) or missed it (GEN_MISS);
    both stay in the IN_DSL denominator.
    """
    h = hid(u["rid"])
    label = classify_target(h)
    if label != IN_DSL:
        return label, float("-inf"), -1
    g = target_reading(h)
    if not in_pool(u["slots"], g):
        return "GEN_MISS", float("-inf"), -1
    s = u["scorer"].score(g)
    n_above, _ = u["scorer"].retained(s)
    return IN_DSL, s, n_above


def cluster_bootstrap(by_cluster: Dict[str, List[int]], n_boot: int = 2000,
                      seed: int = 20260802) -> Tuple[float, float, float]:
    keys = sorted(by_cluster)
    if not keys:
        return float("nan"), float("nan"), float("nan")
    flat = [x for k in keys for x in by_cluster[k]]
    point = sum(flat) / len(flat)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        pick = [by_cluster[keys[rng.randrange(len(keys))]] for _ in keys]
        vals = [x for p in pick for x in p]
        if vals:
            stats.append(sum(vals) / len(vals))
    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[min(len(stats) - 1, int(0.975 * len(stats)))]
    return point, lo, hi


# ---------------------------------------------------------------- stage 3
WEIGHT_GRID = dict(var=[0.30, 0.40, 0.50], num=[0.10, 0.18, 0.26],
                   op=[0.08, 0.14, 0.20], win=[0.06, 0.12, 0.18],
                   forx=[0.04, 0.08, 0.14], agg=[0.04, 0.08, 0.14])


def fit_weights(dev: Sequence[dict], catalog: Catalog, sample: int = 220) -> dict:
    """Coordinate search minimising mean log(1 + #candidates ranked >= gold)."""
    rng = random.Random(7)
    pool = list(dev)
    rng.shuffle(pool)
    pool = pool[:sample]

    def objective(w) -> float:
        tot, n = 0.0, 0
        for e in pool:
            u = unit_view(e, catalog, w)
            label, s, n_above = gold_info(u)
            if label != IN_DSL:
                continue
            tot += math.log1p(n_above)
            n += 1
        return tot / max(n, 1)

    w = dict(DEFAULT_WEIGHTS)
    best = objective(w)
    for _ in range(2):
        for k, vals in WEIGHT_GRID.items():
            for v in vals:
                if v == w[k]:
                    continue
                cand = dict(w)
                cand[k] = v
                o = objective(cand)
                if o < best - 1e-9:
                    best, w = o, cand
    return dict(weights=w, dev_objective=best, n_dev_units=len(pool))


# ---------------------------------------------------------------- stage 5/6
def evaluate(entries: Sequence[dict], catalog: Catalog, weights, qhat=None,
             qood=None, tag: str = "", mode: str = "complete") -> dict:
    rows = []
    for e in entries:
        u = unit_view(e, catalog, weights, mode=mode)
        label, s, n_above = gold_info(u)
        # bottom-candidate signal: strength of the best lexical evidence for
        # any licensed metric. No evidence -> the rule is unsupported.
        best_var = max((sc for _k, sc in u["scorer"].tables[0]), default=0.0)
        ood_signal = 1.0 - (best_var / max(weights["var"], 1e-9))
        row = dict(rid=u["rid"], cluster=u["cluster"], repo=u["repo"],
                   label=label, gold_score=s if s != float("-inf") else None,
                   n_above=n_above, pool=pool_size(u["slots"]),
                   ood_signal=round(ood_signal, 6),
                   n_metrics=len(u["slots"]["metrics"]))
        if qhat is not None and label == IN_DSL:
            row["retained_gold"] = bool(s >= qhat)
            cnt, _ = u["scorer"].retained(qhat)
            row["set_size"] = cnt
            row["maximal_size"] = u["scorer"].maximal_count(qhat)
        if qood is not None:
            row["abstain"] = bool(ood_signal >= qood) or (
                row.get("set_size", 1) == 0)
        rows.append(row)
    return dict(tag=tag, n=len(rows), rows=rows)


def summarise(rows: Sequence[dict], qhat: float, qood: float) -> dict:
    ind = [r for r in rows if r["label"] == IN_DSL]
    reach = [r for r in rows if r["label"] in (IN_DSL, "GEN_MISS")]
    ood = [r for r in rows if r["label"] not in (IN_DSL, "GEN_MISS")]

    def by_cluster(sel, pred):
        d = defaultdict(list)
        for r in sel:
            d[r["cluster"]].append(1 if pred(r) else 0)
        return d

    gen_pt, gen_lo, gen_hi = cluster_bootstrap(
        by_cluster(reach, lambda r: r["label"] == IN_DSL))
    ret_pt, ret_lo, ret_hi = cluster_bootstrap(
        by_cluster(ind, lambda r: r.get("retained_gold", False)))
    e2e_pt, e2e_lo, e2e_hi = cluster_bootstrap(
        by_cluster(rows, lambda r: r["label"] == IN_DSL and r.get("retained_gold", False)))
    ood_pt, ood_lo, ood_hi = cluster_bootstrap(
        by_cluster(ood, lambda r: r.get("abstain", False)))
    fa_pt, fa_lo, fa_hi = cluster_bootstrap(
        by_cluster(ind, lambda r: r.get("abstain", False)))
    sizes = sorted(r["set_size"] for r in ind if "set_size" in r)
    maxes = sorted(r["maximal_size"] for r in ind if "maximal_size" in r)
    pools = sorted(r["pool"] for r in rows)
    labels = defaultdict(int)
    for r in rows:
        labels[r["label"]] += 1
    return dict(
        n_units=len(rows), n_in_dsl=len(ind), n_grammar_reachable=len(reach),
        n_ood=len(ood),
        n_clusters=len({r["cluster"] for r in rows}),
        eligibility=dict(labels),
        in_dsl_fraction=round(len(ind) / max(len(rows), 1), 4),
        rho_gen=dict(point=round(gen_pt, 4), lo=round(gen_lo, 4), hi=round(gen_hi, 4)),
        rho_ret_given_gen=dict(point=round(ret_pt, 4), lo=round(ret_lo, 4),
                               hi=round(ret_hi, 4)),
        rho_e2e=dict(point=round(e2e_pt, 4), lo=round(e2e_lo, 4), hi=round(e2e_hi, 4)),
        ood_recall=dict(point=round(ood_pt, 4), lo=round(ood_lo, 4), hi=round(ood_hi, 4)),
        false_abstention=dict(point=round(fa_pt, 4), lo=round(fa_lo, 4),
                              hi=round(fa_hi, 4)),
        set_size=dict(mean=round(sum(sizes) / max(len(sizes), 1), 2),
                      median=sizes[len(sizes) // 2] if sizes else None,
                      p90=sizes[int(0.9 * len(sizes))] if sizes else None),
        maximal_set_size=dict(
            mean=round(sum(maxes) / max(len(maxes), 1), 2),
            median=maxes[len(maxes) // 2] if maxes else None,
            p90=maxes[int(0.9 * len(maxes))] if maxes else None),
        pool_size=dict(mean=round(sum(pools) / max(len(pools), 1), 1),
                       median=pools[len(pools) // 2] if pools else None,
                       p90=pools[int(0.9 * len(pools))] if pools else None),
        qhat=qhat, qood=qood)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-split", type=int, default=0)
    args = ap.parse_args()

    dev, cal, test = load_split("dev"), load_split("cal"), load_split("test")
    if args.max_per_split:
        dev, cal, test = (x[:args.max_per_split] for x in (dev, cal, test))

    print(f"[1] catalog from dev ({len(dev)} units)")
    if CATALOG_VERSION.startswith("v4") or CATALOG_VERSION == "v5":
        from corset_e2e.source_grounded.build_catalog_v4 import RepoCatalogsV4
        catalog = RepoCatalogsV4(union_with_v3=(CATALOG_VERSION == "v4u"))
        print(f"    {CATALOG_VERSION}: file-namespace catalog, sizes={catalog.sizes}")
    elif CATALOG_VERSION == "v3":
        catalog = RepoCatalogsV3(CATALOG_V3)
        print(f"    v3: full deployment namespaces, sizes={catalog.sizes}")
    elif CATALOG_VERSION == "v2":
        catalog = RepoCatalogs(CATALOG_V2)
        print(f"    v2: per-deployment namespaces, sizes={catalog.sizes}")
    else:
        cat_payload = build_catalog([e["rule_id"] for e in dev], HID, CATALOG)
        catalog = Catalog(cat_payload["metrics"])
        print(f"    v1: {cat_payload['n_metrics']} public metric names")

    print("[3] fitting score weights on dev")
    fit = fit_weights(dev, catalog)
    weights = fit["weights"]
    print("    weights:", weights, "obj:", round(fit["dev_objective"], 4))

    print(f"[5] calibrating on cal ({len(cal)} units)")
    cal_ev = evaluate(cal, catalog, weights, tag="cal")
    gold = sorted(r["gold_score"] for r in cal_ev["rows"]
                  if r["label"] == IN_DSL and r["gold_score"] is not None)
    n = len(gold)
    if n < 20:
        raise SystemExit(f"only {n} in-DSL calibration units; refusing to "
                         "calibrate a threshold on that (need >= 20)")
    k = max(1, math.floor(DELTA_SEM * (n + 1)))
    qhat = gold[k - 1]
    ood_sig = sorted(r["ood_signal"] for r in cal_ev["rows"] if r["label"] != IN_DSL)
    qood = ood_sig[max(0, int((1 - OOD_TARGET_RECALL) * len(ood_sig)))] if ood_sig else 1.0
    print(f"    n_cal_in_dsl={n} k={k} qhat={qhat:.6f} qood={qood:.6f}")

    print(f"[6] frozen test pass ({len(test)} units)")
    modes = {}
    for mode in ("complete", "budgeted"):
        cal_m = evaluate(cal, catalog, weights, tag="cal", mode=mode)
        gold_m = sorted(r["gold_score"] for r in cal_m["rows"]
                        if r["label"] == IN_DSL and r["gold_score"] is not None)
        nm = len(gold_m)
        km = max(1, math.floor(DELTA_SEM * (nm + 1)))
        qh = gold_m[km - 1]
        sig = sorted(r["ood_signal"] for r in cal_m["rows"] if r["label"] != IN_DSL)
        qo = sig[max(0, int((1 - OOD_TARGET_RECALL) * len(sig)))] if sig else 1.0
        te = evaluate(test, catalog, weights, qhat=qh, qood=qo, tag="test", mode=mode)
        ce = evaluate(cal, catalog, weights, qhat=qh, qood=qo, tag="cal", mode=mode)
        modes[mode] = dict(qhat=qh, qood=qo, n_cal_in_dsl=nm, k=km,
                           calibration=summarise(ce["rows"], qh, qo),
                           test=summarise(te["rows"], qh, qo))
        json.dump(te["rows"], open(os.path.join(OUT, f"e2e_test_rows_{CATALOG_VERSION}_{mode}.json"), "w"))
        print(f"  [{mode}] qhat={qh:.6f} "
              f"rho_gen={modes[mode]['test']['rho_gen']['point']} "
              f"rho_e2e={modes[mode]['test']['rho_e2e']['point']}")
    report = dict(catalog_version=CATALOG_VERSION,
                  dsl=dsl_manifest(), weights=weights, dev_fit=fit,
                  delta_sem=DELTA_SEM, ood_target_recall=OOD_TARGET_RECALL,
                  splits=dict(dev=len(dev), cal=len(cal), test=len(test)),
                  leak_audit=json.load(open(os.path.join(OUT, "leak_audit.json"))),
                  modes=modes)
    json.dump(report, open(os.path.join(OUT, f"e2e_report_{CATALOG_VERSION}.json"), "w"), indent=1)
    print(json.dumps(modes["complete"]["test"], indent=1))


if __name__ == "__main__":
    main()
