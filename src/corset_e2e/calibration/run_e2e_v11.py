"""v11 pipeline run: fit on dev, calibrate on cal, one frozen test pass.

Protocol is run_e2e_v4.py's, unchanged: coordinate-search score weights on
the development split, qhat/qood calibrated on the calibration split at
delta_sem=0.10, a single frozen test execution per mode.  The ONLY change
is the metric-axis licenser: CatalogV11 (IDF name channel + repository
context channel + widened verbatim; license_v11.py) served over the v4u
namespace of record (v4 file harvest union v3 leave-one-cluster-out).
Slot grids, additive scorer, retention and in_pool are the frozen modules.

Also computes the registered licensing-only endpoints on the same frozen
test pass (REGISTRATION_V11.md): recall@k curve, ablations (name-only,
context-only, no-normalization, frozen-v4-license-on-v4u), per-repo
rho_gen, and the union with the archived v6 semantic selections.

Run: PYTHONPATH=. python3 corset_e2e/calibration/run_e2e_v11.py
Writes results/e2e/e2e_report_v11.json (+ e2e_test_rows_v11_*.json)
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.dsl.schema import IN_DSL, classify_target, target_reading  # noqa: E402
from corset_e2e.dsl.schema import manifest as dsl_manifest  # noqa: E402
from corset_e2e.generator.generate import (  # noqa: E402
    Catalog, extract_slots, pool_size, tokens)
from corset_e2e.generator import license_v11 as L11  # noqa: E402
from corset_e2e.generator.license_v11 import (  # noqa: E402
    CatalogV11, RepoIndexV11, wide_verbatim)
from corset_e2e.scoring.deterministic import DEFAULT_WEIGHTS, Scorer  # noqa: E402
from corset_e2e.source_grounded.build_catalog_v3 import RepoCatalogsV3  # noqa: E402
from corset_e2e.source_grounded.build_catalog_v4 import RepoCatalogsV4  # noqa: E402
from corset_e2e.calibration.run_e2e_v4 import (  # noqa: E402
    DELTA_SEM, OOD_TARGET_RECALL, WEIGHT_GRID, cluster_bootstrap, gold_info,
    hid, load_split, summarise, vis)

OUT = os.path.join(ROOT, "results/e2e")
KS = (16, 64, 256)


# ------------------------------------------------------------- v11 serving
class V11Server:
    """Per-unit CatalogV11 views over shared per-repo indexes."""

    def __init__(self):
        self.v4u = RepoCatalogsV4(union_with_v3=True)
        v3 = RepoCatalogsV3(os.path.join(OUT, "metric_catalog_v3.json"))
        self.idx: Dict[str, RepoIndexV11] = {}
        for repo, toks_ in sorted(self.v4u.tokens.items()):
            v3all: set = set()
            for cid, r in v3.cluster_repo.items():
                if r == repo:
                    v3all |= v3.cluster_metrics.get(cid, set())
            ctx_path = os.path.join(OUT, "context_v11", f"{repo}.ctx.json")
            ctx = json.load(open(ctx_path))["profiles"] if os.path.exists(ctx_path) else {}
            self.idx[repo] = RepoIndexV11(sorted(set(toks_) | v3all), ctx)
        self.sizes = {r: len(i.metrics) for r, i in self.idx.items()}

    def catalog_for(self, entry: dict, visible: dict, k: int = None) -> CatalogV11:
        allowed = set(self.v4u.metrics_for(entry["cluster_id"]))
        text = f"{visible.get('text', '')} {visible.get('alert_name', '')}"
        return CatalogV11.from_index(self.idx[entry["repo"]], allowed, text,
                                     k=(k or L11.K_PRIMARY))


def unit_view(entry: dict, server: V11Server, weights, mode="complete") -> dict:
    v = vis(entry["rule_id"])
    cat = server.catalog_for(entry, v)
    slots = extract_slots(v, cat, mode=mode)
    sc = Scorer(v, slots, weights)
    return dict(rid=entry["rule_id"], cluster=entry["cluster_id"],
                repo=entry["repo"], visible=v, slots=slots, scorer=sc)


def fit_weights(dev: Sequence[dict], server: V11Server, sample=220) -> dict:
    import random
    rng = random.Random(7)
    pool = list(dev)
    rng.shuffle(pool)
    pool = pool[:sample]

    def objective(w) -> float:
        tot, n = 0.0, 0
        for e in pool:
            u = unit_view(e, server, w)
            label, s, n_above = gold_info(u)
            if label != IN_DSL:
                continue
            tot += math.log1p(n_above)
            n += 1
        return tot / max(n, 1)

    w = dict(DEFAULT_WEIGHTS)
    best = objective(w)
    for _ in range(2):
        for kk, vals in WEIGHT_GRID.items():
            for vv in vals:
                if vv == w[kk]:
                    continue
                cand = dict(w)
                cand[kk] = vv
                o = objective(cand)
                if o < best - 1e-9:
                    best, w = o, cand
    return dict(weights=w, dev_objective=best, n_dev_units=len(pool))


def evaluate(entries, server, weights, qhat=None, qood=None, tag="",
             mode="complete") -> dict:
    rows = []
    for e in entries:
        u = unit_view(e, server, weights, mode=mode)
        label, s, n_above = gold_info(u)
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


# --------------------------------------------- licensing-only endpoints
def licensing_endpoints(test, server: V11Server) -> dict:
    """recall@k curve, ablations, per-repo breakdown, v6 union.

    All computed on the SAME frozen test pass inputs; each variant is a
    deterministic function of visible data and committed artifacts.
    """
    v6sel: Dict[str, List[str]] = {}
    v6dir = os.path.join(OUT, "v6_selections")
    if os.path.isdir(v6dir):
        for f in os.listdir(v6dir):
            if f.startswith("out_") and f.endswith(".txt"):
                for line in open(os.path.join(v6dir, f)):
                    line = line.strip()
                    if "|" in line:
                        rid, ms = line.split("|", 1)
                        v6sel[rid] = [m for m in ms.split(",") if m]

    # the no-normalization ablation needs its OWN indexes: canon() runs at
    # index-build time, so flipping NORM_ON only at license time would score
    # a canonicalized index against raw text and measure nothing
    old_norm = L11.NORM_ON
    L11.NORM_ON = False
    server_nn = V11Server()
    L11.NORM_ON = old_norm

    reach = []
    for e in test:
        h = hid(e["rule_id"])
        if classify_target(h) != IN_DSL:
            continue
        reach.append((e, target_reading(h)[0]))

    hits = defaultdict(lambda: defaultdict(int))
    per_repo = defaultdict(lambda: defaultdict(int))
    n_avail = 0
    for e, gold in reach:
        v = vis(e["rule_id"])
        toks = tokens(v.get("text", "")) + tokens(v.get("alert_name", ""))
        text = f"{v.get('text', '')} {v.get('alert_name', '')}".lower()
        wv = set(wide_verbatim(text))
        allowed = set(server.v4u.metrics_for(e["cluster_id"]))
        avail = gold in allowed
        n_avail += avail
        cat = server.catalog_for(e, v, k=max(KS))
        ranked = cat.license(toks, [])

        variants = {}
        variants["v11"] = {k: (gold in set(ranked[:k])) or (gold in wv)
                           for k in KS}
        old_wctx = L11.W_CTX
        L11.W_CTX = 0.0
        name_only = cat.license(toks, [])
        L11.W_CTX = 1e9
        ctx_only = cat.license(toks, [])
        L11.W_CTX = old_wctx
        variants["name_only"] = {64: (gold in set(name_only[:64])) or (gold in wv)}
        variants["ctx_only"] = {64: (gold in set(ctx_only[:64])) or (gold in wv)}
        old_norm2 = L11.NORM_ON
        L11.NORM_ON = False
        no_norm = server_nn.catalog_for(e, v, k=64).license(toks, [])
        L11.NORM_ON = old_norm2
        variants["no_norm"] = {64: (gold in set(no_norm[:64])) or (gold in wv)}
        frozen = Catalog(sorted(allowed))
        slots_f = extract_slots(v, frozen, mode="complete")
        variants["frozen_v4_on_v4u"] = {64: gold in set(slots_f["metrics"])}
        v6m = v6sel.get(e["rule_id"], [])
        variants["v11_union_v6"] = {64: (gold in set(ranked[:64])) or
                                    (gold in wv) or (gold in set(v6m))}
        variants["v6_archived"] = {16: gold in set(v6m)}

        for var, kk in variants.items():
            for k, hit in kk.items():
                hits[var][k] += hit
        per_repo[e["repo"]]["n"] += 1
        per_repo[e["repo"]]["hit"] += variants["v11"][64]

    n = len(reach)
    return dict(
        n_grammar_reachable=n, n_gold_available_v4u=n_avail,
        availability_v4u=round(n_avail / n, 4),
        recall={var: {str(k): round(c / n, 4) for k, c in ks.items()}
                for var, ks in hits.items()},
        per_repo={r: dict(n=d["n"], rho_gen_at64=round(d["hit"] / d["n"], 4))
                  for r, d in sorted(per_repo.items())})


def main() -> None:
    dev, cal, test = load_split("dev"), load_split("cal"), load_split("test")
    print(f"[1] v11 server (v4u namespaces + context indexes)")
    server = V11Server()
    print("    sizes:", server.sizes)

    print("[3] fitting score weights on dev")
    fit = fit_weights(dev, server)
    weights = fit["weights"]
    print("    weights:", weights, "obj:", round(fit["dev_objective"], 4))

    print(f"[5/6] calibrate ({len(cal)}) and one frozen test pass ({len(test)})")
    modes = {}
    for mode in ("complete", "budgeted"):
        cal_m = evaluate(cal, server, weights, tag="cal", mode=mode)
        gold_m = sorted(r["gold_score"] for r in cal_m["rows"]
                        if r["label"] == IN_DSL and r["gold_score"] is not None)
        nm = len(gold_m)
        if nm < 20:
            raise SystemExit(f"only {nm} in-DSL cal units in mode {mode}")
        km = max(1, math.floor(DELTA_SEM * (nm + 1)))
        qh = gold_m[km - 1]
        sig = sorted(r["ood_signal"] for r in cal_m["rows"] if r["label"] != IN_DSL)
        qo = sig[max(0, int((1 - OOD_TARGET_RECALL) * len(sig)))] if sig else 1.0
        te = evaluate(test, server, weights, qhat=qh, qood=qo, tag="test", mode=mode)
        ce = evaluate(cal, server, weights, qhat=qh, qood=qo, tag="cal", mode=mode)
        modes[mode] = dict(qhat=qh, qood=qo, n_cal_in_dsl=nm, k=km,
                           calibration=summarise(ce["rows"], qh, qo),
                           test=summarise(te["rows"], qh, qo))
        json.dump(te["rows"], open(os.path.join(
            OUT, f"e2e_test_rows_v11_{mode}.json"), "w"))
        print(f"  [{mode}] qhat={qh:.6f} "
              f"rho_gen={modes[mode]['test']['rho_gen']['point']} "
              f"rho_e2e={modes[mode]['test']['rho_e2e']['point']}")

    print("[7] licensing-only endpoints on the frozen test pass")
    lic = licensing_endpoints(test, server)
    print(json.dumps({k: v for k, v in lic.items() if k != "per_repo"}, indent=1))

    audit_path = os.path.join(OUT, "leak_audit_v11.json")
    audit = json.load(open(audit_path)) if os.path.exists(audit_path) else None
    report = dict(catalog_version="v11",
                  licenser=dict(k_primary=L11.K_PRIMARY, w_ctx=L11.W_CTX,
                                abbrev_pairs=len(L11.ABBREV)),
                  dsl=dsl_manifest(), weights=weights, dev_fit=fit,
                  delta_sem=DELTA_SEM, ood_target_recall=OOD_TARGET_RECALL,
                  splits=dict(dev=len(dev), cal=len(cal), test=len(test)),
                  leak_audit=audit, licensing_endpoints=lic, modes=modes)
    json.dump(report, open(os.path.join(OUT, "e2e_report_v11.json"), "w"),
              indent=1)
    print("wrote e2e_report_v11.json")


if __name__ == "__main__":
    main()
