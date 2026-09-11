"""V51 step 2: Keep recalibrated on the calibration split for the union
generator (frozen v11 retrieval license + the V51 LLM selections), then one
evaluation of the test split. Side by side with v11 at 0.41 (archived), the
archived-v6 union at 0.41 (V50) and the new union at 0.41.

Run: SAORL_CONFORMAL=1 PYTHONPATH=. python3 corset_e2e/analysis/union_calibrated_v51.py
Writes results/e2e/union_calibrated_v51.json
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.calibration.run_e2e_v11 import V11Server  # noqa: E402
from corset_e2e.calibration.run_e2e_v4 import hid, load_split, vis, gold_info, cluster_bootstrap, DELTA_SEM  # noqa: E402
from corset_e2e.generator.generate import extract_slots, pool_size  # noqa: E402
from corset_e2e.scoring.deterministic import Scorer  # noqa: E402
from corset_e2e.dsl.schema import IN_DSL, classify_target, target_reading  # noqa: E402
from corset_e2e.analysis.open_domain_decide import antichain_groups, pareto_min, faithful_g0, UPPER, LOWER, DUR_CAP, K_MAX, _R, D  # noqa: E402
from corset_e2e.analysis.union_pass_v50 import UnionCatalog  # noqa: E402
from saorl.benchmark_sg.control_suite import compile_instance  # noqa: E402
from saorl.benchmark_sg.exact_nonnested import analyse  # noqa: E402

R = os.path.join(ROOT, "results/e2e")
SEL = os.path.join(R, "v51_selections", "selections.jsonl")
OUT = os.path.join(R, "union_calibrated_v51.json")


def load_sel():
    sel = {}
    for line in open(SEL):
        d = json.loads(line); sel[d["rid"]] = d["names"]
    return sel


def view(entry, server, weights, extra):
    v = vis(entry["rule_id"]); cat = server.catalog_for(entry, v)
    if extra is not None:
        cat = UnionCatalog(cat, extra)
    slots = extract_slots(v, cat, mode="complete"); sc = Scorer(v, slots, weights)
    return dict(rid=entry["rule_id"], cluster=entry["cluster_id"], repo=entry["repo"], visible=v, slots=slots, scorer=sc)


def gold_score(u):
    label, s, _n = gold_info(u)
    return label, (None if s == float("-inf") else s)


def eval_row(u, qhat, label, s):
    sc = u["scorer"]; t0 = time.perf_counter()
    cnt, _ = sc.retained(qhat); groups = antichain_groups(sc.tables, qhat)
    row = dict(label=label, gold_score=s, n_metrics_licensed=len(u["slots"]["metrics"]), pool=pool_size(u["slots"]), set_size=cnt,
               maximal_size=len(groups), n_retained_metrics=len({g[0] for g in groups}), retained_gold=bool(label == IN_DSL and s is not None and s >= qhat - 1e-12),
               decide_lps=0, decide_seconds=0.0)
    if row["retained_gold"]:
        g = target_reading(hid(u["rid"])); g_metric, g_cmp, g_thr, g_win, g_agg, g_for = g
        direction = "upper" if g_cmp in UPPER else ("lower" if g_cmp in LOWER else "other")
        same = [gr for gr in groups if gr[0] == g_metric]; row["n_same_metric"] = len(same)
        row["share_same_metric"] = (len(same) / len(groups) if groups else None)
        if direction != "other":
            slice_groups = [gr for gr in same if gr[3] == g_agg and ((gr[1] in UPPER) if direction == "upper" else (gr[1] in LOWER))]
            mt, ct, tt, wt, at, ft = sc.tables; pairs = set()
            for (_m, _c, _w, _a, s4) in slice_groups:
                for th, st in tt:
                    for f, sf in ft:
                        if s4 + st + sf >= qhat - 1e-12:
                            pairs.add((float(th), float(f)))
            if pairs:
                pm = pareto_min(pairs, direction == "upper"); K = len(pm); row["K"] = K
                caps = [max(1, min(DUR_CAP, int(round(f / 60.0)))) for _x, f in pm]; L = len({x for x, _f in pm}) + 1
                row["n_states_needed"] = 1 + L * math.prod(c + 1 for c in caps)
                if 2 <= K <= K_MAX:
                    t1 = time.perf_counter(); m = compile_instance([_R(x if direction == "upper" else -x, f) for x, f in pm]); a = analyse(m, D)
                    row["decide_lps"] = K + 1 + K * (K - 1); row["decide_seconds"] = time.perf_counter() - t1
                    row["decide"] = dict(feasible=a["feasible"], fires_value=bool(a["fires_value"]), face_clear=a.get("face_clear"), rpoa=a.get("rpoa"))
                elif K > K_MAX:
                    row["decide_lps"] = None
        else:
            row["K"] = None
    row["antichain_seconds"] = time.perf_counter() - t0
    return row


def main():
    t0 = time.perf_counter()
    rep = json.load(open(os.path.join(R, "e2e_report_v11.json"))); weights = rep["weights"]; q_v11 = rep["modes"]["complete"]["qhat"]
    sel = load_sel(); server = V11Server()
    v50 = {r["rid"]: r for r in json.load(open(os.path.join(R, "union_pass_v50.json")))["rows"]}
    # ---- calibration split: gold scores under v11 and under the union
    cal = [e for e in load_split("cal") if classify_target(hid(e["rule_id"])) == IN_DSL]
    cal_rows = []
    for i, e in enumerate(cal):
        rid = e["rule_id"]
        l0, s0 = gold_score(view(e, server, weights, None))
        l1, s1 = gold_score(view(e, server, weights, sel.get(rid, [])))
        cal_rows.append(dict(rid=rid, repo=e["repo"], cluster=e["cluster_id"], v11=dict(label=l0, gold_score=s0), union=dict(label=l1, gold_score=s1),
                             faithful=bool(faithful_g0(hid(rid))), n_llm=len(sel.get(rid, []))))
        if i % 60 == 0:
            print(f"  cal {i}/{len(cal)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    def quantile(rows, arm):
        gold = sorted(r[arm]["gold_score"] for r in rows if r[arm]["label"] == IN_DSL)
        n = len(gold); k = max(1, math.floor(DELTA_SEM * (n + 1)))
        return dict(n_in_dsl=n, k=k, qhat=gold[k - 1], recall=n / len(rows), cal_retention_at_qhat=sum(g >= gold[k - 1] - 1e-12 for g in gold) / n)
    q11 = quantile(cal_rows, "v11"); qun = quantile(cal_rows, "union")
    assert abs(q11["qhat"] - q_v11) < 1e-9 and q11["n_in_dsl"] == rep["modes"]["complete"]["n_cal_in_dsl"], (q11, q_v11)
    print(f"calibration: v11 n_in_dsl {q11['n_in_dsl']} k {q11['k']} qhat {q11['qhat']:.4f} (archived {q_v11:.4f}); union n_in_dsl {qun['n_in_dsl']} k {qun['k']} qhat {qun['qhat']:.4f}", flush=True)
    # ---- test split
    test = [e for e in load_split("test") if classify_target(hid(e["rule_id"])) == IN_DSL]
    rows = []
    for i, e in enumerate(test):
        rid = e["rule_id"]; u = view(e, server, weights, sel.get(rid, [])); label, s = gold_score(u)
        rows.append(dict(rid=rid, repo=e["repo"], cluster=e["cluster_id"], faithful=bool(faithful_g0(hid(rid))), n_llm=len(sel.get(rid, [])),
                         llm_hits_gold=(target_reading(hid(rid))[0] in set(sel.get(rid, []))),
                         union_new_q=eval_row(u, qun["qhat"], label, s), union_old_q=eval_row(u, q_v11, label, s),
                         v11=v50[rid]["v11"], union_v6_old_q=v50[rid]["union"]))
        if i % 40 == 0:
            print(f"  test {i}/{len(test)} {rid} union {label} q_new maximal {rows[-1]['union_new_q']['maximal_size']} q_old {rows[-1]['union_old_q']['maximal_size']} ({time.perf_counter()-t0:.0f}s)", flush=True)

    def med(xs):
        xs = [x for x in xs if x is not None]; return float(statistics.median(xs)) if xs else None

    def mean(xs):
        xs = [x for x in xs if x is not None]; return float(sum(xs) / len(xs)) if xs else None

    def agg(sub, arm):
        rs = [r[arm] for r in sub]; gen = [r for r in rs if r["label"] == IN_DSL]; ret = [r for r in gen if r["retained_gold"]]
        byc = defaultdict(list)
        for r, x in zip(sub, rs):
            if x["label"] == IN_DSL:
                byc[r["cluster"]].append(1 if x["retained_gold"] else 0)
        pt, lo, hi = cluster_bootstrap(byc) if byc else (None, None, None)
        ks = [r.get("K") for r in ret if r.get("K") is not None]
        return dict(n=len(rs), licensed=len(gen), proposal_recall=len(gen) / len(rs) if rs else None, retained=len(ret), retained_recall=len(ret) / len(rs) if rs else None,
                    retention_given_generated=(len(ret) / len(gen) if gen else None), retention_ci95=[lo, hi],
                    pool_mean=mean([r["pool"] for r in rs]), pool_median=med([r["pool"] for r in rs]),
                    set_mean=mean([r["set_size"] for r in rs]), set_median=med([r["set_size"] for r in rs]),
                    maximal_mean=mean([r["maximal_size"] for r in rs]), maximal_median=med([r["maximal_size"] for r in rs]),
                    retained_metrics_median=med([r["n_retained_metrics"] for r in rs]), metrics_licensed_median=med([r["n_metrics_licensed"] for r in rs]),
                    K_eq1=sum(k == 1 for k in ks), K_2_4=sum(2 <= k <= K_MAX for k in ks), K_gt4=sum(k > K_MAX for k in ks), K_none=sum(1 for r in ret if r.get("K") is None),
                    safe_within_slice_by_dominance=(sum(k == 1 for k in ks) / len(ks) if ks else None),
                    decide_lps_total=sum((r.get("decide_lps") or 0) for r in ret), decide_seconds_total=sum(r.get("decide_seconds", 0.0) for r in ret),
                    antichain_seconds_median=med([r.get("antichain_seconds") for r in rs]),
                    share_same_metric_median=med([r.get("share_same_metric") for r in ret]))

    arms = ("v11", "union_v6_old_q", "union_old_q", "union_new_q")
    res = dict(registration="V51 (REGISTRATION_V28.md)", delta_sem=DELTA_SEM, calibration=dict(v11=q11, union=qun, n_cal_reachable=len(cal),
               union_recall_cal_faithful=(sum(1 for r in cal_rows if r["faithful"] and r["union"]["label"] == IN_DSL) / max(1, sum(1 for r in cal_rows if r["faithful"])))),
               llm=dict(test_units_with_selection=sum(1 for r in rows if r["n_llm"] > 0), test_llm_alone_recall=sum(r["llm_hits_gold"] for r in rows) / len(rows),
                        cal_units_with_selection=sum(1 for r in cal_rows if r["n_llm"] > 0)),
               test=dict(all={a: agg(rows, a) for a in arms}, faithful={a: agg([r for r in rows if r["faithful"]], a) for a in arms}),
               per_repo={rp: {a: agg([r for r in rows if r["repo"] == rp], a) for a in ("v11", "union_new_q")} for rp in sorted({r["repo"] for r in rows})},
               cal_rows=cal_rows, rows=rows, seconds=round(time.perf_counter() - t0, 1))
    a = res["test"]["all"]
    res["branch"] = dict(recall_ge_070=a["union_new_q"]["proposal_recall"] >= 0.70,
                         retention_ge_085_ci_covers_090=(a["union_new_q"]["retention_given_generated"] >= 0.85 and a["union_new_q"]["retention_ci95"][0] <= 0.90 <= a["union_new_q"]["retention_ci95"][1]),
                         set_and_antichain_le_125=(a["union_new_q"]["set_median"] <= 1.25 * a["v11"]["set_median"] and a["union_new_q"]["maximal_median"] <= 1.25 * a["v11"]["maximal_median"]),
                         K1_ge_090=(a["union_new_q"]["safe_within_slice_by_dominance"] or 0) >= 0.90)
    res["branch"]["survives"] = all(res["branch"].values())
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nCALIBRATION", json.dumps(res["calibration"])); print("LLM", json.dumps(res["llm"]))
    for lab in ("all", "faithful"):
        print(f"== test {lab}")
        for arm in arms:
            x = res["test"][lab][arm]
            print(f"   {arm:16s} recall {x['proposal_recall']:.3f} retained {x['retained']:3d} ({x['retained_recall']:.3f}) ret|gen {x['retention_given_generated']:.3f} CI {x['retention_ci95']} pool med {x['pool_median']:.3e} set med {x['set_median']:.3e} mean {x['set_mean']:.3e} maximal med {x['maximal_median']} mean {x['maximal_mean']:.0f} metrics {x['metrics_licensed_median']} K1 {x['K_eq1']} K2-4 {x['K_2_4']} K>4 {x['K_gt4']} Knone {x['K_none']} LPs {x['decide_lps_total']} decide_s {x['decide_seconds_total']:.2f}")
    print("BRANCH", json.dumps(res["branch"]))
    for rp, v in res["per_repo"].items():
        print(f"   {rp:30s} n {v['v11']['n']:3d} recall {v['v11']['proposal_recall']:.3f} -> {v['union_new_q']['proposal_recall']:.3f} retained {v['v11']['retained']} -> {v['union_new_q']['retained']} maximal med {v['v11']['maximal_median']} -> {v['union_new_q']['maximal_median']}")
    print(f"wrote {OUT} in {res['seconds']} s")


if __name__ == "__main__":
    main()
