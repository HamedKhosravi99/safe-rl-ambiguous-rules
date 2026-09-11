"""REGISTRATION_V27: anatomy of the calibrated open-domain antichain, and the
exact decide stage on the rule's own metric.

Run: SAORL_CONFORMAL=1 PYTHONPATH=. python3 corset_e2e/analysis/open_domain_decide.py
Writes results/e2e/open_domain_decide.json
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.analysis.faithful_recalibration import faithful_g0  # noqa: E402
from corset_e2e.calibration.run_e2e_v4 import gold_info, hid, load_split  # noqa: E402
from corset_e2e.calibration.run_e2e_v11 import V11Server, unit_view  # noqa: E402
from corset_e2e.dsl.schema import IN_DSL, classify_target, target_reading  # noqa: E402
from saorl.benchmark_sg.control_suite import DUR_CAP, compile_instance  # noqa: E402
from saorl.benchmark_sg.exact_nonnested import analyse  # noqa: E402

R = os.path.join(ROOT, "results/e2e")
OUT = os.path.join(R, "open_domain_decide.json")
D = 0.05
K_MAX = 4
UPPER = {">", ">="}
LOWER = {"<", "<="}


class _R:
    def __init__(self, threshold, for_s):
        self.threshold = threshold
        self.for_s = for_s


def antichain_groups(tables, qhat):
    mt, ct, tt, wt, at, ft = tables
    if not all((mt, ct, tt, wt, at, ft)):
        return []
    tail = max(sc for _k, sc in tt) + max(sc for _k, sc in ft)
    out = []
    for m, sm in mt:
        for c, sc in ct:
            for w, sw in wt:
                for a, sa in at:
                    s4 = sm + sc + sw + sa
                    if s4 + tail >= qhat - 1e-12:
                        out.append((m, c, w, a, s4))
    return out


def pareto_min(pairs, upper):
    pts = sorted({((th if upper else -th), f) for th, f in pairs})
    out = []
    best_f = math.inf
    for x, f in pts:
        if f < best_f:
            out.append((x, f))
            best_f = f
    return out


def main():
    t0 = time.perf_counter()
    rep = json.load(open(os.path.join(R, "e2e_report_v11.json")))
    weights = rep["weights"]
    qhat = rep["modes"]["complete"]["qhat"]
    arch = {r["rid"]: r for r in json.load(open(os.path.join(R, "e2e_test_rows_v11_complete.json")))}
    test = [e for e in load_split("test") if classify_target(hid(e["rule_id"])) == IN_DSL]
    print(f"grammar-reachable test units: {len(test)}; q-hat {qhat}", flush=True)
    server = V11Server()
    rows = []
    for i, e in enumerate(test):
        u = unit_view(e, server, weights, mode="complete")
        label, s, _n_above = gold_info(u)
        if label != IN_DSL:
            continue
        rid = u["rid"]
        g = target_reading(hid(rid))
        g_metric, g_cmp, g_thr, g_win, g_agg, g_for = g
        retained = bool(s >= qhat - 1e-12)
        groups = antichain_groups(u["scorer"].tables, qhat)
        a_row = arch.get(rid, {})
        if retained and "maximal_size" in a_row:
            assert len(groups) == a_row["maximal_size"], (rid, len(groups), a_row["maximal_size"])
        metrics = {gr[0] for gr in groups}
        same = [gr for gr in groups if gr[0] == g_metric]
        direction = "upper" if g_cmp in UPPER else ("lower" if g_cmp in LOWER else "other")
        slice_groups = [gr for gr in same if gr[3] == g_agg and
                        ((gr[1] in UPPER) if direction == "upper" else (gr[1] in LOWER) if direction == "lower" else False)]
        mt, ct, tt, wt, at, ft = u["scorer"].tables
        pairs = set()
        for (_m, _c, _w, _a, s4) in slice_groups:
            for th, st in tt:
                for f, sf in ft:
                    if s4 + st + sf >= qhat - 1e-12:
                        pairs.add((float(th), float(f)))
        row = dict(rid=rid, repo=u["repo"], gold_score=s, retained=retained, faithful=bool(faithful_g0(hid(rid))),
                   gold=dict(metric=g_metric, comparator=g_cmp, threshold=g_thr, window=g_win, aggregation=g_agg, for_s=g_for),
                   n_maximal=len(groups), n_metrics=len(metrics), n_same_metric=len(same),
                   share_same_metric=(len(same) / len(groups) if groups else None),
                   n_other_metric=len(groups) - len(same), n_slice_groups=len(slice_groups), n_slice_pairs=len(pairs),
                   direction=direction)
        if not retained:
            row["status"] = "not retained"
        elif direction == "other":
            row["status"] = "non-order comparator"
        elif not pairs:
            row["status"] = "empty slice"
        else:
            pm = pareto_min(pairs, direction == "upper")
            K = len(pm)
            row["K"] = K
            row["slice_readings"] = [dict(threshold=(x if direction == "upper" else -x), for_s=f) for x, f in pm]
            caps = [max(1, min(DUR_CAP, int(round(f / 60.0)))) for _x, f in pm]
            L = len({x for x, _f in pm}) + 1
            row["n_states_needed"] = 1 + L * math.prod(c + 1 for c in caps)
            if K == 1:
                row["status"] = "nested slice: strictest reading protects the set"
            elif K <= K_MAX:
                m = compile_instance([_R(x, f) for x, f in pm])
                a = analyse(m, D)
                row["status"] = "compiled"
                row["n_states"] = m["nS"]
                row["decide"] = dict(feasible=a["feasible"], V_U=a["V_U"], V_best_single=a["V_best_single"],
                                     poa=a["poa"], rpoa=a["rpoa"], fires_value=bool(a["fires_value"]),
                                     face_clear=a.get("face_clear"), screen_gap=a.get("screen_gap"))
            else:
                row["status"] = "K above compiled range"
        rows.append(row)
        if i % 25 == 0:
            print(f"  {i}/{len(test)} {rid} retained={retained} maximal={len(groups)} metrics={len(metrics)} same={len(same)} K={row.get('K')} status={row.get('status')}", flush=True)

    def agg(sub, name):
        n = len(sub)
        if n == 0:
            return dict(n=0)
        med = lambda k: float(statistics.median(r[k] for r in sub if r.get(k) is not None)) if any(r.get(k) is not None for r in sub) else None
        st = {}
        for r in sub:
            st[r["status"]] = st.get(r["status"], 0) + 1
        comp = [r for r in sub if r["status"] == "compiled"]
        ks = [r["K"] for r in sub if "K" in r]
        out = dict(n=n, median_maximal=med("n_maximal"), median_metrics=med("n_metrics"), median_same_metric=med("n_same_metric"),
                   median_share_same_metric=med("share_same_metric"), median_other_metric=med("n_other_metric"),
                   share_same_below_half=(sum(1 for r in sub if r["share_same_metric"] is not None and r["share_same_metric"] < 0.5) / n),
                   status=st, n_with_K=len(ks), K_eq1=sum(k == 1 for k in ks), K_2_to_4=sum(2 <= k <= K_MAX for k in ks), K_gt4=sum(k > K_MAX for k in ks),
                   K_median=(float(statistics.median(ks)) if ks else None), K_max=(max(ks) if ks else None),
                   n_compiled=len(comp), fires_value=sum(1 for r in comp if r["decide"]["fires_value"]),
                   face_clear=sum(1 for r in comp if r["decide"].get("face_clear")), infeasible=sum(1 for r in comp if not r["decide"]["feasible"]),
                   rpoa_median=(float(statistics.median(r["decide"]["rpoa"] for r in comp if r["decide"]["rpoa"] is not None)) if comp else None),
                   states_needed_median_gt4=(float(statistics.median(r["n_states_needed"] for r in sub if r.get("K", 0) > K_MAX)) if any(r.get("K", 0) > K_MAX for r in sub) else None),
                   states_needed_max=(max((r["n_states_needed"] for r in sub if "n_states_needed" in r), default=None)))
        return out

    ret = [r for r in rows if r["retained"]]
    res = dict(registration="REGISTRATION_V27.md", qhat=qhat, d=D, k_max=K_MAX,
               summary=dict(all_in_grammar=agg(rows, "all"), retained=agg(ret, "retained"),
                            retained_faithful=agg([r for r in ret if r["faithful"]], "retained_faithful")),
               rows=rows, seconds=round(time.perf_counter() - t0, 1))
    res["branch"] = dict(median_share_same_metric_retained=res["summary"]["retained"]["median_share_same_metric"],
                         metric_axis_blocks=bool(res["summary"]["retained"]["median_share_same_metric"] is not None and res["summary"]["retained"]["median_share_same_metric"] < 0.5))
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1))
    print(json.dumps(res["branch"]), f"wrote {OUT} in {res['seconds']} s")


if __name__ == "__main__":
    main()
