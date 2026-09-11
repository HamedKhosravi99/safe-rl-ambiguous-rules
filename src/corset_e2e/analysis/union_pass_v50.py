"""V50: v11 retrieval licenser UNION archived v6 LLM selections, on the 315
grammar-reachable test units, complete mode, frozen weights and q-hat.
Reports recall against candidate/retained/antichain growth and the
own-metric Decide slice. Registration V50 in REGISTRATION_V28.md.

Run: SAORL_CONFORMAL=1 PYTHONPATH=. python3 corset_e2e/analysis/union_pass_v50.py
Writes results/e2e/union_pass_v50.json
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.calibration.run_e2e_v11 import V11Server  # noqa: E402
from corset_e2e.calibration.run_e2e_v4 import hid, load_split, vis, gold_info  # noqa: E402
from corset_e2e.generator.generate import extract_slots, pool_size  # noqa: E402
from corset_e2e.scoring.deterministic import Scorer  # noqa: E402
from corset_e2e.dsl.schema import IN_DSL, classify_target, target_reading  # noqa: E402
from corset_e2e.analysis.open_domain_decide import antichain_groups, pareto_min, faithful_g0, UPPER, LOWER, DUR_CAP, K_MAX  # noqa: E402

R = os.path.join(ROOT, "results/e2e")
OUT = os.path.join(R, "union_pass_v50.json")


class UnionCatalog:
    """The frozen v11 catalog with the unit's archived v6 selections appended (duck-types Catalog)."""

    def __init__(self, base, extra):
        self.base = base; self.idx = base.idx; self.extra = [m for m in extra if m]
        self.allowed = set(base.allowed) | set(self.extra); self.metrics = sorted(self.allowed); self.k = base.k; self._subtok = None

    @property
    def subtok(self):
        if self._subtok is None:
            self._subtok = {m: self.idx.csub.get(m, frozenset()) for m in self.allowed}
        return self._subtok

    def license(self, toks, verbatim):
        out = list(self.base.license(toks, verbatim))
        for m in self.extra:
            if m not in out:
                out.append(m)
        return out


def load_v6():
    sel = {}
    for f in glob.glob(os.path.join(R, "v6_selections", "out_*.txt")):
        for line in open(f):
            line = line.strip()
            if not line or "|" not in line:
                continue
            rid, ms = line.split("|", 1)
            sel[rid.strip()] = [m.strip() for m in ms.split(",") if m.strip()]
    return sel


def unit_row(entry, server, weights, qhat, extra):
    v = vis(entry["rule_id"]); cat = server.catalog_for(entry, v)
    if extra is not None:
        cat = UnionCatalog(cat, extra)
    slots = extract_slots(v, cat, mode="complete"); sc = Scorer(v, slots, weights)
    u = dict(rid=entry["rule_id"], cluster=entry["cluster_id"], repo=entry["repo"], visible=v, slots=slots, scorer=sc)
    label, s, n_above = gold_info(u)
    cnt, _ = sc.retained(qhat); groups = antichain_groups(sc.tables, qhat)
    row = dict(label=label, gold_score=(None if s == float("-inf") else s), n_above=n_above, n_metrics_licensed=len(slots["metrics"]),
               pool=pool_size(slots), set_size=cnt, maximal_size=len(groups), n_retained_metrics=len({g[0] for g in groups}),
               retained_gold=bool(label == IN_DSL and s >= qhat - 1e-12))
    if row["retained_gold"]:
        g = target_reading(hid(entry["rule_id"])); g_metric, g_cmp, g_thr, g_win, g_agg, g_for = g
        direction = "upper" if g_cmp in UPPER else ("lower" if g_cmp in LOWER else "other")
        same = [gr for gr in groups if gr[0] == g_metric]
        row["n_same_metric"] = len(same); row["share_same_metric"] = (len(same) / len(groups) if groups else None)
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
                row["decide_lps"] = 0 if K == 1 else (K + 1 + K * (K - 1))
                caps = [max(1, min(DUR_CAP, int(round(f / 60.0)))) for _x, f in pm]; L = len({x for x, _f in pm}) + 1
                row["n_states_needed"] = 1 + L * math.prod(c + 1 for c in caps)
        else:
            row["K"] = None
    return row


def main():
    t0 = time.perf_counter()
    rep = json.load(open(os.path.join(R, "e2e_report_v11.json"))); weights = rep["weights"]; qhat = rep["modes"]["complete"]["qhat"]
    arch = {r["rid"]: r for r in json.load(open(os.path.join(R, "e2e_test_rows_v11_complete.json")))}
    v6 = load_v6()
    test = [e for e in load_split("test") if classify_target(hid(e["rule_id"])) == IN_DSL]
    print(f"grammar-reachable test units {len(test)}; v6 selections for {sum(1 for e in test if e['rule_id'] in v6)}; q-hat {qhat}", flush=True)
    server = V11Server(); rows = []
    for i, e in enumerate(test):
        rid = e["rule_id"]; base = unit_row(e, server, weights, qhat, None)
        a = arch.get(rid, {})
        assert base["label"] == a.get("label"), (rid, base["label"], a.get("label"))
        if base["label"] == IN_DSL:
            assert base["n_above"] == a["n_above"] and base["pool"] == a["pool"], (rid, base["n_above"], a["n_above"], base["pool"], a["pool"])
        uni = unit_row(e, server, weights, qhat, v6.get(rid, []))
        rows.append(dict(rid=rid, repo=e["repo"], faithful=bool(faithful_g0(hid(rid))), n_v6=len(v6.get(rid, [])),
                         n_v6_new=len(set(v6.get(rid, [])) - set(server.catalog_for(e, vis(rid)).license([], []))), v11=base, union=uni))
        if i % 40 == 0:
            print(f"  {i}/{len(test)} {rid} v11 {base['label']} metrics {base['n_metrics_licensed']} maximal {base['maximal_size']} | union {uni['label']} metrics {uni['n_metrics_licensed']} maximal {uni['maximal_size']} ({time.perf_counter()-t0:.0f}s)", flush=True)

    def med(xs):
        xs = [x for x in xs if x is not None]; return float(statistics.median(xs)) if xs else None

    def agg(sub):
        out = {}
        for arm in ("v11", "union"):
            rs = [r[arm] for r in sub]; gen = [r for r in rs if r["label"] == IN_DSL]; ret = [r for r in gen if r["retained_gold"]]
            out[arm] = dict(n=len(rs), licensed=len(gen), recall=len(gen) / len(rs) if rs else None, retained=len(ret),
                            retention_given_generated=(len(ret) / len(gen) if gen else None), e2e=(len(ret) / len(rs) if rs else None),
                            metrics_licensed_median=med([r["n_metrics_licensed"] for r in rs]), metrics_licensed_mean=(sum(r["n_metrics_licensed"] for r in rs) / len(rs) if rs else None),
                            pool_median=med([r["pool"] for r in rs]), set_size_median=med([r["set_size"] for r in rs]), maximal_median=med([r["maximal_size"] for r in rs]),
                            retained_metrics_median=med([r["n_retained_metrics"] for r in rs]),
                            K_values={str(k): sum(1 for r in ret if r.get("K") == k) for k in (1, 2, 3, 4)}, K_none=sum(1 for r in ret if r.get("K") is None),
                            decide_lps_total=sum(r.get("decide_lps", 0) for r in ret), states_needed_max=max((r.get("n_states_needed", 0) for r in ret), default=None),
                            share_same_metric_median=med([r.get("share_same_metric") for r in ret]))
        pairs = [(r["v11"], r["union"]) for r in sub]
        out["growth"] = dict(pool_ratio_median=med([b["pool"] / a["pool"] for a, b in pairs if a["pool"]]),
                             set_ratio_median=med([b["set_size"] / a["set_size"] for a, b in pairs if a["set_size"]]),
                             maximal_ratio_median=med([b["maximal_size"] / a["maximal_size"] for a, b in pairs if a["maximal_size"]]),
                             retained_metrics_ratio_median=med([b["n_retained_metrics"] / a["n_retained_metrics"] for a, b in pairs if a["n_retained_metrics"]]),
                             metric_bits_v11=med([math.log2(a["n_retained_metrics"]) for a, _ in pairs if a["n_retained_metrics"] > 0]),
                             metric_bits_union=med([math.log2(b["n_retained_metrics"]) for _, b in pairs if b["n_retained_metrics"] > 0]),
                             newly_licensed=sum(1 for a, b in pairs if a["label"] != IN_DSL and b["label"] == IN_DSL),
                             newly_retained=sum(1 for a, b in pairs if not a["retained_gold"] and b["retained_gold"]),
                             lost_retained=sum(1 for a, b in pairs if a["retained_gold"] and not b["retained_gold"]))
        return out

    repos = sorted({r["repo"] for r in rows})
    res = dict(registration="V50 (REGISTRATION_V28.md)", qhat=qhat, n=len(rows), all=agg(rows), faithful=agg([r for r in rows if r["faithful"]]),
               per_repo={rp: agg([r for r in rows if r["repo"] == rp]) for rp in repos}, rows=rows, seconds=round(time.perf_counter() - t0, 1))
    json.dump(res, open(OUT, "w"), indent=1)
    for lab in ("all", "faithful"):
        a = res[lab]; print(f"== {lab}: n={a['v11']['n']}")
        for arm in ("v11", "union"):
            x = a[arm]; print(f"   {arm:6s} recall {x['recall']:.3f} retained {x['retained']} ret|gen {x['retention_given_generated']} e2e {x['e2e']:.3f} metrics med {x['metrics_licensed_median']} pool med {x['pool_median']:.3e} set med {x['set_size_median']:.3e} maximal med {x['maximal_median']} retained-metrics med {x['retained_metrics_median']} K {x['K_values']} Knone {x['K_none']} LPs {x['decide_lps_total']} states_max {x['states_needed_max']}")
        print("   growth", json.dumps(a["growth"]))
    for rp in repos:
        a = res["per_repo"][rp]; print(f"   {rp:30s} n {a['v11']['n']:3d} recall v11 {a['v11']['recall']:.3f} union {a['union']['recall']:.3f} | maximal med {a['v11']['maximal_median']} -> {a['union']['maximal_median']} | newly licensed {a['growth']['newly_licensed']} newly retained {a['growth']['newly_retained']} lost {a['growth']['lost_retained']}")
    print(f"wrote {OUT} in {res['seconds']} s")


if __name__ == "__main__":
    main()
