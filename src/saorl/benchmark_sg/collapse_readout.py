"""V48 read-outs (no new runs): (a) exact single-signal surrogate loss
distribution, (b) the max-gap instances with their learner arms, (c) the
finite-sample decide-single vs fullset effect by threshold and n.
Run: PYTHONPATH=. python3 -m saorl.benchmark_sg.collapse_readout
Writes results/e2e/collapse_readout.json
"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
R = os.path.join(ROOT, "results/e2e")


def dist(x):
    x = np.asarray(x, float)
    return dict(n=int(x.size), median=float(np.median(x)), mean=float(x.mean()), p90=float(np.percentile(x, 90)), max=float(x.max()),
                n_above_1pct=int((x > 0.01).sum()), n_above_2pct=int((x > 0.02).sum()), n_above_5pct=int((x > 0.05).sum()),
                distinct_values=sorted({round(float(v), 4) for v in x}))


def main():
    out = {}
    sp = json.load(open(os.path.join(R, "surrogate_price.json")))
    cu = json.load(open(os.path.join(R, "collapse_utility.json")))
    # (a) exact surrogate loss at the operating budget, all 28 and the 27 certified (screen clear)
    op = [r for r in sp["rows"] if abs(r["budget"] - 0.05) < 1e-9]
    out["a_exact_surrogate_loss_d0.05"] = dict(all28=dist([r["surrogate_gap_rel"] for r in op]),
                                               certified27=dist([r["surrogate_gap_rel"] for r in op if not r["screen_fires"]]),
                                               by_budget={str(b): dist([r["surrogate_gap_rel"] for r in sp["rows"] if abs(r["budget"] - b) < 1e-9]) for b in sp["budgets"]})
    ex = [e for e in cu["exact"] if e.get("certified")]
    for d in (0.05, 0.02):
        vals = [(e["tightened"]["0.0"]["V_U"] - e["tightened"]["0.0"]["V_surr"]) / e["tightened"]["0.0"]["V_U"] for e in ex if e["d"] == d]
        out[f"a_exact_surrogate_loss_from_v48_d{d}"] = dist(vals)
    # (b) max-gap instances and their learner arms
    mx = max(r["surrogate_gap_rel"] for r in op)
    top = sorted({r["rule_id"] for r in op if abs(r["surrogate_gap_rel"] - mx) < 1e-9})
    out["b_max_gap_instances"] = dict(gap=mx, rule_ids=top, V_U=[r["V_U"] for r in op if r["rule_id"] in top][:1], V_surr=[r["V_surr"] for r in op if r["rule_id"] in top][:1],
                                      V_best_single=[r["V_best_single"] for r in op if r["rule_id"] in top][:1])
    b = {}
    for d in (0.05, 0.02):
        for n in (2000, 20000):
            rows = [x for x in cu["rows"] if x["d"] == d and x["n"] == n and x["rule_id"] in top]
            if not rows:
                continue
            per = {}
            for lc in ("fqi", "lp", "lp_tight", "lp_pess"):
                per[lc] = {}
                for arm in ("decide_single", "fullset", "surrogate", "permissive"):
                    S = [x["arms"][lc][arm] for x in rows]; ok = [s for s in S if s is not None]
                    per[lc][arm] = dict(n=len(ok), ret_frac_mean=(float(np.mean([s["ret_frac"] for s in ok])) if ok else None),
                                        ret_frac_median=(float(np.median([s["ret_frac"] for s in ok])) if ok else None),
                                        safe_frac=(float(np.mean([s["safe"] for s in ok])) if ok else None),
                                        cmax_over_d_median=(float(np.median([s["cmax_over_d"] for s in ok])) if ok else None),
                                        ship_frac=(float(np.mean([s["ship"] for s in ok])) if ok else None))
            b[f"d{d}_n{n}"] = dict(n_runs=len(rows), instances=sorted({x["rule_id"] for x in rows}), arms=per)
    out["b_max_gap_learner_arms"] = b
    # (c) finite-sample effect: decide-single minus fullset by threshold
    c = {}
    for d in (0.05, 0.02):
        for lc in ("fqi", "lp", "lp_tight", "lp_pess"):
            for n in (2000, 20000):
                rows = [x for x in cu["rows"] if x["d"] == d and x["n"] == n]
                diffs = [x["arms"][lc]["decide_single"]["ret_frac"] - x["arms"][lc]["fullset"]["ret_frac"] for x in rows
                         if x["arms"][lc]["decide_single"] is not None and x["arms"][lc]["fullset"] is not None]
                if not diffs:
                    continue
                dd = np.asarray(diffs)
                c[f"d{d}_{lc}_n{n}"] = dict(n=int(dd.size), mean=float(dd.mean()), median=float(np.median(dd)),
                                            frac_gt_0=float((dd > 1e-9).mean()), frac_gt_0p1=float((dd > 0.001).mean()), frac_gt_0p5=float((dd > 0.005).mean()),
                                            frac_gt_1=float((dd > 0.01).mean()), frac_gt_2=float((dd > 0.02).mean()), frac_lt_minus_0p1=float((dd < -0.001).mean()),
                                            mean_when_positive=(float(dd[dd > 1e-9].mean()) if (dd > 1e-9).any() else 0.0), max=float(dd.max()), min=float(dd.min()))
    out["c_single_minus_fullset"] = c
    json.dump(out, open(os.path.join(R, "collapse_readout.json"), "w"), indent=1)
    print(json.dumps(out["a_exact_surrogate_loss_d0.05"], indent=1))
    for k in list(out.keys()):
        if k.startswith("a_exact_surrogate_loss_from"):
            print(k, json.dumps(out[k]))
    print("MAX-GAP:", json.dumps(out["b_max_gap_instances"]))
    for k, v in b.items():
        print("==", k, v["instances"])
        for lc, arms in v["arms"].items():
            print("  ", lc, {a: (None if s["ret_frac_mean"] is None else f"ret {s['ret_frac_mean']:.3f} safe {s['safe_frac']:.2f} ship {s['ship_frac']:.2f}") for a, s in arms.items()})
    print("FINITE-SAMPLE single-fullset:")
    for k, v in c.items():
        print(f"  {k:24s} mean {v['mean']:+.4f} >0 {v['frac_gt_0']:.3f} >0.1% {v['frac_gt_0p1']:.3f} >0.5% {v['frac_gt_0p5']:.3f} >1% {v['frac_gt_1']:.3f} >2% {v['frac_gt_2']:.3f} <-0.1% {v['frac_lt_minus_0p1']:.3f} mean|pos {v['mean_when_positive']:.4f} max {v['max']:.4f}")


if __name__ == "__main__":
    main()
