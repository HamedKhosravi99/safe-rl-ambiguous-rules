#!/usr/bin/env python3
"""Macros and table rows for the practical-applicability additions: V25 adaptive certificate (50k), V24 exposure ceiling,
V26 oracle clarification study, ARTEMIS Recall@K, and the margin law of the finite-data run. Every number asserted."""
import json, math, collections, re, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)) + "/"
TE = os.path.join(ROOT, "results", "theory_extension") + "/"
GEN = os.path.join(ROOT, "paper", "generated") + "/"
out, tables = [], {}
def macro(n, v): out.append(f"\\newcommand{{\\{n}}}{{{v}}}")
def pct(x): return f"{100*x:.0f}\\%"
def thousands(n): return f"{int(round(n)):,}".replace(",", "{,}")
# ---- V25: 50k-episode gates ----
e = json.load(open(ROOT + "results/conformal/lp/evaluator_scale_50k.json"))
S = e["summary"]; N = int(e["n_episodes"])
assert N == 50000, N
cells = [("real/corset", "C-MAPSS, set"), ("real/single", "C-MAPSS, single"),
         ("synthetic/corset", "Synthetic, set"), ("synthetic/single", "Synthetic, single")]
rows = []
for key, label in cells:
    c = S[key]
    assert c["reproduces_archive"] == c["n_seeds"] == 50
    med = c.get("first_crossing_median")
    rows.append(f"{label} & {c['true_mean_below_d']}/{c['n_seeds']} & {c['cp_ships']} & {c['hoeffding_ships']} & {c['bernstein_ships']} & {c['betting_ships']} & {thousands(med) if med else '--'} \\\\\n")
tables["gen_ek50k_table.tex"] = "".join(rows)
rc, sc = S["real/corset"], S["synthetic/corset"]
assert rc["betting_ships"] == 43 and rc["true_mean_below_d"] == 49 and rc["first_crossing_median"] == 16543 and sc["first_crossing_median"] == 476
macro("EkN", thousands(N)); macro("EkCmapBet", rc["betting_ships"]); macro("EkCmapBelow", rc["true_mean_below_d"]); macro("EkCmapStop", thousands(rc["first_crossing_median"]))
macro("EkCmapCP", rc["cp_ships"]); macro("EkCmapBern", rc["bernstein_ships"]); macro("EkSynStop", thousands(sc["first_crossing_median"])); macro("EkSynBet", f"{sc['betting_ships']}/{sc['true_mean_below_d']}")
# ---- V24: exposure ceiling (C-MAPSS = the domain with the archived B of 53.3) ----
x = json.load(open(ROOT + "results/conformal/lp/exposure_ceiling.json"))
dom_key = max(x["domains"], key=lambda k: x["domains"][k]["B_archived"]); dom = x["domains"][dom_key]
assert abs(dom["B_archived"] - 53.33) < 0.1, (dom_key, dom["B_archived"])
mid = next(v for k, v in dom["per_reading"].items() if k.startswith("mid"))
assert mid["ceiling_short_k0"] and abs(mid["Z_max"] - 0.745) < 0.01 and abs(mid["Z_need_k0"] - 1.955) < 0.01, mid
macro("XcZgreedy", f"{mid['Z_greedy']:.2f}"); macro("XcZmax", f"{mid['Z_max']:.2f}"); macro("XcZneed", f"{mid['Z_need_k0']:.2f}"); macro("XcBarch", f"{dom['B_archived']:.1f}")
bceil = dom.get("B_max_over_readings", {}).get("ceiling") if isinstance(dom.get("B_max_over_readings"), dict) else dom.get("B_max_over_readings")
if bceil is None: bceil = min(v["B_max"] for v in dom["per_reading"].values() if v.get("maximal")) if any(v.get("maximal") for v in dom["per_reading"].values()) else min(v["B_max"] for v in dom["per_reading"].values())
macro("XcBceil", f"{bceil:.1f}")
# ---- V26: oracle clarification ----
q = json.load(open(ROOT + "results/e2e/query_loop.json"))
assert q["branch"]["populations_where_poa_beats_split"] == 3 and q["operating_budget"] == 0.05
names = {"compiled": "Compiled monitoring", "admission": "Admission", "monitoring_free": "Monitoring, free class"}
rule_names = {"poa": "price-guided", "split": "most balanced split", "random": "random fixture"}
rows = []
for pop in ("compiled", "admission", "monitoring_free"):
    sm = q["populations"][pop]["summary"]
    for r in ("poa", "split", "random"):
        rec = sm["rules"][r]["recovered_mean"]; qs = sm["rules"][r]["q"]; assert qs == [0, 1, 2, 4]
        rows.append(f"{names[pop]} ({sm['n']}) & {rule_names[r]} & {pct(rec[1])} & {pct(rec[2])} & {pct(rec[3])} \\\\\n")
tables["gen_query_table.tex"] = "".join(rows)
cp, mf = q["populations"]["compiled"]["summary"]["rules"], q["populations"]["monitoring_free"]["summary"]["rules"]
assert abs(cp["poa"]["recovered_mean"][1] - 0.894) < 0.005 and abs(mf["poa"]["recovered_mean"][1] - 0.645) < 0.005
macro("QlBudget", "0.05"); macro("QlCompRec", pct(cp["poa"]["recovered_mean"][1])); macro("QlMonRec", pct(mf["poa"]["recovered_mean"][1]))
macro("QlCompSplit", pct(cp["split"]["recovered_mean"][1])); macro("QlMonSplit", pct(mf["split"]["recovered_mean"][1]))
macro("QlAdmRec", pct(q["populations"]["admission"]["summary"]["rules"]["poa"]["recovered_mean"][1]))
macro("QlCompN", cp and q["populations"]["compiled"]["summary"]["n"]); macro("QlMonN", q["populations"]["monitoring_free"]["summary"]["n"]); macro("QlAdmN", q["populations"]["admission"]["summary"]["n"])
# ---- ARTEMIS Recall@K ----
a = json.load(open(ROOT + "results/e2e/artemis_units.json"))
Ks = [1, 2, 5, 10, 20, None]
res = {}
for pool in ("P1", "P2"):
    per_k_cls, per_k_any = collections.defaultdict(list), collections.defaultdict(list)
    for u in a["units"]:
        trials = [t for t in u["trials"] if t["valid"] and (t["in_p1"] if pool == "P1" else True)]
        if not trials: continue
        first, cnt = {}, collections.Counter()
        for i, t in enumerate(trials):
            cnt[t["cls"]] += 1; first.setdefault(t["cls"], i)
        ranked = sorted(cnt, key=lambda c: (-cnt[c], first[c]))
        nlab = u["label_classes"]
        for K in Ks:
            top = ranked if K is None else ranked[:K]
            matched = set()
            for c in top: matched.update(u["cand_match"].get(str(c), []))
            per_k_cls[K].append(len(matched) / nlab if nlab else 0.0); per_k_any[K].append(1.0 if matched else 0.0)
    res[pool] = {K: (float(np.mean(per_k_cls[K])), float(np.mean(per_k_any[K])), len(per_k_cls[K])) for K in Ks}
# denominator: all 182 units (misses count as zero), the convention of tab:artemis; validate against the printed pool recall
gen_ar = open(GEN + "gen_artemis.tex").read()
printed = float(re.search(r"\\newcommand\{\\ArTwoPoolRecall\}\{([0-9.]+)\\%\}", gen_ar).group(1))
assert abs(100 * res["P2"][None][0] - printed) < 0.15, (res["P2"][None][0], printed)
assert res["P2"][None][2] == 182, res["P2"][None][2]
rows = []
for K in Ks:
    lab = "all" if K is None else str(K)
    rows.append(f"{lab} & {pct(res['P1'][K][0])} & {pct(res['P1'][K][1])} & {pct(res['P2'][K][0])} & {pct(res['P2'][K][1])} \\\\\n")
tables["gen_artemis_recall_table.tex"] = "".join(rows)
macro("ArRecOne", pct(res["P2"][1][0])); macro("ArRecTen", pct(res["P2"][10][0])); macro("ArRecAll", pct(res["P2"][None][0]))
# generation term of the deployment theorem on the union pool: any-plausible coverage by candidate budget K, the number of
# generation misses among the 182 units, and a one-sided 95% Clopper-Pearson upper confidence bound on the miss probability
from scipy.stats import beta as _beta
_m = res["P2"][None][2]; _g = int(round(_m * (1 - res["P2"][None][1])))
assert _m == 182 and abs(_m * (1 - res["P2"][None][1]) - _g) < 1e-6, (_m, res["P2"][None][1])
assert res["P2"][1][1] < res["P2"][10][1] < res["P2"][None][1], "coverage must rise with the candidate budget; text must change"
macro("ArKOneAny", pct(res["P2"][1][1])); macro("ArKTenAny", pct(res["P2"][10][1])); macro("ArGenMisses", _g)
macro("ArDgenUcb", f"{100 * _beta.ppf(0.95, _g + 1, _m - _g):.1f}\\%")
macro("ArAnyOne", pct(res["P2"][1][1])); macro("ArAnyTen", pct(res["P2"][10][1])); macro("ArAnyAll", pct(res["P2"][None][1]))
macro("ArPtwoN", res["P2"][None][2]); macro("ArPoneN", res["P1"][None][2])
# ---- margin law from the finite-data run ----
fd = json.load(open(TE + "real_rules_finite_data.json"))
pts = {}
for key in ("d=0.02|bernstein", "d=0.05|bernstein", "d=0.1|bernstein"):
    for v in fd[key]["n_star"].values(): pts[(key, round(v["margin"], 4))] = v["n_star"]
g = np.array([m for (_, m) in pts]); n = np.array(list(pts.values()))
slope = np.polyfit(np.log(1 / g ** 2), np.log(n), 1)[0]; r = np.corrcoef(np.log(1 / g ** 2), np.log(n))[0, 1]
assert 0.95 < slope < 1.25 and r > 0.98, (slope, r)
rec = json.load(open(TE + "real_rules_finite_data_records.json"))
suff = collections.defaultdict(lambda: collections.defaultdict(list))
for xrow in rec:
    if xrow["truth"] and xrow["bound"] == "bernstein": suff[(xrow["d"], xrow["uid"], xrow["k"])][xrow["n"]].append(xrow["pass"])
sizes = sorted(set(xrow["n"] for xrow in rec))
nstar = {k: min([s for s in sizes if v.get(s) and np.mean(v[s]) >= 0.9], default=None) for k, v in suff.items()}
cdf = lambda thr: sum(1 for v in nstar.values() if v is not None and v <= thr * 1.0001) / len(nstar)
assert abs(cdf(10 ** 7.5) - 0.57) < 0.02 and abs(cdf(1e9) - 0.94) < 0.02, (cdf(10 ** 7.5), cdf(1e9))
macro("FdSlopeInv", f"{slope:.2f}"); macro("FdCorr", f"{r:.3f}"); macro("FdCdfA", pct(cdf(10 ** 7.5))); macro("FdCdfB", pct(cdf(1e9))); macro("FdCdfC", pct(cdf(1e7)))
for fn, body in tables.items(): open(GEN + fn, "w").write(body)
open(GEN + "gen_applic.tex", "w").write("% generated by make_gen_applic.py -- do not edit\n" + "\n".join(out) + "\n")
print(f"gen_applic: {len(out)} macros; tables: {list(tables)}")
print("ARTEMIS P2 recall@K:", {("all" if K is None else K): (round(res['P2'][K][0], 3), round(res['P2'][K][1], 3)) for K in Ks})
print("50k rows:\n" + tables["gen_ek50k_table.tex"]); print("query rows:\n" + tables["gen_query_table.tex"])
