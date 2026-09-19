"""Emit generated/gen_v53.tex (macros), gen_v53_table.tex (certified counts by size and certificate, d = 0.05),
gen_v53_classes.tex (three scales per sufficient-reading class: n* of each certificate and the witness-based information
floor kl(1-delta,delta)/Ibar, Ibar the divergence of the positive-margin witness chain and an upper bound on I*) from
results/e2e/certificate_v53.json (REGISTRATION_V53).  Assertions are structural only (shapes, identical draw ids across
certificates, ranges, truth matching the exact archive); method ordering, monotonicity and the false-certificate count are
reported as outcomes, the last one loudly if nonzero."""
import json, os, math, collections
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAPER = os.environ.get("ARROW_PAPER_DIR", os.path.join(ROOT, "paper")); GEN = os.path.join(PAPER, "generated")
r = json.load(open(os.path.join(ROOT, "results", "theory_extension", "certificate_v53.json")))
exact = json.load(open(os.path.join(ROOT, "results", "theory_extension", "real_rules_exact.json")))["exact"]
CERTS = ("UNIFORM", "OCC", "DUAL"); LABEL = {"UNIFORM": "uniform (archived)", "OCC": "occupancy-weighted", "DUAL": "robust dual (\\method{})"}
meta = r["meta"]; grid = meta["n_grid"]; D = 0.05
M = []
def mac(n, v): M.append(f"\\newcommand{{\\{n}}}{{{v}}}")
def fmt_n(n):
    e = math.log10(n)
    return f"$10^{{{int(round(e))}}}$" if abs(e - round(e)) < 1e-6 else f"${10 ** (e - math.floor(e)):.1f}\\times10^{{{int(math.floor(e))}}}$"
def pct(x, k=0): return f"{100 * x:.{k}f}\\%"
def key(x): return f"{x['uid']}|{x['k']}"

# ---------------------------------------------------------------- structural checks
for x in r["primary"] + r["extra_hard"]:
    assert all(c in x and isinstance(x[c]["pass"], bool) for c in CERTS), x            # every record carries all three certificates on the same draw
    assert len(x["N"]) == 3 and all(v >= 2 for v in x["N"])
    rec = exact[x["uid"]][str(x["d"])][str(x["k"])]
    assert rec is not None and (rec["margin_eps"][str(meta["eps"])] >= -1e-9) == x["truth"], x  # truth matches the exact archive
assert len({(x["n"], x["rep"], x["d"], x["uid"], x["k"]) for x in r["primary"]}) == len(r["primary"])
cls = {d: {c: set(m) for c, m in v.items()} for d, v in r["classes"].items()}
hard = cls["0.05"][min(cls["0.05"], key=lambda s: float(s.split("=")[1]))]; easy = cls["0.05"][max(cls["0.05"], key=lambda s: float(s.split("=")[1]))]
assert len(hard) == 5 and len(easy) == 22
mac("VfHardCount", len(hard)); mac("VfEasyCount", len(easy))
n_insuff = sum(1 for x in r["primary"] if not x["truth"] and x["d"] == D and x["n"] == grid[0])
assert n_insuff == 29 * meta["reps"]

# ---------------------------------------------------------------- outcomes
false = {c: sum(x[c]["pass"] for x in r["primary"] + r["extra_hard"] if not x["truth"]) for c in CERTS}
if any(false.values()): print("!!! FALSE CERTIFICATES OBSERVED:", false)
mac("VfFalseTotal", sum(false.values())); mac("VfInsuffRecords", f"{sum(1 for x in r['primary'] if not x['truth']):,}".replace(",", "{,}"))
def rate(rows, c): return float(np.mean([x[c]["pass"] for x in rows])) if rows else float("nan")
def nstar(d, members, c, thresh=0.9):
    for n in grid:
        rows = [x for x in r["primary"] if x["d"] == d and x["n"] == n and key(x) in members]
        if rows and rate(rows, c) >= thresh: return n
    return None
# main table (d = 0.05): certified sufficient readings by size and certificate, split by class, plus false certificates
T = []
for n in grid:
    rows = [x for x in r["primary"] if x["d"] == D and x["n"] == n]
    cells = []
    for c in CERTS:
        e = sum(x[c]["pass"] for x in rows if key(x) in easy); h = sum(x[c]["pass"] for x in rows if key(x) in hard)
        tot = e + h; reps = meta["reps"]
        cells.append((f"{tot // reps}" if tot % reps == 0 else f"{tot / reps:.1f}") + " / 27")
    f = sum(x[c]["pass"] for c in CERTS for x in rows if not x["truth"])
    T.append(f"{fmt_n(n)} & " + " & ".join(cells) + f" & {f} \\\\")
# n* per class and certificate (all budgets) with the witness-based information floor
C = []
for d in ("0.02", "0.05", "0.1"):
    for cname, members in sorted(cls[d].items(), key=lambda kv: float(kv[0].split("=")[1])):
        I = r["istar"][d][cname]; kappa = float(cname.split("=")[1])
        mant, expo = f"{I['I']:.2e}".split("e"); Icell = f"${mant}\\times10^{{{int(expo)}}}$"
        ns = {c: nstar(float(d), members, c) for c in CERTS}
        ninfo = f"{I['n_info']:,.0f}".replace(",", "{,}")
        C.append(f"${float(d):.2f}$ & {len(members)} & ${kappa:.4f}$ & {Icell} & ${ninfo}$ & " + " & ".join(fmt_n(ns[c]) if ns[c] else f"$>{fmt_n(grid[-1])[1:-1]}$" for c in CERTS) + " \\\\")
# headline macros (d = 0.05)
ns_easy = {c: nstar(D, easy, c) for c in CERTS}; ns_hard = {c: nstar(D, hard, c) for c in CERTS}
mac("VfEasyNstarUniform", fmt_n(ns_easy["UNIFORM"])); mac("VfEasyNstarOcc", fmt_n(ns_easy["OCC"])); mac("VfEasyNstarDual", fmt_n(ns_easy["DUAL"]))
mac("VfHardNstarDual", fmt_n(ns_hard["DUAL"])); mac("VfHardNstarUniformArchive", "$10^{10}$")
assert ns_hard["UNIFORM"] is None and ns_hard["OCC"] is None                            # structural: they did not certify within the grid (outcome reported as >1e8)
Ih = r["istar"]["0.05"][[c for c in cls["0.05"] if cls["0.05"][c] == hard][0]]; Ie = r["istar"]["0.05"][[c for c in cls["0.05"] if cls["0.05"][c] == easy][0]]
mac("VfHardIstar", f"{Ih['I']:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}"); mac("VfHardNinfo", f"{Ih['n_info']:,.0f}".replace(",", "{,}"))
mac("VfEasyIstar", f"{Ie['I']:.2e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}"); mac("VfEasyNinfo", f"{Ie['n_info']:,.0f}".replace(",", "{,}"))
mac("VfHardKappa", f"{float([c for c in cls['0.05'] if cls['0.05'][c]==hard][0].split('=')[1]):.4f}"); mac("VfEasyKappa", f"{float([c for c in cls['0.05'] if cls['0.05'][c]==easy][0].split('=')[1]):.4f}")
Qh = np.array(Ih["Q"]); mac("VfHardFlipRow", " ".join(f"{v:.3f}" for v in Qh[1])); mac("VfHardFlipLone", f"{max(Ih['l1_rows']):.3f}")
mac("VfHardRatio", f"{ns_hard['DUAL'] / Ih['n_info']:.0f}"); mac("VfEasyRatio", f"{ns_easy['DUAL'] / Ie['n_info']:.0f}")
ratios = []
for d in ("0.02", "0.05", "0.1"):
    for cname, members in cls[d].items():
        n_d = nstar(float(d), members, "DUAL"); ratios.append(n_d / r["istar"][d][cname]["n_info"])
mac("VfRatioLo", f"{min(ratios):.0f}"); mac("VfRatioHi", f"{max(ratios):.0f}"); mac("VfClasses", len(ratios))
# generic uniform certificate against the floor, on the classes it reaches within the grid
def sci(x):
    m, e = f"{x:.1e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"
gen_ratios = []
for d in ("0.02", "0.05", "0.1"):
    for cname, members in cls[d].items():
        n_u = nstar(float(d), members, "UNIFORM")
        if n_u is not None: gen_ratios.append(n_u / r["istar"][d][cname]["n_info"])
assert gen_ratios and min(gen_ratios) > max(ratios)
mac("VfGenericRatioLo", sci(min(gen_ratios))); mac("VfGenericRatioHi", sci(max(gen_ratios))); mac("VfGenericClasses", len(gen_ratios))
# hard class over all independent draws
for n, name in ((1e5, "VfHardAtEfive"), (10 ** 5.5, "VfHardAtThreeEfive"), (1e6, "VfHardAtEsix")):
    rows = [x for x in r["primary"] + r["extra_hard"] if abs(x["n"] - n) < 1e-6 and x["d"] == D and key(x) in hard]
    reps = sorted({x["rep"] for x in rows}); per = [all(x["DUAL"]["pass"] for x in rows if x["rep"] == rep) for rep in reps]
    mac(name, pct(np.mean(per))); mac(name + "Draws", len(reps))
# the d = 0.10 ranking reversal used in the text: smaller kappa, larger I*, certified at a higher rate at 1e4 (asserted: text must change otherwise)
ten = sorted(cls["0.1"].items(), key=lambda kv: float(kv[0].split("=")[1]))[:2]           # the two smallest-kappa classes at d = 0.10
(k_small, m_small), (k_large, m_large) = ten
I_small, I_large = r["istar"]["0.1"][k_small]["I"], r["istar"]["0.1"][k_large]["I"]
r_small = rate([x for x in r["primary"] if x["d"] == 0.1 and abs(x["n"] - 1e4) < 1e-6 and key(x) in m_small], "DUAL")
r_large = rate([x for x in r["primary"] if x["d"] == 0.1 and abs(x["n"] - 1e4) < 1e-6 and key(x) in m_large], "DUAL")
assert float(k_small.split("=")[1]) < float(k_large.split("=")[1]) and I_small > I_large and r_small > r_large, (k_small, k_large, I_small, I_large, r_small, r_large)
mac("VfTenSmallKappa", f"{float(k_small.split('=')[1]):.4f}"); mac("VfTenLargeKappa", f"{float(k_large.split('=')[1]):.4f}")
mac("VfTenSmallRateEfour", pct(r_small)); mac("VfTenLargeRateEfour", pct(r_large))
mac("VfHardTrueRow", " ".join(f"{v:.3f}" for v in np.array(meta["M_true"])[1]))
# secondary (row counts from sampled load-chain paths): the text says it reproduces every conclusion; checked, not tabulated
for c in ("UNIFORM", "DUAL"): assert sum(x[c]["pass"] for x in r["secondary"] if not x["truth"]) == 0
assert all(rate([x for x in r["secondary"] if x["n"] == n and key(x) in easy], "DUAL") == 1.0 for n in meta["secondary"]["n"])
assert rate([x for x in r["secondary"] if x["n"] == max(meta["secondary"]["n"]) and key(x) in hard], "DUAL") == 1.0
mac("VfSeedsPrimary", meta["reps"]); mac("VfHardExtraDraws", meta["hard_extra"]["reps"]); mac("VfDelta", meta["delta"]); mac("VfEps", meta["eps"])
H = "% AUTO-GENERATED by paper/final/make_gen_v53.py -- do not edit\n"
open(os.path.join(GEN, "gen_v53.tex"), "w").write(H + "\n".join(M) + "\n")
open(os.path.join(GEN, "gen_v53_table.tex"), "w").write(H + "\n".join(T) + "\n")
open(os.path.join(GEN, "gen_v53_classes.tex"), "w").write(H + "\n".join(C) + "\n")
print("\n".join(M)); print("\nTABLE\n" + "\n".join(T)); print("\nCLASSES\n" + "\n".join(C))
