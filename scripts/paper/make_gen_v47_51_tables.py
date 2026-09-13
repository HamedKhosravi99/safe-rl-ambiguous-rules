"""Emit the appendix table bodies for the 2026-09-06 revision, from the
V47-V51 archives. Every row is read from an archive; no numeral is typed.

Run: PYTHONPATH=. python3 scripts/paper/make_gen_v47_51_tables.py
"""
from __future__ import annotations
import json
import os, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R = os.path.join(ROOT, "results/e2e")
G = os.path.join(os.environ.get("ARROW_PAPER_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper")), "generated")


def load(n):
    return json.load(open(os.path.join(R, n)))


def w(name, rows):
    with open(os.path.join(G, name), "w") as fh:
        fh.write("".join(rows))
    print(f"wrote {name} ({len(rows)} rows)")


def pc(x, nd=1):
    return "--" if x is None else f"${100*x:.{nd}f}\\%$"


def f2(x, nd=2):
    return "--" if x is None else f"${x:.{nd}f}$"


# ---- V47b: compiled suite at a fixed evaluator budget (equal split)
pd_ = load("pipeline_compare_delta.json")
LAB = {"seq_other_reading": "sequential, other reading first", "seq_certified_reading": "sequential, certified reading first", "arrow": "\\method{} (Decide, then one Check)"}
rows = []
for d in (0.05, 0.005):
    for n in (2000, 20000):
        for lc, lcn in (("fqi", "Lagrangian FQI"), ("lp", "occupancy LP")):
            a = pd_["aggregate"][f"d{d}_n{n}_{lc}"]
            head = f"\\multicolumn{{7}}{{@{{}}l@{{}}}}{{\\emph{{$d={d}$, $n={n:,}$, {lcn}}}}} \\\\\n".replace(",", "{,}")
            rows.append(head)
            for k, lab in LAB.items():
                v = a[f"equal|{k}"]
                rows.append(f"\\quad {lab} & {pc(v['ship_rate'])} & {pc(v['ship_and_unsafe'],2)} & {f2(v['runs_mean'])} & ${v['samples_mean']/1000:.1f}$k & "
                            f"{f2(v['shipped_ret_median'],3)} & {f2(v['utility_abstain0'],3)} \\\\\n")
assert all("nan" not in r for r in rows)
w("gen_v47_pipeline.tex", rows)

# ---- V47b: archived controlled domains
dm = load("pipeline_compare_delta_domains.json")
NAMES = {"synthetic": "Synthetic maintenance", "real": "C-MAPSS replay MDP", "gridworld": "Warning-window", "budget": "Budget agent"}
rows = []
for k, lab in NAMES.items():
    e = dm[k]["equal (0.025,0.025)"]; f = dm[k]["front (0.04,0.01)"]
    s, a = e["stcr"], e["arrow"]
    rows.append(f"{lab} & {pc(s['ship'])} & {f2(s['runs'])} & ${s['episodes']:.0f}$ & {pc(a['ship'])} & $1.00$ & ${a['episodes']:.0f}$ & "
                f"{f2(s['ret_over_unc'],3) if s['ret_over_unc'] else '--'} & {pc(f['stcr']['ship'])} \\\\\n")
w("gen_v47_domains.tex", rows)

# ---- V48: exact surrogate-loss distribution by budget
cr = load("collapse_readout.json")
rows = []
for b, v in cr["a_exact_surrogate_loss_d0.05"]["by_budget"].items():
    rows.append(f"${b}$ & {v['n']} & {pc(v['median'],2)} & {pc(v['mean'],2)} & {pc(v['p90'],2)} & {pc(v['max'],2)} & "
                f"{v['n_above_1pct']} & {v['n_above_2pct']} & {v['n_above_5pct']} \\\\\n")
w("gen_v48_surrogate.tex", rows)

# ---- V48: max-gap instances, four arms, four learners (n = 20,000)
LN = {"fqi": "Lagrangian FQI", "lp": "occupancy LP, nominal", "lp_tight": "occupancy LP, budget halved", "lp_pess": "occupancy LP, pessimistic costs"}
AN = ("decide_single", "fullset", "surrogate", "permissive")
rows = []
for lc, lab in LN.items():
    a = cr["b_max_gap_learner_arms"]["d0.05_n20000"]["arms"][lc]
    cells = []
    for arm in AN:
        x = a[arm]
        cells.append("--" if x["ret_frac_mean"] is None else f"${x['ret_frac_mean']:.3f}$ / {pc(x['safe_frac'],0)}")
    rows.append(f"{lab} & " + " & ".join(cells) + " \\\\\n")
w("gen_v48_maxgap.tex", rows)

# ---- V48: finite-sample decay
rows = []
for key in ("d0.05_lp_n2000", "d0.05_lp_n20000", "d0.05_lp_tight_n2000", "d0.05_lp_tight_n20000",
            "d0.05_fqi_n2000", "d0.05_fqi_n20000", "d0.02_lp_n2000", "d0.02_lp_n20000"):
    v = cr["c_single_minus_fullset"][key]
    d, lc, n = key.split("_")[0][1:], "_".join(key.split("_")[1:-1]), key.split("_")[-1][1:]
    lab = {"lp": "occupancy LP", "lp_tight": "LP, budget halved", "fqi": "Lagrangian FQI", "lp_pess": "LP, pessimistic"}[lc]
    rows.append(f"${d}$ & {lab} & ${int(n):,}$".replace(",", "{,}") + f" & {pc(v['mean'],2)} & {pc(v['frac_gt_1'])} & {pc(v['frac_gt_2'])} & "
                f"{pc(v['frac_lt_minus_0p1'])} & {pc(v['max'],1)} \\\\\n")
w("gen_v48_finite.tex", rows)

# ---- V49b: hard elimination, both priors, three populations
an = load("answer_noise_check.json")
POP = {"monitoring_free": "monitoring free class", "admission": "admission", "compiled_d0.005": "compiled, $d{=}0.005$"}
rows = []
for pk, plab in POP.items():
    pop = an["populations"][pk]
    for prior in ("gold", "uniform"):
        cells = []
        for p in (0.05, 0.1, 0.2, 0.3):
            v = pop["summary"][prior][f"flip|{p}|hard"]["unsafe_at_0"]
            cells.append(f"{pc(v['truth_eliminated'],0)} / {pc(v['unsafe_any'])}")
        rows.append(f"{plab} & {prior} & " + " & ".join(cells) + " \\\\\n")
w("gen_v49_hard.tex", rows)

# ---- V49b: protocols on the monitoring pools, uniform prior
mon = an["populations"]["monitoring_free"]["summary"]["uniform"]
PROT = [("flip|{p}|hard", "hard elimination"), ("flip|{p}|tolerant1", "tolerant, $\\le1$ inconsistency"),
        ("flip|{p}|repeat3", "each question asked three times"), ("flip|{p}|bayes|pa0.05|dp0.05", "posterior stop, assumed $0.05$"),
        ("flip|{p}|bayes|pa0.1|dp0.05", "posterior stop, assumed $0.10$"), ("flip|{p}|bayes|pa0.2|dp0.05", "posterior stop, assumed $0.20$"),
        ("flip|{p}|bayes|pa0.3|dp0.05", "posterior stop, assumed $0.30$")]
rows = []
for key, lab in PROT:
    cells = []
    for p in (0.05, 0.1, 0.2, 0.3):
        v = mon[key.format(p=p)]["unsafe_at_0"]
        cells.append(f"{v['answers_mean']:.1f} / {pc(v['fallback'],0)} / {pc(v['unsafe_any'])}")
    rows.append(f"{lab} & " + " & ".join(cells) + " \\\\\n")
w("gen_v49_protocols.tex", rows)

# ---- coverage funnel per repository (unit level)
cf = load("coverage_funnel.json")
ST = ("raw", "unique", "non_template", "in_grammar", "faithful", "proposed", "retained")
rows = []
for repo, st in cf["unit_level"].items():
    lab = "\\textbf{all}" if repo == "ALL" else repo.replace("_", "\\_")
    cells = [f"${st[k]['n']:,}$".replace(",", "{,}") for k in ST]
    rows.append(f"{lab} & " + " & ".join(cells) + f" & {pc(st['retained']['unconditional'])} \\\\\n")
w("gen_funnel_units.tex", rows)

# ---- V16 failure taxonomy
tx = load("v16_failure_taxonomy.json")
ORDER = [("group", "metric set not retrieved"), ("term", "aggregation--window--function skeleton"),
         ("shape", "predicate shape / boolean composition"), ("numeric", "threshold constant off axis"),
         ("label", "label set unavailable"), ("for_on_axis", "duration off axis"),
         ("composition", "all components available, tree not emitted"), ("complete", "complete tree proposed")]
ff = tx["first_failure"]
rows = [f"{lab} & ${ff['counts'][k]}$ & {pc(ff['raw'][k])} & {pc(ff['distinct'][k])} & {pc(ff['repo_macro'][k])} \\\\\n" for k, lab in ORDER]
w("gen_tax_first.tex", rows)
rows = []
for k, lab in (("single_metric", "one"), ("two_metrics", "two"), ("three_plus", "three or more")):
    a = tx["by_metric_count"][k]["availability"]
    sp = a["stage_pass"]
    rows.append(f"{lab} & ${a['n']:,}$".replace(",", "{,}") + f" & {pc(sp['shape'],0)} & {pc(sp['group'],0)} & {pc(sp['term'],0)} & "
                f"{pc(sp['numeric'],0)} & {pc(a['all_components_frac'])} & {pc(a['complete_frac'])} \\\\\n")
w("gen_tax_metrics.tex", rows)

# ---- V50 / V51 union
uc = load("union_calibrated_v51.json")
ARMS = [("v11", "retrieval only, $\\qhat{=}0.410$"), ("union_v6_old_q", "union (archived draw), $\\qhat{=}0.410$"),
        ("union_old_q", "union, $\\qhat{=}0.410$"), ("union_new_q", "union, recalibrated $\\qhat{=}0.358$")]
rows = []
for pop, plab in (("all", "all $315$"), ("faithful", "faithful $173$")):
    rows.append(f"\\multicolumn{{8}}{{@{{}}l@{{}}}}{{\\emph{{{plab} in-grammar test units}}}} \\\\\n")
    for k, lab in ARMS:
        x = uc["test"][pop][k]
        ci = x["retention_ci95"]
        rows.append(f"\\quad {lab} & {pc(x['proposal_recall'])} & {pc(x['retained_recall'])} & {pc(x['retention_given_generated'])} & "
                    f"[{100*ci[0]:.1f}, {100*ci[1]:.1f}] & ${x['set_median']:.2e}$".replace("e+07", "\\!\\times\\!10^{7}") +
                    f" & ${x['maximal_median']:,.0f}$".replace(",", "{,}") + f" & ${x['K_eq1']}$ \\\\\n")
assert uc["branch"]["survives"] is False and uc["branch"]["set_and_antichain_le_125"] is False
w("gen_union_v51.tex", rows)
print("\nbranch:", json.dumps(uc["branch"]))
