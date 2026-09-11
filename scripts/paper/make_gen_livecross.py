"""Emit generated/gen_livecross.tex from results/e2e/live_crossing_replay.json.

Asserts every claim the crossed-pool replay sentences state, so the build
fails if a re-run moves a number out from under its sentence.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "results/e2e", "live_crossing_replay.json")
KEEP = os.path.join(ROOT, "results/e2e", "live_crossing_keep.json")
STAB = os.path.join(ROOT, "results/e2e", "live_crossing_keep_stability.json")
OUT = os.path.join(ROOT, "paper", "generated", "gen_livecross.tex")


def main() -> None:
    with open(SRC, encoding="utf8") as fh:
        a = json.load(fh)
    cr, lg = a["crossed_pool"], a["logged"]
    safe = cr["some_optimum_set_safe"]

    # --- claims the body states --------------------------------------------
    assert not a["nested_pool"]["fires"], "nested pool fires"
    assert not cr["fires"], \
        "the crossed pool now fires; Section 5.1 says the screen clears " \
        "existentially and must be rewritten"
    assert safe["total3"] and sum(safe.values()) == 1, \
        "value-sufficiency no longer unique to the tightest total cap"
    assert lg["corset_k3"]["violates"]["run2"] and \
        not any(v for k, v in lg["corset_k3"]["violates"].items() if k != "run2"), \
        "the logged k=3 schedule's violation profile moved"
    sv = sum(lg["single_k5"]["violates"].values())
    n_read = len(cr["singles"])
    assert abs(cr["V_U"] - cr["singles"]["total3"]) < 1e-9, \
        "V_U no longer ties the tightest cap's value"

    # --- Keep closure: run caps scored through the frozen ensemble ---------
    with open(KEEP, encoding="utf8") as fh:
        keep = json.load(fh)
    rc = keep["runcap_per_candidate"]
    qhat = keep["qhat"]
    assert keep["runcap_retained"] == ["psi_run>=3"], \
        "the ensemble no longer retains exactly run<=3; Section 5.1 states " \
        "that outcome and must be rewritten"
    assert rc["psi_run>=2"]["mean"] < qhat <= rc["psi_run>=3"]["mean"], \
        "the score ordering around qhat moved"
    cal = a["calibrated_pool"]
    assert cal is not None and not cal["fires"], \
        "the calibrated crossed pool now fires"
    assert cal["some_optimum_set_safe"]["total3"], \
        "total<=3 lost its set-safe optimum on the calibrated pool"
    retained_names = {f"total{k}" for k in (3, 4, 5, 6, 8)} | \
        {f"run{b}" for b in a["calibrated_run_caps"]}
    guard_viol = {k for k, v in lg["corset_k3"]["violates"].items() if v}
    assert not (guard_viol & retained_names), \
        "the deployed guard now violates a CALIBRATED-retained reading; " \
        "Section 5.1 says it survives the calibrated pool"

    # --- stability of the closure: replicate panels ------------------------
    with open(STAB, encoding="utf8") as fh:
        stab = json.load(fh)
    sc = stab["per_candidate"]
    n_rep = stab["n_panels"]
    assert sc["psi_run>=2"]["retained_in"] == 0, \
        "run<=2 retained in a replicate panel; the guard-safety stability " \
        "sentence is false"
    # totals INCLUDING the registered panel, which the paper counts
    tot_panels = n_rep + 1
    r3_tot = sc["psi_run>=3"]["retained_in"] + 1   # registered retained it
    r4_tot = sc["psi_run>=4"]["retained_in"] + 0   # registered rejected it
    assert rc["psi_run>=3"]["retained"] and not rc["psi_run>=4"]["retained"]
    # Section 5.1 says EVERY panel's calibrated pool clears the screen and
    # the guard survives it.  That needs total<=3 to imply every burst
    # reading any panel retained.  Only run<=3 / run<=4 were ever retained
    # (asserted above); check the implication by enumeration, not argument.
    def max_run(bits):
        best = run = 0
        for b in bits:
            run = run + 1 if b else 0
            best = max(best, run)
        return best
    for B in (3, 4):
        for x in range(4096):
            bits = [(x >> t) & 1 for t in range(12)]
            if sum(bits) <= 3:
                assert max_run(bits) <= B, \
                    f"total<=3 no longer implies run<={B}"

    # --- exact face accounting for the cleaned build (Section 5.1 / Table 22) ---
    paid, free = a["step_mean_paid"], a["step_mean_free"]
    W = len(paid)

    def _val(bits):
        return sum(p if b else f for b, p, f in zip(bits, paid, free))

    def _maxrun(bits):
        best = run = 0
        for b in bits:
            run = run + 1 if b else 0
            best = max(best, run)
        return best
    feas3 = [tuple((x >> t) & 1 for t in range(W)) for x in range(2 ** W)]
    feas3 = [s for s in feas3 if sum(s) <= 3]
    best3 = max(_val(s) for s in feas3)
    face3 = [s for s in feas3 if abs(_val(s) - best3) < 1e-9]
    assert len(face3) == 1 and "".join(map(str, face3[0])) == cr["argmax_U"], \
        "the tightest cap's optimal face is no longer the single archived schedule; Section 5.1 must change"
    assert _maxrun(face3[0]) <= 2 and sum(face3[0]) <= 3, "the face schedule is no longer set-safe"
    dep = tuple(int(c) for c in lg["corset_k3"]["pattern"])
    # the archived per-step means are rounded to four decimals, so sums agree to ~1e-4
    assert abs(_val(dep) - lg["corset_k3"]["mean_value"]) < 1e-3
    assert abs(best3 - cr["V_U"]) < 1e-3
    assert best3 - _val(dep) > 0.05, "the deployed guard's schedule is now (near-)optimal; the 'not on the face' sentence must change"
    vals3 = sorted({round(_val(s), 9) for s in feas3}, reverse=True)
    dep_rank = 1 + next(i for i, v in enumerate(vals3) if abs(v - _val(dep)) < 1e-9)
    assert sc["psi_run>=2"]["retained_in"] == 0 and not rc["psi_run>=2"]["retained"], \
        "run<=2 is retained somewhere; the 'in no calibrated set' sentence is false"
    m_extra = [
        f"\\newcommand{{\\lcVUfour}}{{{cr['V_U']:.4f}}}",
        f"\\newcommand{{\\lcCorsetValFour}}{{{lg['corset_k3']['mean_value']:.4f}}}",
        f"\\newcommand{{\\lcSingleValFour}}{{{lg['single_k5']['mean_value']:.4f}}}",
        f"\\newcommand{{\\lcFaceSize}}{{{len(face3)}}}",
        f"\\newcommand{{\\lcDeployedRank}}{{{dep_rank}}}",
        f"\\newcommand{{\\lcDeployedClassN}}{{{len(vals3)}}}",
        f"\\newcommand{{\\lcDeployedGap}}{{{cr['V_U'] - lg['corset_k3']['mean_value']:.4f}}}",
    ]
    rows_cleaned = None
    m = [
        "% AUTO-GENERATED by paper/final/make_gen_livecross.py -- do not edit",
        f"\\newcommand{{\\lcEpisodes}}{{{a['n_episodes']}}}",
        f"\\newcommand{{\\lcPoolN}}{{{n_read}}}",
        f"\\newcommand{{\\lcVU}}{{{cr['V_U']:.2f}}}",
        f"\\newcommand{{\\lcArgmax}}{{\\texttt{{{cr['argmax_U']}}}}}",
        f"\\newcommand{{\\lcSingleViolN}}{{{sv}}}",
        f"\\newcommand{{\\lcCorsetVal}}{{{lg['corset_k3']['mean_value']:.2f}}}",
        f"\\newcommand{{\\lcSingleVal}}{{{lg['single_k5']['mean_value']:.2f}}}",
        f"\\newcommand{{\\lcQhat}}{{{qhat:g}}}",
        f"\\newcommand{{\\lcRunScoreTwo}}{{{rc['psi_run>=2']['mean']:.2f}}}",
        f"\\newcommand{{\\lcRunScoreThree}}{{{rc['psi_run>=3']['mean']:.2f}}}",
        f"\\newcommand{{\\lcRunScoreFour}}{{{rc['psi_run>=4']['mean']:.2f}}}",
        f"\\newcommand{{\\lcCalPoolN}}{{{len(cal['singles'])}}}",
        f"\\newcommand{{\\lcPanelsTot}}{{{tot_panels}}}",
        f"\\newcommand{{\\lcRThreeTot}}{{{r3_tot}}}",
        f"\\newcommand{{\\lcRFourTot}}{{{r4_tot}}}",
        f"\\newcommand{{\\lcStabRTwoLo}}{{{sc['psi_run>=2']['lo']:.2f}}}",
        f"\\newcommand{{\\lcStabRTwoHi}}{{{sc['psi_run>=2']['hi']:.2f}}}",
    ]
    rows = []
    for key, label in (("single_k5", "Single reading ($k{=}5$), as run"),
                       ("corset_k3", "Set guard ($k{=}3$), as run")):
        viol = [r for r, b in lg[key]["violates"].items() if b]
        pretty = ", ".join(r.replace("total", "$k{\\le}")
                            .replace("run", "$B{\\le}") + "$" for r in viol)
        rows.append(f"{label} & \\texttt{{{lg[key]['pattern']}}} & "
                    f"${lg[key]['mean_value']:.2f}$ & {pretty} \\\\")
    rows.append(f"Set-safe tied optimum (replayed) & \\texttt{{{cr['argmax_U']}}}"
                f" & ${cr['V_U']:.2f}$ & none \\\\")
    rows_cleaned = rows[:-1] + [f"Optimum of $k{{\\le}}3$ (replayed), set-safe & \\texttt{{{cr['argmax_U']}}}"
                                f" & ${cr['V_U']:.2f}$ & none \\\\"]
    m += m_extra
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf8") as fh:
        fh.write("\n".join(m) + "\n")
    with open(OUT.replace(".tex", "_rows_cleaned.tex"), "w", encoding="utf8") as fh:
        fh.write("% AUTO-GENERATED by paper/final/make_gen_livecross.py"
                 " -- do not edit\n" + "\n".join(rows_cleaned) + "\n")
    rows_out = OUT.replace(".tex", "_rows.tex")
    with open(rows_out, "w", encoding="utf8") as fh:
        fh.write("% AUTO-GENERATED by paper/final/make_gen_livecross.py"
                 " -- do not edit\n" + "\n".join(rows) + "\n")
    print("\n".join(m))


if __name__ == "__main__":
    main()
