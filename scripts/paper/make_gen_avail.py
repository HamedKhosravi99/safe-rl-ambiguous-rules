#!/usr/bin/env python3
"""Emit generated/gen_avail.tex -- the candidate-availability and
candidate-generation macros used in Section 5.4 of the ARROW 9-page body.

Reads only archived run files and raises rather than emitting a placeholder
if a field is missing, so a build cannot contain an unresolved number.  Same
contract as make_unified_numbers.py, which emits the same generation rates as
proportions (\\AccGenVfour etc.); this script emits the percentage forms the
body quotes, from the same archives, so the two cannot drift apart.

Run: python3 scripts/paper/make_gen_avail.py
"""
from __future__ import annotations

import json
import math
import os

from scipy.stats import beta as _beta


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact binomial interval; the conventional choice for a coverage rate."""
    lo = 0.0 if k == 0 else float(_beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "paper", "generated", "gen_avail.tex")


def load(*parts: str):
    path = os.path.join(REPO, *parts)
    with open(path) as fh:
        return json.load(fh)


def dig(obj, path: str):
    for key in path.split("/"):
        obj = obj[key]
    return obj


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}\\%"


def main() -> None:
    # Availability: where the candidate resource is read from.  recall_v4 is
    # the catalog harvested from the repositories' own NON-rule files (docs,
    # dashboards, source metric definitions); every alert and recording-rule
    # file is excluded, so the harvest is target-blind by construction.
    # recall_v3_loco is the leave-one-cluster-out catalog built from sibling
    # rule files, measured under the same membership convention -- the
    # comparison that isolates *where* the resource is read from.
    av = load("results/e2e", "v4_recall.json")
    for key in ("n_units", "recall_v3_loco", "recall_v4", "recall_union"):
        if key not in av:
            raise KeyError(f"v4_recall.json is missing {key}")

    # Generation: the same rates make_unified_numbers.py emits as proportions.
    e0 = load("results/conformal", "e0", "e0_report.json")
    t4 = dig(load("results/e2e", "e2e_report_v4.json"), "modes/complete/test")
    v6 = load("results/e2e", "v6_calibrated.json")
    blind = dig(e0, "generator_secondary/recall")
    gen = dig(t4, "rho_gen/point")
    ret = dig(t4, "rho_ret_given_gen/point")
    lic = dig(v6, "test/rho_e2e")

    # The v4 pair is the only pair that composes: same population, same
    # frozen scorer and generator.  The v6 licenser is a separate campaign
    # with a different generator, so it is reported, never multiplied in.
    if t4["n_grammar_reachable"] != v6["test"]["n"]:
        raise AssertionError("v4 and v6 no longer share the expressible-unit "
                             "population; Section 5.4 says they do")

    # Section 5.3 says the confidence event, the optimizer-vs-fallback event
    # and true retained-set feasibility ALL hold in the same fraction of
    # draws, and that the comparator-margin premise never fires at the
    # OPERATING budget.  Check both budgets' archives per instance and size,
    # so neither sentence can outlive the fact.  feas_opt (not just the
    # fallback-substituted feas_sub) is checked: the claim is true
    # feasibility, not feasibility credited to the fallback.
    fires = set()
    for archive, budget in (("certified_at_scale.json", 0.01),
                            ("certified_at_scale_d05.json", 0.05)):
        run = load("results/conformal", "lp", archive)
        if run["d"] != budget:
            raise AssertionError(f"{archive} is no longer at d={budget}")
        for inst in run["instances"]:
            for size, cell in inst["sizes"].items():
                for event in ("conf", "opt", "feas_opt", "feas_sub"):
                    if cell[event] != 1.0:
                        raise AssertionError(
                            f"{archive}: {inst['instance']} at N={size} has "
                            f"{event}={cell[event]}; Section 5.3 states one "
                            "rate for conf/opt/feas")
                if budget == 0.05:
                    fires.add(cell["margin"] * cell["n"])
    if fires != {0.0}:
        raise AssertionError("the comparator-margin premise now fires at the "
                             f"operating budget (draws {sorted(fires)}); "
                             "Section 5.3 says it never does")
    margin_fires = 0

    # Frozen-benchmark coverage.  coverage_in_dsl is the conditional quantity
    # Lemma 1 guarantees and the one comparable to the nominal level (q-hat is
    # calibrated in-DSL-conditionally); coverage is the marginal rate, whose
    # denominator adds the out-of-DSL rows.  Emitted as bare numbers so the
    # body can keep them inside its own math.
    bench = load("results", "corpus_eval", "test_report_openai.json")
    nominal = 1.0 - bench["metadata"]["delta_sem"]
    cov_in = dig(bench, "test/coverage_in_dsl/rate")
    cov_all = dig(bench, "test/coverage/rate")
    cov_swap = dig(bench, "transfer_heldout_generator/coverage_in_dsl/rate")
    cov_dom = dig(bench, "transfer_heldout_domain_battery/coverage_in_dsl/rate")
    # Section 5.4 says the swapped-generator POINT ESTIMATE sits below nominal;
    # if a rerun ever lifts it, the sentence must change with it.
    if not cov_swap < nominal <= cov_in:
        raise AssertionError(f"benchmark coverage moved (in-DSL {cov_in}, "
                             f"swapped {cov_swap}, nominal {nominal}); "
                             "Section 5.4 states their order")
    # The point estimate is not the whole story and the paper should not have
    # reported it alone: coverage_in_dsl carries an n, so the exact binomial
    # interval is available and decides whether the shortfall is real.  At
    # n=225 it is not -- the interval covers nominal -- so Section 5.4 states
    # the interval rather than an unqualified "falls below".
    n_in = int(dig(bench, "test/coverage_in_dsl/n"))
    n_swap = int(dig(bench, "transfer_heldout_generator/coverage_in_dsl/n"))
    swap_lo, swap_hi = clopper_pearson(round(cov_swap * n_swap), n_swap)
    in_lo, in_hi = clopper_pearson(round(cov_in * n_in), n_in)
    # Guard the two directions Section 5.4 now asserts.
    if not swap_lo <= nominal <= swap_hi:
        raise AssertionError(
            f"the swapped-generator interval [{swap_lo:.4f},{swap_hi:.4f}] no "
            f"longer covers nominal {nominal}; Section 5.4 says the shortfall "
            "is within sampling error and must be rewritten")
    if not in_lo > nominal:
        raise AssertionError(
            f"the in-DSL interval [{in_lo:.4f},{in_hi:.4f}] no longer sits "
            f"strictly above nominal {nominal}; Section 5.4 claims it does")

    # Whole-repository holdout: a separate corpus, not this benchmark and not
    # the extension.
    v9all = load("results/e2e", "v9_independent_corpus.json")
    v9 = dig(v9all, "repo_splits/d=0.1")
    if v9["clears"]:
        raise AssertionError("the repository holdout now clears nominal; "
                             "Section 5.4 reports it as a shortfall")
    # The B=4000 draws in the archive are NOT 4000 independent replications:
    # each draw calibrates on floor(R/2) repositories and tests on the rest,
    # so the mean summarizes at most C(R, floor(R/2)) distinct pairings and
    # the draws only reweight them.  Section 5.4 must not quote the mean as
    # though it carried 4000 draws' worth of precision, so emit the two
    # quantities that bound its resolution.
    v9repos = v9all["repos"]
    v9pairs = math.comb(len(v9repos), len(v9repos) // 2)
    if v9pairs > v9all["B"]:
        raise AssertionError("pairing count exceeds the draw budget; the "
                             "resolution caveat in Section 5.4 assumes the "
                             "draws resample a smaller set of pairings")
    lines = [
        "% AUTO-GENERATED by paper/final/make_gen_avail.py -- do not edit\n",
        "\\newcommand{\\CertAllEvents}{100\\%}\n",
        f"\\newcommand{{\\MarginFiresD}}{{{margin_fires}}}\n",
        f"\\newcommand{{\\CovInDsl}}{{{100 * cov_in:.1f}}}\n",
        f"\\newcommand{{\\CovNominal}}{{{100 * nominal:.0f}}}\n",
        f"\\newcommand{{\\CovSwap}}{{{100 * cov_swap:.1f}}}\n",
        f"\\newcommand{{\\CovSwapN}}{{{n_swap}}}\n",
        f"\\newcommand{{\\CovSwapLo}}{{{100 * swap_lo:.1f}}}\n",
        f"\\newcommand{{\\CovSwapHi}}{{{100 * swap_hi:.1f}}}\n",
        f"\\newcommand{{\\CovInDslN}}{{{n_in}}}\n",
        f"\\newcommand{{\\CovInDslLo}}{{{100 * in_lo:.1f}}}\n",
        f"\\newcommand{{\\CovInDslHi}}{{{100 * in_hi:.1f}}}\n",
        f"\\newcommand{{\\CovDomain}}{{{100 * cov_dom:.1f}}}\n",
        f"\\newcommand{{\\CovOverall}}{{{cov_all:.3f}}}\n",
        f"\\newcommand{{\\CovRepoHoldout}}{{{v9['mean_cov']:.3f}}}\n",
        f"\\newcommand{{\\CovRepoNominal}}{{{v9['nominal']:.2f}}}\n",
        f"\\newcommand{{\\CovRepoRepos}}{{{len(v9repos)}}}\n",
        f"\\newcommand{{\\CovRepoN}}{{{v9all['n']}}}\n",
        f"\\newcommand{{\\CovRepoPairs}}{{{v9pairs}}}\n",
        f"\\newcommand{{\\AvailN}}{{{av['n_units']}}}\n",
        f"\\newcommand{{\\AvailSibling}}{{{pct(av['recall_v3_loco'])}}}\n",
        f"\\newcommand{{\\AvailRepo}}{{{pct(av['recall_v4'])}}}\n",
        f"\\newcommand{{\\AvailUnion}}{{{pct(av['recall_union'])}}}\n",
        f"\\newcommand{{\\GenBlindPct}}{{{pct(blind)}}}\n",
        f"\\newcommand{{\\AccGenPct}}{{{pct(gen)}}}\n",
        f"\\newcommand{{\\AccRetPct}}{{{pct(ret)}}}\n",
        f"\\newcommand{{\\AccComposedPct}}{{{pct(gen * ret)}}}\n",
        f"\\newcommand{{\\AccLicPct}}{{{pct(lic)}}}\n",
    ]
    with open(OUT, "w") as fh:
        fh.write("".join(lines))
    print("wrote", OUT)
    print("".join(lines[1:]), end="")


if __name__ == "__main__":
    main()
