"""W15/W16 (blueprint sections 17-18, run plan 30.10): query library, prior
sensitivity, and ask-vs-act decision rules -- all recomputed from the frozen
price report (results/conformal/price/price_report.json).  NO retraining: every
value below is read from the per-seed singleton oracles (v_psi), the robust
value (v_robust), and the plausibility posterior already serialized there.

Query library (per domain with |U| >= 2): one singleton-revealing query per
member psi -- answer partition {{psi}, U \\ {psi}} -- plus the full-reveal
query whose |U| answer cells are the singletons.  For |U| = 2 all of these
induce the same partition (noted in the output).

Cell values V_{U_j}: a singleton cell is the per-seed oracle v_psi; the cell
U_j = U is v_robust.  A strict multi-member cell (the 'rest' cell of a
singleton-reveal query when |U| = 3) has NO trained robust value in the
existing data.  We use only the sandwich identity readable from the report:
v_robust <= V_{U_j} <= min_{psi in U_j} v_psi (adding constraints cannot raise
the optimum), so whenever min_{psi in U_j} v_psi == v_robust on a seed the cell
value is pinned exactly to v_robust.  Seeds where the sandwich does not close
are marked n/a and excluded -- nothing is fabricated or retrained.

Priors over answers: uniform; plausibility-normalized (the deployed posterior,
exactly the price report's); worst-case-over-simplex, for which
VoQ_robust = min_j V_{U_j} - V_U (all prior mass on the least valuable cell).

Decision rules, evaluated on the FULL-REVEAL query (always exactly computable)
at query costs kappa in {0.1, 1, 5, 20} return units: always-ask, never-ask,
set-size trigger (ask iff |U| > 1), expected-VoQ trigger (ask iff plausibility
VoQ > kappa), robust-VoQ trigger (ask iff robust VoQ > kappa).  Realized net
value of a rule = (E_answer[V_answer] - kappa) if it asks, else v_robust, with
the answer drawn from (a) the plausibility posterior as ground truth and
(b) a uniform ground truth (sensitivity).

Cross-check: the plausibility-prior full-reveal VoQ must reproduce the price
report's `voq` per domain to float precision (same numbers, same formula).

Run:  PYTHONPATH=. python3 -m saorl.query_library
Writes results/conformal/query_library.json
   and paper/generated/gen_query.tex
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).parent.parent.parent
PRICE_REPORT = _REPO / "results/conformal" / "price" / "price_report.json"

KAPPAS = (0.1, 1.0, 5.0, 20.0)
PRIORS = ("uniform", "plausibility", "worst_case")
RULES = ("always", "never", "set_size", "exp_voq", "rob_voq",
         "entropy", "score_gap")
RULE_LABEL = {"always": "always-ask", "never": "never-ask",
              "set_size": "set-size", "exp_voq": "exp-VoQ",
              "rob_voq": "rob-VoQ", "entropy": "entropy",
              "score_gap": "score-gap"}

# Two posterior-*shape* ask triggers (blueprint 30.10), value-blind by design:
# both look only at the plausibility posterior over the retained readings, never
# at the readings' values, so they cannot see the price of a wrong commitment.
#   ENTROPY:   ask iff the normalized posterior entropy H(post)/log|U| exceeds
#              ENTROPY_TAU -- the posterior carries more than half its maximum
#              possible entropy, i.e. it sits closer to uniform (maximally
#              ambiguous) than to a confident point mass.  0.5 is the natural
#              "more uncertain than not" midpoint and needs no tuning.
#   SCORE-GAP: ask iff the gap between the top-two posterior scores is below
#              GAP_TAU -- no reading wins by a clear margin.  0.2 is a standard
#              "decisive lead" cutoff (a >20-point posterior lead is a confident
#              call).
# Contrast the price triggers (exp_voq / rob_voq), which ask iff the value of
# resolving the ambiguity, VoQ, exceeds the query cost kappa -- the correct
# signal (Theorem thm:voq).
ENTROPY_TAU = 0.5
GAP_TAU = 0.2
DOMAIN_LABEL = {"synthetic": r"Maint.\ (synthetic)",
                "real": r"Maint.\ (C-MAPSS)",
                "gridworld": r"Warning-window",
                "budget": r"Budget agent"}
_TOL = 1e-9


# --------------------------------------------------------------------------
# Cell values from the frozen per-seed report rows
# --------------------------------------------------------------------------

def _cell_value(cell: Sequence[str], v_psi: Dict[str, float],
                v_robust: float, n_total: int) -> Optional[float]:
    """Exact cell value readable from existing data, else None (n/a)."""
    if len(cell) == 1:
        return v_psi[cell[0]]
    if len(cell) == n_total:
        return v_robust
    m = min(v_psi[n] for n in cell)
    if abs(m - v_robust) <= _TOL:      # sandwich closes: v_robust <= V <= m
        return v_robust
    return None


def _queries(names: List[str]) -> List[dict]:
    qs = []
    for n in names:
        rest = [m for m in names if m != n]
        qs.append(dict(id=f"reveal[{n}]", cells=[[n], rest]))
    qs.append(dict(id="full-reveal", cells=[[n] for n in names]))
    return qs


def _voq(cells: List[List[str]], values: List[float], prior_cells: List[float],
         v_robust: float) -> float:
    return float(sum(p * v for p, v in zip(prior_cells, values)) - v_robust)


def _shape_triggers(post: Dict[str, float], names: List[str]) -> Tuple[float, float]:
    """(normalized posterior entropy H/log|U|, top-2 posterior score gap) over
    the retained readings -- the two value-blind ask signals.  Both read only
    the plausibility posterior committed in the price report."""
    n = len(names)
    p = np.array([post[m] for m in names], dtype=float)
    s = float(p.sum())
    if s > 0:
        p = p / s
    nz = p[p > 0]
    H = float(-np.sum(nz * np.log(nz))) if nz.size else 0.0
    norm_H = H / float(np.log(n)) if n > 1 else 0.0
    sp = np.sort(p)[::-1]
    gap = float(sp[0] - sp[1]) if n >= 2 else 1.0
    return norm_H, gap


def _prior_over_cells(prior: str, cells: List[List[str]],
                      post: Dict[str, float]) -> Optional[List[float]]:
    n = sum(len(c) for c in cells)
    if prior == "uniform":
        return [len(c) / n for c in cells]
    if prior == "plausibility":
        return [sum(post[m] for m in c) for c in cells]
    return None  # worst_case handled analytically (min over cells)


# --------------------------------------------------------------------------
# Per-domain analysis
# --------------------------------------------------------------------------

def _analyze_domain(rows: List[dict]) -> dict:
    names = rows[0]["u_names"]
    n = len(names)
    queries = _queries(names)

    # per-query, per-prior VoQ over the seeds where every cell value is exact
    qout = []
    for q in queries:
        per_prior: Dict[str, dict] = {}
        na_seeds: List[int] = []
        per_seed_vals: List[Tuple[dict, List[float]]] = []
        for r in rows:
            v_psi = {p["psi"]: p["v_psi"] for p in r["per_psi"]}
            vals = [_cell_value(c, v_psi, r["v_robust"], n) for c in q["cells"]]
            if any(v is None for v in vals):
                na_seeds.append(r["seed"])
                continue
            per_seed_vals.append((r, vals))
        for prior in PRIORS:
            samples = []
            for r, vals in per_seed_vals:
                post = {p["psi"]: p["posterior"] for p in r["per_psi"]}
                if prior == "worst_case":
                    samples.append(float(min(vals) - r["v_robust"]))
                else:
                    pc = _prior_over_cells(prior, q["cells"], post)
                    samples.append(_voq(q["cells"], vals, pc, r["v_robust"]))
            if samples:
                per_prior[prior] = dict(
                    mean=round(float(np.mean(samples)), 4),
                    sd=round(float(np.std(samples)), 4),
                    n_seeds=len(samples))
            else:
                per_prior[prior] = dict(
                    mean=None, sd=None, n_seeds=0,
                    na_reason="no seed pins the multi-member cell value "
                              "(sandwich open); would need a robust retrain")
        qout.append(dict(
            id=q["id"], cells=q["cells"],
            n_seeds_available=len(per_seed_vals),
            na_seeds=na_seeds,
            voq=per_prior,
        ))

    # decision rules on the full-reveal query (always exact)
    decisions: Dict[str, dict] = {}
    for kappa in KAPPAS:
        per_gt: Dict[str, dict] = {}
        for gt in ("plausibility", "uniform"):
            nets = {rule: [] for rule in RULES}
            for r in rows:
                v_psi = {p["psi"]: p["v_psi"] for p in r["per_psi"]}
                post = {p["psi"]: p["posterior"] for p in r["per_psi"]}
                vr = r["v_robust"]
                voq_plaus = float(sum(post[m] * v_psi[m] for m in names) - vr)
                voq_rob = float(min(v_psi.values()) - vr)
                g = (post if gt == "plausibility"
                     else {m: 1.0 / n for m in names})
                post_answer = float(sum(g[m] * v_psi[m] for m in names))
                norm_H, top2_gap = _shape_triggers(post, names)
                ask = {"always": True, "never": False, "set_size": n > 1,
                       "exp_voq": voq_plaus > kappa, "rob_voq": voq_rob > kappa,
                       "entropy": norm_H > ENTROPY_TAU,
                       "score_gap": top2_gap < GAP_TAU}
                for rule in RULES:
                    nets[rule].append(post_answer - kappa if ask[rule] else vr)
            means = {rule: round(float(np.mean(v)), 4) for rule, v in nets.items()}
            order = sorted(RULES, key=lambda x: -means[x])
            groups: List[List[str]] = []
            for rule in order:
                if groups and abs(means[groups[-1][0]] - means[rule]) <= 1e-9:
                    groups[-1].append(rule)
                else:
                    groups.append([rule])
            ranking = " > ".join("=".join(RULE_LABEL[x] for x in grp)
                                 for grp in groups)
            per_gt[gt] = dict(net=means, best=groups[0], ranking=ranking)
        decisions[str(kappa)] = per_gt

    # per-seed plausibility VoQ (for the price-report cross-check)
    voq_seeds = [float(sum(p["posterior"] * p["v_psi"] for p in r["per_psi"])
                       - r["v_robust"]) for r in rows]
    voq_mean = float(np.mean(voq_seeds))

    # ---- ask-vs-act trigger audit (blueprint 30.10): the two value-blind
    # shape triggers (entropy, score-gap) vs the ground-truth "should ask"
    # (VoQ > kappa, the price signal).  A mis-fire is an OVER-ASK (trigger asks
    # where VoQ <= kappa) or an UNDER-ASK (trigger acts where VoQ > kappa). ----
    nH_seeds, gap_seeds = [], []
    for r in rows:
        post = {p["psi"]: p["posterior"] for p in r["per_psi"]}
        nH, gap = _shape_triggers(post, names)
        nH_seeds.append(nH)
        gap_seeds.append(gap)
    norm_H = float(np.mean(nH_seeds))
    top2_gap = float(np.mean(gap_seeds))
    entropy_ask = norm_H > ENTROPY_TAU        # seed-invariant posterior here
    gap_ask = top2_gap < GAP_TAU
    per_kappa = {}
    for kappa in KAPPAS:
        gt_ask = voq_mean > kappa
        per_kappa[str(kappa)] = dict(
            voq=round(voq_mean, 4),
            ground_truth_ask=bool(gt_ask),
            entropy_ask=bool(entropy_ask),
            score_gap_ask=bool(gap_ask),
            entropy_misfire=("over_ask" if entropy_ask and not gt_ask
                             else "under_ask" if gt_ask and not entropy_ask
                             else "correct"),
            score_gap_misfire=("over_ask" if gap_ask and not gt_ask
                               else "under_ask" if gt_ask and not gap_ask
                               else "correct"),
        )
    triggers = dict(
        norm_entropy=round(norm_H, 4),
        entropy_nats=round(float(np.mean(nH_seeds)) * float(np.log(n)), 4),
        top2_gap=round(top2_gap, 4),
        entropy_tau=ENTROPY_TAU, gap_tau=GAP_TAU,
        entropy_ask=bool(entropy_ask), score_gap_ask=bool(gap_ask),
        voq_plaus_mean=round(voq_mean, 4),
        per_kappa=per_kappa,
        note="posterior entropy and score-gap are functions of the plausibility "
             "posterior only (value-blind); ground truth is VoQ > kappa",
    )

    return dict(
        u_names=names, size=n, n_seeds=len(rows),
        partition_note=("all singleton-reveal queries coincide with "
                        "full-reveal (|U|=2: one nontrivial partition)"
                        if n == 2 else None),
        queries=qout,
        decision_rules=decisions,
        ask_vs_act_triggers=triggers,
        voq_plaus_mean=round(voq_mean, 6),
    )


def run() -> dict:
    rep = json.load(open(PRICE_REPORT))
    out: Dict[str, dict] = {}
    crosscheck: Dict[str, dict] = {}
    for dom, dd in rep["domains"].items():
        rows = dd["rows"]
        if len(rows[0]["u_names"]) < 2:
            continue
        res = _analyze_domain(rows)
        out[dom] = res
        want = float(dd["agg"]["voq"])
        got = res["voq_plaus_mean"]
        crosscheck[dom] = dict(price_report_voq=round(want, 6),
                               full_reveal_plaus_voq=got,
                               match=bool(abs(want - got) < 1e-6))
    assert all(c["match"] for c in crosscheck.values()), crosscheck

    # ask-vs-act mis-fire tally across all (domain, kappa) cells
    tally = {"entropy": {"over_ask": 0, "under_ask": 0, "correct": 0},
             "score_gap": {"over_ask": 0, "under_ask": 0, "correct": 0}}
    misfire_cells = {"entropy": [], "score_gap": []}
    for dom, res in out.items():
        for kappa, d in res["ask_vs_act_triggers"]["per_kappa"].items():
            for rule in ("entropy", "score_gap"):
                verdict = d[f"{rule}_misfire"]
                tally[rule][verdict] += 1
                if verdict != "correct":
                    misfire_cells[rule].append(f"{dom}@kappa={kappa}({verdict})")
    n_cells = len(KAPPAS) * len(out)
    trigger_audit = dict(
        thresholds=dict(entropy_tau=ENTROPY_TAU, gap_tau=GAP_TAU,
                        entropy_rule="ask iff norm posterior entropy H/log|U| "
                                     f"> {ENTROPY_TAU}",
                        score_gap_rule=f"ask iff top-2 posterior gap < {GAP_TAU}"),
        ground_truth="ask iff plausibility VoQ > kappa (the price signal)",
        n_cells=n_cells,
        tally=tally,
        misfire_cells=misfire_cells,
        note="entropy and score-gap read only the (near-uniform, "
             "seed-invariant) plausibility posterior; both collapse to "
             "always-ask on this benchmark and over-ask exactly where the "
             "price trigger correctly abstains.",
    )

    return dict(
        source=str(PRICE_REPORT),
        seeds=rep["config"]["seeds"],
        kappas=list(KAPPAS),
        note=("All values read from the frozen price report; multi-member "
              "cells are pinned only via the sandwich identity "
              "v_robust <= V_cell <= min singleton (see module docstring); "
              "unpinned seeds are excluded, never imputed."),
        domains=out,
        crosscheck_voq_vs_price_report=crosscheck,
        ask_vs_act_trigger_audit=trigger_audit,
    )


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

def _fmt(v: dict) -> str:
    if v["n_seeds"] == 0:
        return "n/a"
    s = f"${v['mean']:.2f}\\pm{v['sd']:.2f}$"
    return s


def _write_tex(report: dict, path: Path) -> None:
    lines = ["% AUTO-GENERATED by saorl/query_library.py -- do not edit",
             "% Block 1: full-reveal VoQ under three priors "
             "(mean +/- sd over seeds)",
             "% domain & |U| & VoQ uniform & VoQ plausibility & VoQ worst-case"]
    for dom, res in report["domains"].items():
        fr = next(q for q in res["queries"] if q["id"] == "full-reveal")
        lines.append(
            f"{DOMAIN_LABEL[dom]} & {res['size']} & "
            f"{_fmt(fr['voq']['uniform'])} & {_fmt(fr['voq']['plausibility'])} & "
            f"{_fmt(fr['voq']['worst_case'])} \\\\")
    lines += ["% Block 2: decision rules at query cost kappa "
              "(plausibility ground truth; full-reveal query)",
              "% ranking now also ranks the two value-blind shape triggers of "
              "blueprint 30.10:",
              "%   entropy   = ask iff norm posterior entropy H/log|U| > "
              f"{ENTROPY_TAU}",
              "%   score-gap = ask iff top-2 posterior gap < "
              f"{GAP_TAU}",
              "% both collapse to always-ask here (near-uniform posteriors) and "
              "over-ask exactly where VoQ<=kappa; see query_library.json "
              "ask_vs_act_trigger_audit.",
              "% domain & kappa & best rule & net(best) & net(never-ask) & "
              "ranking"]
    for dom, res in report["domains"].items():
        for kappa in KAPPAS:
            d = res["decision_rules"][str(kappa)]["plausibility"]
            best = d["best"][0]
            lines.append(
                f"{DOMAIN_LABEL[dom]} & {kappa:g} & {RULE_LABEL[best]} & "
                f"{d['net'][best]:.2f} & {d['net']['never']:.2f} & "
                f"{d['ranking']} \\\\")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    report = run()
    out_json = _REPO / "results/conformal" / "query_library.json"
    json.dump(report, open(out_json, "w"), indent=1)
    out_tex = _REPO / "paper" / "generated" / "gen_query.tex"
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    _write_tex(report, out_tex)

    for dom, res in report["domains"].items():
        fr = next(q for q in res["queries"] if q["id"] == "full-reveal")
        print(f"[{dom}] |U|={res['size']}  full-reveal VoQ: "
              f"uniform={_fmt(fr['voq']['uniform'])}  "
              f"plaus={_fmt(fr['voq']['plausibility'])}  "
              f"worst-case={_fmt(fr['voq']['worst_case'])}")
        for q in res["queries"]:
            if q["id"] == "full-reveal":
                continue
            v = q["voq"]["plausibility"]
            avail = f"{q['n_seeds_available']}/{res['n_seeds']} seeds"
            print(f"    {q['id']:28s} plaus VoQ "
                  f"{_fmt(v):>16s}  ({avail})")
        for kappa in KAPPAS:
            d = res["decision_rules"][str(kappa)]
            print(f"    kappa={kappa:<5g} best(plaus GT)="
                  f"{'/'.join(RULE_LABEL[b] for b in d['plausibility']['best']):24s}"
                  f" best(unif GT)="
                  f"{'/'.join(RULE_LABEL[b] for b in d['uniform']['best'])}")
    print("cross-check full-reveal plaus VoQ vs price_report voq:",
          {d: c["match"] for d, c in
           report["crosscheck_voq_vs_price_report"].items()})

    # ---- ask-vs-act trigger audit (entropy / score-gap vs the price signal) ----
    ta = report["ask_vs_act_trigger_audit"]
    print("\nASK-VS-ACT TRIGGER AUDIT (value-blind shape triggers vs VoQ>kappa):")
    print(f"  entropy rule:   {ta['thresholds']['entropy_rule']}")
    print(f"  score-gap rule: {ta['thresholds']['score_gap_rule']}")
    for dom, res in report["domains"].items():
        t = res["ask_vs_act_triggers"]
        print(f"  [{dom}] normH={t['norm_entropy']:.3f} top2gap={t['top2_gap']:.3f} "
              f"VoQ={t['voq_plaus_mean']:.2f}  entropy_ask={t['entropy_ask']} "
              f"score_gap_ask={t['score_gap_ask']}")
        for kappa in KAPPAS:
            d = t["per_kappa"][str(kappa)]
            print(f"      kappa={kappa:<5g} GT_ask={str(d['ground_truth_ask']):5s} "
                  f"entropy={d['entropy_misfire']:9s} "
                  f"score_gap={d['score_gap_misfire']:9s}")
    print(f"  tally over {ta['n_cells']} (domain,kappa) cells: "
          f"entropy {ta['tally']['entropy']}  "
          f"score_gap {ta['tally']['score_gap']}")
    print(f"wrote {out_json}\nwrote {out_tex}")


if __name__ == "__main__":
    main()
