"""Judge-perturbation stability of U_alpha (reviewer additions #5 and #6).

The construction replaced recruited human annotators with a frozen 7-persona LM
ensemble precisely to fix the original protocol's fragility (a single annotator
flip changed the retained set ~67% of the time -- see saorl.construct). Reviewers
ask the natural follow-up: how stable is U_alpha to perturbing the JUDGES? This
sweeps two learning-free, cache-only perturbations of the frozen plausibility
tuples (no LLM is re-queried -- the 7 persona scores are already cached, so this is
an exact recomputation, not a new measurement):

  * leave-one-judge-out (LOJO): drop persona j from every candidate's score tuple
    and rebuild U_alpha. Reports Jaccard(U^{-j}, U^full) -- does removing any one
    judge change the retained set? A robust ensemble keeps Jaccard ~ 1.
  * single-judge: keep ONLY persona j and rebuild U_alpha. Reports the same
    Jaccard against the full-ensemble set. This is the degenerate "one annotator"
    regime the ensemble is meant to improve on; we expect markedly lower, more
    variable Jaccard -- the empirical motivation for the ensemble.

It also reports an ENSEMBLE-INTERNAL agreement summary (explicitly NOT human IAA):
for each candidate, the per-persona plausibility vote (score >= TAU_PLAUS) versus
the ensemble-mean decision, i.e. how often a lone judge agrees with the panel.

Honesty note: this validates U_alpha's STABILITY to judge perturbation and the
panel's internal agreement; it is NOT a human precision/recall study (no gold
labels are invoked) and does NOT probe judge position-bias (which needs
re-querying the LM with permuted option order -- flagged separately). Those remain
the genuinely human-/LM-budget-bound parts of reviewer #5.

Run:  PYTHONPATH=. python3 -m saorl.ablate_judges --seeds 20
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .construct import TAU_PLAUS, construct_U_alpha
from .experiments import _synthetic_seed
from .offline import semantic_gap


def _set_plaus(c, scores):
    """Return a copy of candidate c with its plausibility tuple replaced."""
    return dataclasses.replace(c, plausibility=tuple(scores))


def _names(U) -> set:
    return {c.name for c in U}


def _jaccard_names(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass
class JudgeRow:
    mode: str          # "full" | "lojo" | "single"
    judge: int         # dropped (lojo) or kept (single) persona index; -1 for full
    seed: int
    u_size: int
    jaccard_full: float
    sem_gap: float


def _construct_names_gap(pool, data, tau=TAU_PLAUS):
    dsem = data.to_semantic_dataset()
    U = construct_U_alpha(pool, dsem, tau_plaus=tau).U_alpha
    return _names(U), len(U), float(semantic_gap(data, U, normalize="active"))


def collect(seeds: Sequence[int]) -> List[JudgeRow]:
    rows: List[JudgeRow] = []
    for seed in seeds:
        data, _, _, _, pool = _synthetic_seed(seed)
        n_judges = len(getattr(pool[0], "plausibility", ()) or ())
        # full-ensemble reference
        full_names, full_u, full_gap = _construct_names_gap(pool, data)
        rows.append(JudgeRow("full", -1, seed, full_u, 1.0, full_gap))
        for j in range(n_judges):
            lojo_pool = [_set_plaus(c, c.plausibility[:j] + c.plausibility[j + 1:])
                         for c in pool]
            names, u, gap = _construct_names_gap(lojo_pool, data)
            rows.append(JudgeRow("lojo", j, seed, u, _jaccard_names(names, full_names), gap))

            single_pool = [_set_plaus(c, (c.plausibility[j],)) for c in pool]
            names, u, gap = _construct_names_gap(single_pool, data)
            rows.append(JudgeRow("single", j, seed, u, _jaccard_names(names, full_names), gap))
        print(f"  seed={seed}: full |U|={full_u} gap={full_gap:.3f}; swept {n_judges} judges x2")
    return rows


def _agreement(seeds: Sequence[int], tau: float = TAU_PLAUS) -> dict:
    """Ensemble-internal agreement (NOT human IAA): per candidate, fraction of
    personas whose vote (score>=tau) matches the ensemble-mean decision, plus how
    often a lone judge agrees with the panel, averaged over candidates."""
    data, _, _, _, pool = _synthetic_seed(seeds[0])  # pool/scores are seed-invariant
    per_cand = []
    judge_agree = None
    for c in pool:
        p = np.asarray(c.plausibility, float)
        ens_decision = (p.mean() >= tau)
        votes = (p >= tau)
        frac_match = float(np.mean(votes == ens_decision))
        per_cand.append(dict(name=c.name, plaus_mean=round(float(p.mean()), 3),
                             ens_retain_gate=bool(ens_decision),
                             frac_personas_agree=round(frac_match, 3)))
        ja = (votes == ens_decision).astype(float)
        judge_agree = ja if judge_agree is None else judge_agree + ja
    judge_agree = (judge_agree / len(pool)).round(3).tolist()
    return dict(per_candidate=per_cand,
                mean_persona_agreement=round(float(np.mean([d["frac_personas_agree"]
                                                            for d in per_cand])), 3),
                per_judge_agreement_with_panel=judge_agree)


def _ci95(xs):
    a = np.asarray(xs, float)
    n = len(a)
    if n == 0:
        return 0.0, 0.0
    m = float(a.mean())
    if n < 2:
        return m, 0.0
    from scipy import stats
    return m, float(stats.t.ppf(0.975, n - 1) * a.std(ddof=1) / np.sqrt(n))


def summarize(rows: List[JudgeRow]) -> dict:
    out: dict = {}
    for mode in ("lojo", "single"):
        sub = [r for r in rows if r.mode == mode]
        jm, jh = _ci95([r.jaccard_full for r in sub])
        gm, gh = _ci95([r.sem_gap for r in sub])
        um, uh = _ci95([r.u_size for r in sub])
        worst_j = min((r.jaccard_full for r in sub), default=1.0)
        frac_identical = float(np.mean([r.jaccard_full == 1.0 for r in sub])) if sub else 0.0
        out[mode] = dict(
            jaccard=(jm, jh), worst_jaccard=worst_j, frac_identical_to_full=round(frac_identical, 3),
            sem_gap=(gm, gh), u_size=(um, uh), n=len(sub))
        print(f"\n  === {mode} === ({len(sub)} judge-perturbations)")
        print(f"    Jaccard(U_perturbed, U_full) = {jm:.3f}+/-{jh:.3f}  "
              f"(worst {worst_j:.3f}; identical-to-full {frac_identical:.0%})")
        print(f"    |U|={um:.2f}+/-{uh:.2f}   semantic_gap={gm:.3f}+/-{gh:.3f}")
    full = [r for r in rows if r.mode == "full"]
    fg, fgh = _ci95([r.sem_gap for r in full])
    fu, fuh = _ci95([r.u_size for r in full])
    out["full"] = dict(sem_gap=(fg, fgh), u_size=(fu, fuh), n=len(full))
    print(f"\n  === full ensemble (reference) ===  |U|={fu:.2f}+/-{fuh:.2f}  "
          f"semantic_gap={fg:.3f}+/-{fgh:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser(description="judge-perturbation stability of U_alpha (#5/#6)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-list", type=str, default=None)
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()
    seeds = ([int(s) for s in args.seed_list.split(",") if s.strip()]
             if args.seed_list else list(range(args.seeds)))
    print(f"judge-perturbation ablation: seeds={seeds} (synthetic maintenance)")
    t0 = time.time()
    rows = collect(seeds)
    summary = summarize(rows)
    agree = _agreement(seeds)
    print(f"\n  ensemble-internal agreement (NOT human IAA): "
          f"mean persona-vs-panel agreement = {agree['mean_persona_agreement']:.0%}")
    for d in agree["per_candidate"]:
        print(f"    {d['name']:18s} plaus_mean={d['plaus_mean']:.2f} "
              f"gate={'retain' if d['ens_retain_gate'] else 'reject'} "
              f"personas-agree={d['frac_personas_agree']:.0%}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"ablate_judges_{stamp}.json"
    out_path.write_text(json.dumps(dict(
        config=dict(seeds=seeds, tau=TAU_PLAUS),
        rows=[dataclasses.asdict(r) for r in rows],
        summary=summary, agreement=agree,
    ), indent=2))
    print(f"\n  wrote {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
