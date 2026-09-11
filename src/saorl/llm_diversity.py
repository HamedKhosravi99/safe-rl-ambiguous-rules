"""LLM translation diversity (plan section 8.8.3): how many distinct DSL readings
the cached language ensemble actually endorses, per rule.

The non-arbitrariness worry a reviewer raises is: ``maybe |U_alpha|>1 is a data
artifact, not genuine linguistic ambiguity.'' This analysis answers it from the
*language side alone*, before any data gate. Each of the 7 frozen ensemble
personas is treated as an independent translator that ENDORSES every candidate
reading it scores at least TAU_PLAUS faithful. We then report, per rule:

  * LLM-single   -- the single most-endorsed reading (a single-translation pick);
  * LLM-majority -- readings a strict majority (>= ceil(7/2)=4) of personas endorse;
  * LLM-union    -- readings ANY persona endorses (the diversity set);
  * top-pick set -- the distinct per-persona argmax readings;
  * |U_alpha|    -- the data-gated retained set size (mean over seeds), for context.

The headline: for a SHARP rule the ensemble converges to one endorsed reading
(LLM-majority = 1), so there is no retained ambiguity to protect; for VAGUE rules
it endorses several genuinely competing readings (LLM-majority and -union > 1),
which is exactly where SA-ORL's protection turns on. This makes |U_alpha|>1 a
property of the *language*, not of the calibration data.

Run:  PYTHONPATH=. python3 -m saorl.llm_diversity
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .build_plaus_cache import RULE_SETS
from .construct import TAU_PLAUS, construct_U_alpha
from .judge import attach_plausibility, load_llm_cache

N_SEEDS = 16


@dataclass
class DiversityRow:
    domain: str
    rule: str
    n_cand: int
    n_persona: int
    llm_single: str
    n_majority: int      # readings endorsed by a strict majority of personas
    n_union: int         # readings endorsed by >= 1 persona
    n_toppick: int       # distinct per-persona argmax readings
    disagree: float      # mean over candidates of persona-score std (spread)
    u_alpha_mean: float  # data-gated retained-set size, mean over seeds
    u_alpha_max: int


def _scores_matrix(entry: dict) -> (List[str], np.ndarray):
    """Return (candidate names, [n_cand x n_persona] score matrix) for a rule set.
    Rows with unequal persona counts are right-truncated to the common length."""
    names = list(entry["scores"].keys())
    rows = [entry["scores"][n] for n in names]
    m = min(len(r) for r in rows)
    mat = np.array([r[:m] for r in rows], dtype=float)
    return names, mat


def _llm_translation_stats(names: List[str], mat: np.ndarray, tau: float):
    """Per-persona endorsement (score >= tau) over the candidate readings."""
    n_cand, n_persona = mat.shape
    endorsed = mat >= tau                       # [n_cand x n_persona] boolean
    per_cand_votes = endorsed.sum(axis=1)       # how many personas endorse each
    maj = int(ceil(n_persona / 2))
    n_majority = int((per_cand_votes >= maj).sum())
    n_union = int((per_cand_votes >= 1).sum())
    # single = the most-endorsed reading (tie-break by higher mean faithfulness)
    order = sorted(range(n_cand),
                   key=lambda i: (per_cand_votes[i], mat[i].mean()), reverse=True)
    llm_single = names[order[0]]
    # distinct per-persona argmax readings (each persona's single best translation)
    toppicks = {int(mat[:, p].argmax()) for p in range(n_persona)}
    n_toppick = len(toppicks)
    disagree = float(mat.std(axis=1).mean())
    return llm_single, n_majority, n_union, n_toppick, disagree


def _u_alpha_sizes(domain: str, sname: str) -> (float, int):
    """Mean / max |U_alpha| over seeds, for context alongside the LLM stats."""
    sizes: List[int] = []
    if domain == "maintenance":
        from .env import MaintenanceMDP
        from .offline import make_offline_rl_dataset
        raw = [c for c, _ in RULE_SETS[sname]["items"]]
        judge = load_llm_cache(sname)
        mdp = MaintenanceMDP(r_op=4.0, c_fail=20.0, c_replace=40.0)
        for seed in range(N_SEEDS):
            data = make_offline_rl_dataset(mdp, seed=seed)
            U = construct_U_alpha(attach_plausibility(raw, judge),
                                  data.to_semantic_dataset()).U_alpha
            sizes.append(len(U))
    else:  # gridworld
        from .gridworld import (RULE_SETS_T, CrossingGridworld,
                                gridworld_semantic_dataset, make_gridworld_dataset)
        cache = str(Path(__file__).with_name("plausibility_cache_temporal.json"))
        raw = [c for c, _ in RULE_SETS_T[sname]["items"]]
        judge = load_llm_cache(sname, cache)
        for seed in range(N_SEEDS):
            env = CrossingGridworld(c_accident=2.0)
            data = make_gridworld_dataset(env, seed=seed)
            U = construct_U_alpha(attach_plausibility(raw, judge),
                                  gridworld_semantic_dataset(data)).U_alpha
            sizes.append(len(U))
    return float(np.mean(sizes)), int(max(sizes))


def collect(tau: float = TAU_PLAUS) -> List[DiversityRow]:
    rows: List[DiversityRow] = []
    specs = [
        ("maintenance", "saorl/plausibility_cache.json",
         ["sharp", "conservative", "ambiguous"]),
        ("gridworld", "saorl/plausibility_cache_temporal.json",
         ["window_sharp", "window_conservative", "window_ambiguous"]),
    ]
    label = {"sharp": "sharp", "conservative": "conservative", "ambiguous": "ambiguous",
             "window_sharp": "sharp", "window_conservative": "conservative",
             "window_ambiguous": "ambiguous"}
    for domain, cache_path, snames in specs:
        blob = json.loads(Path(cache_path).read_text())
        for sname in snames:
            entry = blob["sets"][sname]
            names, mat = _scores_matrix(entry)
            single, n_maj, n_uni, n_top, dis = _llm_translation_stats(names, mat, tau)
            u_mean, u_max = _u_alpha_sizes(domain, sname)
            rows.append(DiversityRow(
                domain=domain, rule=label[sname], n_cand=mat.shape[0],
                n_persona=mat.shape[1], llm_single=single, n_majority=n_maj,
                n_union=n_uni, n_toppick=n_top, disagree=dis,
                u_alpha_mean=u_mean, u_alpha_max=u_max))
    return rows


def main():
    rows = collect()
    print(f"LLM translation diversity (tau={TAU_PLAUS}, {N_SEEDS} seeds for |U_alpha|)\n")
    hdr = (f"  {'domain':12s} {'rule':12s} {'#cand':>5s} {'LLM-single':>16s} "
           f"{'#maj':>5s} {'#union':>6s} {'#toppk':>6s} {'spread':>7s} "
           f"{'|U_a|mean':>9s} {'max':>4s}")
    print(hdr)
    for r in rows:
        print(f"  {r.domain:12s} {r.rule:12s} {r.n_cand:5d} {r.llm_single:>16s} "
              f"{r.n_majority:5d} {r.n_union:6d} {r.n_toppick:6d} {r.disagree:7.3f} "
              f"{r.u_alpha_mean:9.2f} {r.u_alpha_max:4d}")
    out = Path(__file__).with_name("llm_diversity.json")
    out.write_text(json.dumps([r.__dict__ for r in rows], indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
