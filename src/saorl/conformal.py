"""CORSET Stage 1: conformal interpretation sets over the frozen DSL pools.

This module replaces the fixed plausibility gate (TAU_PLAUS = 0.50) and the
soft alpha-band of `construct.py` with a *split-conformal* threshold q-hat
calibrated on a corpus of (rule, gold formalization) pairs.  The composite
score

    s(rule, psi) = ensemble-mean plausibility of psi under `rule`
                   if psi passes the two frozen data-calibration gates
                   (incident activation >= RHO_INC, overblocking <= RHO_OVER)
                   else 0,

is exactly the audit signal the original construction thresholds -- the gates
and the LM ensemble are unchanged -- but the *threshold* is now the k-th
smallest gold score over the calibration corpus, k = floor(delta_sem (n+1)).
By exchangeability of the corpus rows with a freshly deployed rule, the
retained set U(rule) = {psi : s(rule, psi) >= q-hat} contains the intended
reading with probability >= 1 - delta_sem, *regardless of the LM's quality*
(Theorem 1 of the paper).  A bad score function can only make U larger
(less efficient), never invalid.

Gold labels (authored ground truth, fixed before calibration -- provenance
strings recorded in the output): each domain's environment *defines* the
hazard the rule was written about.  The maintenance incident label is true
RUL <= 20 (env.py / cmapss.py), so the gold reading of the SEVERE rule is the
tightest pool member covering that region (RUL_hat<22 in the conservative
pool, RUL_hat<20 in the DSL pool).  The gridworld hazard lasts
D ~ Uniform{3..5} steps (gridworld.py, dur_max=5), so the gold window is
Within_5.  The budget agent's authored over-budget label is spend >= $55
(budget.py, over_budget=55.0), so the gold cap is the smallest pool cap
dominating it, spend >= $60.  The sharp rules name their own reading
("exactly the four steps" -> Within_4; the wear-out design gloss -> RUL<15).
The sharp $50-cap rule's gold (spend >= 50) is NOT in the candidate pool and
is therefore excluded from calibration and reported separately as the
DSL-misspecification showcase: every candidate scores ~0 and the conformal
set is empty at any calibrated threshold -- the construction abstains.

Everything here is cache-only and deterministic given the frozen caches and
the corpus seed; no LM is queried.

Run:  PYTHONPATH=. python3 -m saorl.conformal
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .build_plaus_cache import RULE_SETS
from .construct import RHO_INC, RHO_OVER, construct_U_alpha
from .dataset import SemanticDataset, p_incident, p_normal
from .dsl import Candidate
from .judge import attach_plausibility, load_llm_cache

DELTA_SEM = 0.1          # target semantic miscoverage
CORPUS_SEED = 0          # frozen seed for the semantic splits used by the gates
_ROUND = 6               # scores are quantized (LM grid 0.05, means of 7) --
                         # rounding kills float knife-edges so exact ties at
                         # q-hat are INCLUDED, per the conformal convention

_HERE = Path(__file__).parent
_CACHE_MAINT = str(_HERE / "plausibility_cache.json")
_CACHE_TEMP = str(_HERE / "plausibility_cache_temporal.json")
_CACHE_BUDGET = str(_HERE / "plausibility_cache_budget.json")
_JUDGE_ROBUST_DIR = _HERE.parent.parent / "results/paper_extra" / "judge_robustness"


# --------------------------------------------------------------------------
# Domain semantic splits (frozen at CORPUS_SEED) and pools
# --------------------------------------------------------------------------

def _maint_dsem() -> SemanticDataset:
    from .env import MaintenanceMDP
    from .offline import make_offline_rl_dataset

    env = MaintenanceMDP(r_op=4.0, c_fail=20.0, c_replace=40.0)
    return make_offline_rl_dataset(env, seed=CORPUS_SEED).to_semantic_dataset()


def _gridworld_dsem() -> SemanticDataset:
    from .gridworld import (
        CrossingGridworld,
        gridworld_semantic_dataset,
        make_gridworld_dataset,
    )

    env = CrossingGridworld(c_accident=2.0)
    return gridworld_semantic_dataset(make_gridworld_dataset(env, seed=CORPUS_SEED))


def _budget_dsem() -> SemanticDataset:
    from .budget import BudgetAgentMDP, budget_semantic_dataset, make_budget_dataset

    env = BudgetAgentMDP()
    return budget_semantic_dataset(env, make_budget_dataset(env, seed=CORPUS_SEED))


def _cmapss_dsem(subset: str) -> SemanticDataset:
    """Real NASA C-MAPSS features: the deployed maintenance rule's gates are
    recomputed on each subset's real degradation data (seed-free, as in
    experiments._real_spec)."""
    from .cmapss import load_features, to_trajectories

    return SemanticDataset(to_trajectories(load_features(subset)))


def _union_activation(pool: Sequence[Candidate], names: Sequence[str],
                      dsem: SemanticDataset) -> frozenset:
    by = {c.name: c for c in pool}
    u: set = set()
    for n in names:
        u |= set(dsem.activation_set(by[n]))
    return frozenset(u)


def binding_equal(pool: Sequence[Candidate], names_a: Sequence[str],
                  names_b: Sequence[str], dsems: Sequence[SemanticDataset]) -> bool:
    """True iff the two retained sets have identical activation UNIONS on every
    provided dataset.  Union equality determines the pointwise shaped cost the
    Lagrangian learner trains on, but full robust-PROGRAM equivalence (the
    max-of-per-reading-expectations constraint) additionally needs member
    dominance (paper Lemma `dominance pruning`).  In this repository every
    union-equal deployment is in fact an identical set, so the distinction is
    moot here; treat a True from non-identical, non-nested sets with care."""
    return all(_union_activation(pool, names_a, d) == _union_activation(pool, names_b, d)
               for d in dsems)


def _pool(rule_sets: dict, set_name: str, cache: str) -> List[Candidate]:
    raw = [c for c, _ in rule_sets[set_name]["items"]]
    return attach_plausibility(raw, load_llm_cache(set_name, cache))


def _pools() -> Dict[str, Tuple[List[Candidate], str]]:
    """(pool with plausibility attached, domain key) per cache set."""
    from .budget import RULE_SETS_B
    from .gridworld import RULE_SETS_T

    return {
        "ambiguous": (_pool(RULE_SETS, "ambiguous", _CACHE_MAINT), "maint"),
        "sharp": (_pool(RULE_SETS, "sharp", _CACHE_MAINT), "maint"),
        "conservative": (_pool(RULE_SETS, "conservative", _CACHE_MAINT), "maint"),
        "window_ambiguous": (_pool(RULE_SETS_T, "window_ambiguous", _CACHE_TEMP), "grid"),
        "window_sharp": (_pool(RULE_SETS_T, "window_sharp", _CACHE_TEMP), "grid"),
        "window_conservative": (_pool(RULE_SETS_T, "window_conservative", _CACHE_TEMP), "grid"),
        "budget_ambiguous": (_pool(RULE_SETS_B, "budget_ambiguous", _CACHE_BUDGET), "budget"),
        "budget_sharp": (_pool(RULE_SETS_B, "budget_sharp", _CACHE_BUDGET), "budget"),
        "budget_conservative": (_pool(RULE_SETS_B, "budget_conservative", _CACHE_BUDGET), "budget"),
    }


# --------------------------------------------------------------------------
# The frozen composite score
# --------------------------------------------------------------------------

def gate_pass(cand: Candidate, dsem: SemanticDataset,
              rho_inc: float = RHO_INC, rho_over: float = RHO_OVER) -> Tuple[bool, float, float]:
    a = dsem.activation_set(cand)
    pinc = p_incident(a, dsem.incident_points())
    pnorm = p_normal(a, dsem.normal_points())
    return (pinc >= rho_inc and pnorm <= rho_over), pinc, pnorm


def composite_scores(pool: Sequence[Candidate], dsem: SemanticDataset,
                     judge_idx: Optional[int] = None) -> Dict[str, float]:
    """s(rule, psi): per-persona (judge_idx) or ensemble-mean (None) plausibility,
    hard-zeroed when a data-calibration gate fails."""
    out: Dict[str, float] = {}
    for c in pool:
        ok, _, _ = gate_pass(c, dsem)
        if not ok:
            out[c.name] = 0.0
        elif judge_idx is None:
            out[c.name] = round(c.plaus_mean(), _ROUND)
        else:
            out[c.name] = round(float(c.plausibility[judge_idx]), _ROUND)
    return out


# --------------------------------------------------------------------------
# Calibration corpus: (rule instance, gold reading), authored-ground-truth
# --------------------------------------------------------------------------

@dataclass
class CorpusRow:
    rule_id: str          # unique row id
    set_name: str         # cache set providing the pool + scores (or 'paraphrase-i')
    domain: str           # maint | grid | budget
    gold: str             # gold candidate name (must be in the pool)
    provenance: str       # why this gold, fixed before calibration
    means_override: Optional[Dict[str, float]] = None  # paraphrase rows: recorded live means


def corpus_rows() -> List[CorpusRow]:
    rows = [
        CorpusRow("maint-sharp", "sharp", "maint", "phi1: RUL_hat<15",
                  "authored design gloss: the wear-out rule's faithful reading"),
        CorpusRow("maint-conservative", "conservative", "maint", "mid: RUL<22",
                  "authored incident label RUL<=20 (env.py); tightest covering pool reading"),
        CorpusRow("maint-ambiguous", "ambiguous", "maint", "psi1: RUL_hat<20",
                  "authored incident label RUL<=20 (env.py); the pool reading matching it"),
        CorpusRow("window-sharp", "window_sharp", "grid", "psi_W4: Within_4(warn)",
                  "rule text names the reading: 'exactly the four steps'"),
        CorpusRow("window-ambiguous", "window_ambiguous", "grid", "psi_W5: Within_5(warn)",
                  "authored hazard duration D~U{3..5} (gridworld.py dur_max=5)"),
        CorpusRow("window-conservative", "window_conservative", "grid", "psi_W5: Within_5(warn)",
                  "authored hazard duration D~U{3..5} (gridworld.py dur_max=5)"),
        CorpusRow("budget-ambiguous", "budget_ambiguous", "budget", "psi_$60: spend>=60",
                  "authored over-budget label spend>=$55 (budget.py); smallest covering cap"),
        CorpusRow("budget-conservative", "budget_conservative", "budget", "psi_$60: spend>=60",
                  "authored over-budget label spend>=$55 (budget.py); smallest covering cap"),
    ]
    # Live paraphrase re-queries of the SEVERE rule (frozen in
    # results/paper_extra/judge_robustness/*.json): three more exchangeable rows over
    # the conservative pool, same authored gold as maint-conservative.
    para = _load_paraphrase_rows()
    rows.extend(para)
    return rows


def _load_paraphrase_rows() -> List[CorpusRow]:
    files = sorted(_JUDGE_ROBUST_DIR.glob("*.json"))
    if not files:
        return []
    data = json.load(open(files[0]))
    rows = []
    for i, r in enumerate(data.get("paraphrase", {}).get("rows", []), 1):
        rows.append(CorpusRow(
            rule_id=f"maint-paraphrase-{i}",
            set_name="conservative",
            domain="maint",
            gold="mid: RUL<22",
            provenance="paraphrase of the SEVERE rule (live re-query, frozen); "
                       "authored incident label RUL<=20",
            means_override={k: float(v) for k, v in r["means"].items()},
        ))
    return rows


# --------------------------------------------------------------------------
# Split conformal
# --------------------------------------------------------------------------

def conformal_threshold(gold_scores: Sequence[float], delta: float = DELTA_SEM) -> float:
    """q-hat = k-th smallest gold score, k = floor(delta * (n+1)).

    Requires n >= 1/delta - 1 (else no valid k >= 1 exists and we return -inf,
    i.e. U = full pool: validity preserved by maximal conservatism)."""
    n = len(gold_scores)
    k = math.floor(delta * (n + 1))
    if k < 1:
        return float("-inf")
    return float(sorted(gold_scores)[k - 1])


@dataclass
class RuleScores:
    scores: Dict[str, float]
    gate_detail: Dict[str, dict]


def _score_all(judge_idx: Optional[int] = None) -> Dict[str, RuleScores]:
    """Composite scores for every cache set, keyed by set name."""
    dsems = {"maint": _maint_dsem(), "grid": _gridworld_dsem(), "budget": _budget_dsem()}
    out: Dict[str, RuleScores] = {}
    for set_name, (pool, dom) in _pools().items():
        dsem = dsems[dom]
        detail = {}
        for c in pool:
            ok, pinc, pnorm = gate_pass(c, dsem)
            detail[c.name] = dict(gate=ok, p_inc=round(pinc, 4), p_norm=round(pnorm, 4),
                                  plaus_mean=round(c.plaus_mean(), 4))
        out[set_name] = RuleScores(composite_scores(pool, dsem, judge_idx), detail)
    return out


def _row_score(row: CorpusRow, scored: Dict[str, RuleScores]) -> float:
    rs = scored[row.set_name]
    if row.means_override is not None:
        # paraphrase rows: live-recorded ensemble means, same frozen gates
        gate_ok = rs.gate_detail[row.gold]["gate"]
        return float(row.means_override[row.gold]) if gate_ok else 0.0
    return rs.scores[row.gold]


def _row_set(row: CorpusRow, scored: Dict[str, RuleScores], qhat: float) -> List[str]:
    rs = scored[row.set_name]
    if row.means_override is not None:
        gates = {n: d["gate"] for n, d in rs.gate_detail.items()}
        eff = {n: (row.means_override.get(n, 0.0) if gates[n] else 0.0)
               for n in rs.scores}
        return [n for n, s in eff.items() if s >= qhat]
    return [n for n, s in rs.scores.items() if s >= qhat]


# --------------------------------------------------------------------------
# Main report
# --------------------------------------------------------------------------

def run(delta: float = DELTA_SEM) -> dict:
    scored = _score_all()
    rows = corpus_rows()
    sigmas = {r.rule_id: _row_score(r, scored) for r in rows}
    n = len(rows)

    # Leave-one-out: each row held out, q-hat calibrated on the rest.
    loo = {}
    for r in rows:
        others = [sigmas[o.rule_id] for o in rows if o.rule_id != r.rule_id]
        q = conformal_threshold(others, delta)
        covered = sigmas[r.rule_id] >= q
        loo[r.rule_id] = dict(sigma=round(sigmas[r.rule_id], 4),
                              qhat_loo=round(q, 4) if math.isfinite(q) else None,
                              covered=bool(covered),
                              gold=r.gold, provenance=r.provenance)
    coverage = float(np.mean([v["covered"] for v in loo.values()]))

    # Full-corpus threshold (for fresh rules, e.g. the DSRL rule).
    q_full = conformal_threshold([sigmas[r.rule_id] for r in rows], delta)

    # Deployed sets: for each rule that backs RL experiments, the conformal set
    # under its LEAVE-THAT-RULE-OUT threshold (no row informs its own set),
    # compared against the original audit's U_alpha.  `binding_equal` reports
    # whether the two sets induce the same pointwise worst-case cost on the
    # domain's data (checked on the corpus-seed dataset and seeds 1..9), i.e.
    # whether downstream policies/results are provably unchanged.
    deployed = {}
    dsems = {"maint": _maint_dsem(), "grid": _gridworld_dsem(), "budget": _budget_dsem()}

    def _multi_dsems(dom: str) -> List[SemanticDataset]:
        global CORPUS_SEED
        keep = CORPUS_SEED
        out = []
        for s in range(10):
            CORPUS_SEED = s
            out.append({"maint": _maint_dsem, "grid": _gridworld_dsem,
                        "budget": _budget_dsem}[dom]())
        CORPUS_SEED = keep
        return out

    for r in rows:
        if r.means_override is not None:
            continue
        others = [sigmas[o.rule_id] for o in rows if o.rule_id != r.rule_id]
        q = conformal_threshold(others, delta)
        U_conf = _row_set(r, scored, q)
        pool, dom = _pools()[r.set_name]
        old = construct_U_alpha(pool, dsems[dom])
        beq = binding_equal(pool, U_conf, old.names(), _multi_dsems(dom))
        deployed[r.rule_id] = dict(
            qhat=round(q, 4),
            U_conformal=sorted(U_conf),
            U_alpha_old=sorted(old.names()),
            gold=r.gold,
            gold_in_conformal=r.gold in U_conf,
            gold_in_old=r.gold in old.names(),
            binding_equal_to_old=bool(beq),
        )

    # Real C-MAPSS deployments of the SEVERE rule: same conservative pool and
    # LOO threshold as maint-conservative, gates recomputed on each subset's
    # real degradation features (as the deployed audit does).
    q_maint = conformal_threshold(
        [sigmas[o.rule_id] for o in rows if o.rule_id != "maint-conservative"], delta)
    pool_c, _ = _pools()["conservative"]
    for subset in ("FD001", "FD002", "FD003", "FD004"):
        dsem_r = _cmapss_dsem(subset)
        sc = composite_scores(pool_c, dsem_r)
        U_conf = [nm for nm, s in sc.items() if s >= q_maint]
        old = construct_U_alpha(pool_c, dsem_r)
        beq = binding_equal(pool_c, U_conf, old.names(), [dsem_r])
        deployed[f"cmapss-{subset}"] = dict(
            qhat=round(q_maint, 4),
            U_conformal=sorted(U_conf),
            U_alpha_old=sorted(old.names()),
            gold="mid: RUL<22",
            gold_in_conformal="mid: RUL<22" in U_conf,
            gold_in_old="mid: RUL<22" in old.names(),
            binding_equal_to_old=bool(beq),
            scores={k: round(v, 4) for k, v in sc.items()},
        )

    # DSRL/OSRL: a FRESH deployed rule (not a calibration row -- the benchmark
    # defines no authored intent), thresholded at the full-corpus q-hat.  No
    # semantic split exists, so the composite score reduces to the language
    # term (Theorem 1 is score-agnostic; only efficiency is affected).
    dsrl_cache = _HERE / "plausibility_cache_dsrl.json"
    if dsrl_cache.exists():
        dc = json.load(open(dsrl_cache))["sets"]["dsrl_cost"]
        dscores = {k: round(float(np.mean(v)), _ROUND)
                   for k, v in dc["scores"].items()}
        deployed["dsrl-cost"] = dict(
            qhat=round(q_full, 4),
            U_conformal=sorted([k for k, v in dscores.items() if v >= q_full]),
            U_alpha_old=["psi_b10: cost<=10", "psi_b20: cost<=20", "psi_b40: cost<=40"],
            gold=None,
            note="fresh rule at deployment; canonical OSRL thresholds as pool; "
                 "no data gates (no labeled semantic split on DSRL)",
            scores=dscores,
            rule_text=dc["rule_text"],
        )

    # Misspecification showcase: the sharp $50-cap rule (gold not in pool).
    rs = scored["budget_sharp"]
    mis = dict(rule="budget-sharp ($50 cap)",
               scores={k: round(v, 4) for k, v in rs.scores.items()},
               U_conformal_at_full_qhat=[k for k, v in rs.scores.items() if v >= q_full],
               note="gold spend>=50 is not in the DSL pool; every candidate scores "
                    "~0 and the conformal set is empty: the construction abstains")

    # Per-judge (single-persona) scores: validity is threshold-free by Theorem 1
    # (any frozen score), efficiency varies. delta for n=8 non-paraphrase rows
    # would need n >= 9, so per-judge uses the same 11 rows but with persona
    # scores where available and recorded means elsewhere (paraphrase rows keep
    # ensemble means: the live re-query stored no per-persona breakdown).
    per_judge = {}
    n_personas = 7
    for j in range(n_personas):
        scored_j = _score_all(judge_idx=j)
        sig_j = {}
        for r in rows:
            if r.means_override is not None:
                sig_j[r.rule_id] = _row_score(r, scored)  # ensemble mean fallback
            else:
                sig_j[r.rule_id] = _row_score(r, scored_j)
        cov = []
        sizes = []
        for r in rows:
            others = [sig_j[o.rule_id] for o in rows if o.rule_id != r.rule_id]
            q = conformal_threshold(others, delta)
            cov.append(sig_j[r.rule_id] >= q)
            if r.means_override is None:
                sizes.append(len(_row_set(r, scored_j, q)))
        per_judge[f"judge_{j}"] = dict(
            loo_coverage=round(float(np.mean(cov)), 4),
            mean_set_size=round(float(np.mean(sizes)), 3),
        )
    ens_sizes = [len(_row_set(r, scored,
                              conformal_threshold([sigmas[o.rule_id] for o in rows
                                                   if o.rule_id != r.rule_id], delta)))
                 for r in rows if r.means_override is None]

    report = dict(
        delta_sem=delta,
        n_corpus=n,
        corpus_seed=CORPUS_SEED,
        achievable_delta_note=f"k=floor(delta*(n+1))={math.floor(delta * (n + 1))}; "
                              f"guarantee level 1-{math.floor(delta * (n + 1))}/{n + 1}",
        loo=loo,
        loo_coverage=coverage,
        qhat_full_corpus=round(q_full, 4),
        deployed=deployed,
        misspecification_showcase=mis,
        per_judge=per_judge,
        ensemble_mean_set_size=round(float(np.mean(ens_sizes)), 3),
        gate_detail={s: rs.gate_detail for s, rs in scored.items()},
    )
    return report


def main() -> None:
    report = run()
    out_dir = _HERE.parent.parent / "results/paper_extra" / "conformal"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "conformal_report.json"
    json.dump(report, open(out, "w"), indent=1)
    print(f"n={report['n_corpus']}  LOO coverage={report['loo_coverage']:.3f}  "
          f"target >= {1 - report['delta_sem'] - 1.0 / (report['n_corpus'] + 1):.3f} "
          f"(granularity 1/(n+1))")
    print(f"q-hat (full corpus) = {report['qhat_full_corpus']}")
    for rid, d in report["deployed"].items():
        print(f"  {rid:22s} qhat={d['qhat']:.3f}  U_conf={d['U_conformal']}")
        # fresh-rule entries (dsrl-cost) have no authored gold, hence no key
        print(f"  {'':22s} old U_alpha={d['U_alpha_old']}"
              f"  gold_in_conf={d.get('gold_in_conformal', 'n/a (fresh rule)')}")
    print("misspecification:", report["misspecification_showcase"]["U_conformal_at_full_qhat"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
