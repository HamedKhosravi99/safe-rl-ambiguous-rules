"""Frozen construction of U_alpha (plan v12 Algorithm 1, sections 2.4-2.7).

This is the novel object of the paper: the language-induced, audited semantic
ambiguity set. Everything here is frozen *before* any RL policy is trained.

Human-free construction (project direction):
  The original protocol weighted a 3-annotator plausibility term at lambda=0.40,
  which made it the dominant term and the source of the construction's fragility
  (a single annotator flip changed the retained set ~67% of the time). We remove
  recruited humans entirely. Plausibility now comes from a frozen *ensemble* of
  language judgments (saorl.judge -- a cached LLM ensemble, or a no-LLM
  data-driven ablation), down-weighted to 0.35 so the data-grounded terms
  (incident activation, overblocking, duplication, simplicity) jointly dominate
  at 0.65. The "at least 2 of 3 annotators" eligibility gate becomes a single
  ensemble-mean threshold TAU_PLAUS. Net effect: U_alpha is fully computational,
  reproducible from a cache, and markedly more stable to single-judge perturbation.

Implementation note (flagged for the paper):
  L_dup (plan 2.4) is defined as the max Jaccard overlap with `U_kept`, which is
  order-dependent, yet Algorithm 1 (p.28) computes L_sem for all candidates and
  then band-filters. We resolve the circularity with a deterministic greedy
  rule: candidates are ordered by their base score (L_sem without the dup term);
  L_dup(psi) is the max Jaccard against already-accepted candidates earlier in
  that order. This is a defensible, pre-registerable tie-break and should be
  stated explicitly in the paper's protocol to keep U_alpha reproducible.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .dataset import (
    Point,
    SemanticDataset,
    jaccard,
    p_incident,
    p_normal,
)
from .dsl import Candidate

# Frozen protocol constants (plan 2.4-2.5, human-free revision).
# plaus is the language-ensemble term; the data terms jointly dominate (0.65).
LAMBDA = dict(plaus=0.35, inc=0.25, over=0.20, simp=0.12, dup=0.08)
C_MAX = 5
RHO_INC = 0.60
RHO_OVER = 0.20
ALPHA = 0.15
TAU_PLAUS = 0.50  # ensemble-mean plausibility gate (replaces >=2/3 human votes)


@dataclass
class CandidateReport:
    name: str
    l_plaus: float
    l_inc: float
    l_over: float
    l_simp: float
    l_dup: float
    l_sem: float
    p_inc: float
    p_norm: float
    plaus: float
    retained: bool
    reason: str  # "" if retained, else rejection reason


@dataclass
class ConstructionResult:
    U_alpha: List[Candidate]
    reports: List[CandidateReport]
    l_min: float
    empty_diagnostic: bool

    def size(self) -> int:
        return len(self.U_alpha)

    def names(self) -> List[str]:
        return [c.name for c in self.U_alpha]


# Deployed conformal threshold: the leave-rule-out k-th smallest gold score on
# the frozen calibration corpus at delta_sem = 0.1 (saorl.conformal; identical
# for every deployed rule, see results/paper_extra/conformal/conformal_report.json).
QHAT_DEPLOYED = 0.5


def _construct_conformal(
    candidates: List[Candidate],
    dsem: SemanticDataset,
    rho_inc: float = RHO_INC,
    rho_over: float = RHO_OVER,
) -> ConstructionResult:
    """CORSET Stage 1 at deployment: retain every candidate whose composite
    score (ensemble-mean plausibility, hard-zeroed on a data-gate failure)
    clears the calibrated conformal threshold.  Ties at q-hat are retained,
    per the conformal convention (validity requires >=)."""
    incidents = dsem.incident_points()
    normals = dsem.normal_points()
    U: List[Candidate] = []
    reports: List[CandidateReport] = []
    for c in candidates:
        a = dsem.activation_set(c)
        pinc = p_incident(a, incidents)
        pnorm = p_normal(a, normals)
        plaus = round(c.plaus_mean(), 6) if c.has_plausibility() else 0.0
        gate_ok = pinc >= rho_inc and pnorm <= rho_over
        score = plaus if gate_ok else 0.0
        retained = score >= QHAT_DEPLOYED
        if retained:
            U.append(c)
            reason = ""
        elif not gate_ok:
            reason = ("low incident activation" if pinc < rho_inc else "overblocking")
        else:
            reason = "below conformal q-hat"
        reports.append(
            CandidateReport(
                name=c.name, l_plaus=1.0 - plaus, l_inc=max(0.0, rho_inc - pinc),
                l_over=max(0.0, pnorm - rho_over), l_simp=0.0, l_dup=0.0,
                l_sem=1.0 - score, p_inc=pinc, p_norm=pnorm, plaus=plaus,
                retained=retained, reason=reason,
            )
        )
    return ConstructionResult(
        U_alpha=U, reports=reports,
        l_min=min((r.l_sem for r in reports), default=0.0),
        empty_diagnostic=(len(U) == 0),
    )


def _l_plaus(cand: Candidate) -> float:
    """Language-implausibility = 1 - ensemble-mean plausibility.

    With no ensemble attached the term is neutral (0.0): the candidate is judged
    purely on its data signature rather than penalized for a missing signal.
    """
    if not cand.has_plausibility():
        return 0.0
    return 1.0 - cand.plaus_mean()


def _l_simp(cand: Candidate) -> float:
    return cand.complexity() / C_MAX


def construct_U_alpha(
    candidates: List[Candidate],
    dsem: SemanticDataset,
    alpha: float = ALPHA,
    rho_inc: float = RHO_INC,
    rho_over: float = RHO_OVER,
    tau_plaus: float = TAU_PLAUS,
) -> ConstructionResult:
    """Algorithm 1: frozen construction of the semantic ambiguity set U_alpha.

    alpha/rho_inc/rho_over/tau_plaus default to the frozen protocol constants; they
    are exposed as arguments only so the set-construction ablations (reviewer #6 ---
    sensitivity to the plausibility gate tau and the soft band alpha) can sweep them
    without editing module state. Production callers pass nothing and get the frozen
    protocol.

    CORSET switch: with SAORL_CONFORMAL=1 in the environment, every call site
    is routed to the split-conformal construction instead: the identical
    composite audit signal (LM-ensemble plausibility hard-gated by the two
    data-calibration constraints), thresholded at the corpus-calibrated q-hat
    (saorl.conformal) rather than at the fixed TAU_PLAUS plus alpha-band. The
    alpha/tau knobs are then inert by design -- conformal has no free
    threshold to tune."""
    if os.environ.get("SAORL_CONFORMAL", "0") == "1":
        return _construct_conformal(candidates, dsem, rho_inc, rho_over)
    incidents = dsem.incident_points()
    normals = dsem.normal_points()

    # Precompute per-candidate activation + base loss terms.
    acts: Dict[str, Set[Point]] = {}
    base: Dict[str, dict] = {}
    for c in candidates:
        a = dsem.activation_set(c)
        acts[c.name] = a
        pinc = p_incident(a, incidents)
        pnorm = p_normal(a, normals)
        base[c.name] = dict(
            l_plaus=_l_plaus(c),
            l_inc=max(0.0, rho_inc - pinc),
            l_over=max(0.0, pnorm - rho_over),
            l_simp=_l_simp(c),
            p_inc=pinc,
            p_norm=pnorm,
        )

    # Base score (no dup term) -> deterministic greedy order for L_dup.
    def base_score(name: str) -> float:
        b = base[name]
        return (
            LAMBDA["plaus"] * b["l_plaus"]
            + LAMBDA["inc"] * b["l_inc"]
            + LAMBDA["over"] * b["l_over"]
            + LAMBDA["simp"] * b["l_simp"]
        )

    order = sorted((c.name for c in candidates), key=lambda n: (base_score(n), n))
    l_dup: Dict[str, float] = {}
    kept_acts: List[Set[Point]] = []
    for name in order:
        d = max((jaccard(acts[name], k) for k in kept_acts), default=0.0)
        l_dup[name] = d
        kept_acts.append(acts[name])

    by_name = {c.name: c for c in candidates}
    l_sem: Dict[str, float] = {
        n: base_score(n) + LAMBDA["dup"] * l_dup[n] for n in by_name
    }
    l_min = min(l_sem.values()) if l_sem else 0.0

    U: List[Candidate] = []
    reports: List[CandidateReport] = []
    for c in candidates:
        n = c.name
        b = base[n]
        # Hard, weight-independent filters first (language gate, then the two
        # data calibration constraints); the soft alpha-band trims what remains.
        reason = ""
        if c.has_plausibility() and c.plaus_mean() < tau_plaus:
            reason = "insufficient plausibility"
        elif b["p_inc"] < rho_inc:
            reason = "low incident activation"
        elif b["p_norm"] > rho_over:
            reason = "overblocking"
        elif l_sem[n] > l_min + alpha:
            reason = "outside alpha band"
        retained = reason == ""
        if retained:
            U.append(c)
        reports.append(
            CandidateReport(
                name=n,
                l_plaus=b["l_plaus"],
                l_inc=b["l_inc"],
                l_over=b["l_over"],
                l_simp=b["l_simp"],
                l_dup=l_dup[n],
                l_sem=l_sem[n],
                p_inc=b["p_inc"],
                p_norm=b["p_norm"],
                plaus=c.plaus_mean(),
                retained=retained,
                reason=reason,
            )
        )

    return ConstructionResult(
        U_alpha=U,
        reports=reports,
        l_min=l_min,
        empty_diagnostic=(len(U) == 0),
    )
