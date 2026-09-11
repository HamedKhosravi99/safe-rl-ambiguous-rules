"""alpha-sensitivity and judge-robustness sweeps for U_alpha (plan v12 2.5).

The headline uses alpha=0.15, but reviewers will ask how stable |U_alpha| and
its membership are. This reports:
  * |U_alpha| vs alpha in {0.05, 0.10, 0.15, 0.20, 0.30};
  * ensemble-perturbation robustness: how often forcing a single language-judge
    member to the opposite extreme changes the retained set. This is the
    human-free analogue of the old single-annotator-flip metric -- and, because
    the ensemble has 7 members and the plausibility term is down-weighted, it
    should be far more stable than the 3-annotator design.
"""
from __future__ import annotations

from typing import List

from .construct import ALPHA, construct_U_alpha
from .dataset import SemanticDataset
from .demo import ambiguous_rule_candidates, make_trajectories, sharp_rule_candidates
from .dsl import Candidate

ALPHA_GRID = (0.05, 0.10, 0.15, 0.20, 0.30)


def alpha_curve(cands: List[Candidate], dsem: SemanticDataset):
    return [(a, construct_U_alpha(cands, dsem, alpha=a).size()) for a in ALPHA_GRID]


def _perturb_member(cands: List[Candidate], idx: int, member: int) -> List[Candidate]:
    """Force one ensemble member of one candidate to the opposite extreme."""
    out = list(cands)
    c = cands[idx]
    z = list(c.plausibility)
    if member < len(z):
        z[member] = 0.0 if z[member] > 0.5 else 1.0
    out[idx] = Candidate(c.name, c.predicate, c.forbidden_actions, tuple(z))
    return out


def ensemble_perturbation_robustness(cands: List[Candidate], dsem: SemanticDataset):
    """Fraction of single-member perturbations that change the retained set."""
    base = set(construct_U_alpha(cands, dsem).names())
    changed = 0
    total = 0
    for i, c in enumerate(cands):
        for m in range(len(c.plausibility)):
            new = set(construct_U_alpha(_perturb_member(cands, i, m), dsem).names())
            total += 1
            if new != base:
                changed += 1
    return changed / total if total else 0.0


def main():
    dsem = SemanticDataset(make_trajectories())
    for title, cands in [
        ("Rule L1 (ambiguous)", ambiguous_rule_candidates()),
        ("Rule L_sharp", sharp_rule_candidates()),
    ]:
        print(f"\n=== {title} ===")
        print(f"  headline alpha = {ALPHA}")
        print("  alpha -> |U_alpha|:",
              "  ".join(f"{a:.2f}:{n}" for a, n in alpha_curve(cands, dsem)))
        frac = ensemble_perturbation_robustness(cands, dsem)
        print(f"  single-judge-perturbation change rate: {frac:.2f} "
              f"(lower = more reproducible)")


if __name__ == "__main__":
    main()
