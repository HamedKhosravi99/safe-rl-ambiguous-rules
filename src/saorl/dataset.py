"""Semantic-construction dataset D_sem and activation statistics (plan v12 2.4-2.6).

Incident/normal labels are *construction labels* derived from true RUL inside
D_sem only; they are never deployed policy features. For the maintenance domain
(plan 2.6):  Incident <=> RUL_true <= 20,  Normal <=> RUL_true >= 80.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

from .dsl import Candidate, Trajectory

Point = Tuple[int, int]  # (trajectory_index, t)


@dataclass
class SemanticDataset:
    trajectories: List[Trajectory]
    incident_label: str = "rul_true"
    incident_thresh: float = 20.0
    normal_thresh: float = 80.0

    def incident_points(self) -> Set[Point]:
        pts = set()
        for i, traj in enumerate(self.trajectories):
            for t, s in enumerate(traj):
                if s[self.incident_label] <= self.incident_thresh:
                    pts.add((i, t))
        return pts

    def normal_points(self) -> Set[Point]:
        pts = set()
        for i, traj in enumerate(self.trajectories):
            for t, s in enumerate(traj):
                if s[self.incident_label] >= self.normal_thresh:
                    pts.add((i, t))
        return pts

    def activation_set(self, cand: Candidate) -> Set[Point]:
        """A_psi = {(i, t) : g_psi fires}."""
        acts = set()
        for i, traj in enumerate(self.trajectories):
            for t in range(len(traj)):
                if cand.fires(traj, t):
                    acts.add((i, t))
        return acts


def p_incident(activation: Set[Point], incidents: Set[Point]) -> float:
    """p_inc(psi) = P(g fires | point is an incident)."""
    if not incidents:
        return 0.0
    return len(activation & incidents) / len(incidents)


def p_normal(activation: Set[Point], normals: Set[Point]) -> float:
    """p_norm(psi) = P(g fires | point is normal)."""
    if not normals:
        return 0.0
    return len(activation & normals) / len(normals)


def jaccard(a: Set[Point], b: Set[Point]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
