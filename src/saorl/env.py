"""Maintenance MDP for SA-ORL (plan v12 2.6, Domain 1: C-MAPSS-style upkeep).

A single degrading asset. The latent state is true remaining-useful-life (RUL);
the agent only sees noisy prognostic features (rul_hat, q05, anom) -- the same
schema the semantic DSL predicates read. Actions trade operating revenue against
maintenance cost and failure risk:

    continue       earn r_op, degrade one step; if already dead -> failure (-c_fail)
    minor_repair   pay c_minor, recover `repair_gain` of life
    replace        pay c_replace, reset to full life

This is the substrate the offline dataset (saorl.offline) and the constrained
learner (Algorithm 3, later) operate on. It is deliberately small and CPU-only.
RUL_true is a construction/eval label and is never a policy input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

ACTIONS: Tuple[str, ...] = ("continue", "minor_repair", "replace")
State = Dict[str, float]


@dataclass
class MaintenanceMDP:
    max_life: float = 100.0
    horizon: int = 100
    repair_gain: float = 25.0
    obs_noise: float = 3.0
    anom_noise: float = 0.04
    r_op: float = 1.0       # revenue per healthy operating step
    c_minor: float = 6.0
    c_replace: float = 15.0
    c_fail: float = 60.0
    start_low: float = 70.0
    start_high: float = 90.0

    def observe(self, rul_true: float, rng: np.random.Generator) -> State:
        rul_hat = max(0.0, rul_true + rng.normal(0, self.obs_noise))
        q05 = max(0.0, rul_hat - rng.uniform(4, 8))
        anom = float(
            np.clip(1.0 - rul_true / self.max_life + rng.normal(0, self.anom_noise), 0, 1)
        )
        return dict(rul_true=rul_true, rul_hat=rul_hat, q05=q05, anom=anom)

    def step(self, rul_true: float, action: str) -> Tuple[float, float, bool]:
        """Return (reward, next_rul_true, failed)."""
        if action == "continue":
            if rul_true <= 0.0:
                return -self.c_fail, self.max_life, True  # failure -> forced replace
            return self.r_op, rul_true - 1.0, False
        if action == "minor_repair":
            return -self.c_minor, min(self.max_life, rul_true + self.repair_gain), False
        if action == "replace":
            return -self.c_replace, self.max_life, False
        raise ValueError(f"unknown action: {action}")

    def initial_rul(self, rng: np.random.Generator) -> float:
        return float(rng.integers(int(self.start_low), int(self.start_high) + 1))
