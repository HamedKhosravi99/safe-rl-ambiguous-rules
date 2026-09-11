"""WP-F.2: three non-nested task families, self-contained.

Everything here is pure numpy with pinned seeds: no simulator downloads, no
dataset host, no external benchmark dependency. The previous neural campaign
lost two of nine cells to an unreachable dataset host; this removes that
failure mode entirely.

Each family exposes two cost channels c1, c2 standing for two readings of one
natural-language rule, plus a deterministic labeller for the disagreement
regions D_{1\\2} and D_{2\\1} (plan F.3). The families are parameterised by a
DECORRELATION knob: the two costs are triggered by partially independent
latent processes, which is what makes a genuine crossing possible rather than
a nested threshold pair. Whether a given configuration actually crosses is
decided by the prescreen (plan F.4), never assumed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

HORIZON = 100


@dataclass
class Config:
    family: str
    name: str
    params: Dict[str, float]
    budget: float = 0.05          # d, per-episode normalised cost budget


class BaseEnv:
    """Minimal continuous-control interface with a vector of costs."""
    obs_dim: int = 0
    act_dim: int = 0
    n_costs: int = 2

    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.p = cfg.params
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        return self._reset()

    def _reset(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, a: np.ndarray):
        raise NotImplementedError

    def fallback_action(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Family 1: navigation -- "Move cautiously near people and fragile equipment."
#   psi_1 : speed above theta_v while inside the person radius
#   psi_2 : contact force above theta_F while inside the fragile radius
# The two zones sit on opposite sides of the direct line to the goal, so a
# fast wide route violates only psi_1 and a slow forceful route only psi_2.
# --------------------------------------------------------------------------
class Navigation(BaseEnv):
    """A barrier at x=0 leaves two corridors and forces a choice.

    The upper corridor passes a person, the lower passes fragile equipment.
    A speed-constrained policy takes the lower corridor and pays in contact
    force; a force-constrained policy takes the upper corridor and pays in
    speed. Only a slow, gentle traversal is feasible for both readings, and
    it gives up return. The barrier is what makes the two readings
    non-nested: without it the straight line is safe for both and no
    crossing exists.
    """
    obs_dim, act_dim = 10, 2
    GAP = 0.18

    def _reset(self):
        self.pos = np.array([-1.0, 0.0])
        self.vel = np.zeros(2)
        self.goal = np.array([1.0, 0.0])
        sep = self.p["zone_sep"]
        self.person = np.array([0.0, sep])
        self.fragile = np.array([0.0, -sep])
        return self._obs()

    def _obs(self):
        return np.concatenate([self.pos, self.vel,
                               self.pos - self.person, self.pos - self.fragile,
                               self.goal - self.pos])

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=float), -1, 1)
        self.vel = 0.80 * self.vel + 0.20 * a
        nxt = self.pos + 0.15 * self.vel
        # impassable barrier segment at x=0, |y| < GAP
        if (self.pos[0] < 0.0 <= nxt[0] or nxt[0] < 0.0 <= self.pos[0]):
            frac = (0.0 - self.pos[0]) / (nxt[0] - self.pos[0] + 1e-9)
            y_cross = self.pos[1] + frac * (nxt[1] - self.pos[1])
            if abs(y_cross) < self.GAP:
                nxt = self.pos.copy()
                self.vel = -0.2 * self.vel
        self.pos = np.clip(nxt, -1.6, 1.6)
        self.t += 1
        speed = float(np.linalg.norm(self.vel))
        impact = float(np.linalg.norm(a)) * speed
        d_p = float(np.linalg.norm(self.pos - self.person))
        d_f = float(np.linalg.norm(self.pos - self.fragile))
        c1 = float(d_p < self.p["r_person"] and speed > self.p["theta_v"])
        c2 = float(d_f < self.p["r_fragile"] and impact > self.p["theta_F"])
        prev = float(np.linalg.norm(self.pos - 0.15 * self.vel - self.goal))
        now = float(np.linalg.norm(self.pos - self.goal))
        rew = 10.0 * (prev - now) - 0.01 * float(np.dot(a, a))
        done = bool(now < 0.15 or self.t >= HORIZON)
        if now < 0.15:
            rew += 10.0
        return self._obs(), rew, np.array([c1, c2]), done, {}

    def fallback_action(self, obs):
        return np.zeros(2)                      # stop in place


# --------------------------------------------------------------------------
# Family 2: locomotion -- "Avoid unstable or damaging movement."
#   psi_1 : tilt magnitude above theta_tilt
#   psi_2 : ground impact above theta_impact
# Leaning trades impact for tilt, so a low-impact gait is tilted and an
# upright gait lands hard; an upright slow gait is safe for both.
# --------------------------------------------------------------------------
class Locomotion(BaseEnv):
    obs_dim, act_dim = 6, 2

    def _reset(self):
        self.tilt = 0.0
        self.tilt_rate = 0.0
        self.vel = 0.0
        self.phase = 0.0
        return self._obs()

    def _obs(self):
        return np.array([self.tilt, self.tilt_rate, self.vel,
                         math.sin(self.phase), math.cos(self.phase),
                         float(self.t) / HORIZON])

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=float), -1, 1)
        push, lean = float(a[0]), float(a[1])
        k = self.p["lean_gain"]
        self.tilt_rate = 0.80 * self.tilt_rate + k * lean - 0.25 * self.tilt
        self.tilt = float(np.clip(self.tilt + 0.1 * self.tilt_rate, -1.5, 1.5))
        # leaning converts vertical loading into forward motion: less impact
        lean_relief = self.p["relief"] * abs(self.tilt)
        self.vel = 0.9 * self.vel + 0.35 * max(push, 0.0) * (1.0 + 0.3 * abs(self.tilt))
        impact = max(0.0, 0.9 * max(push, 0.0) + 0.5 * self.vel - lean_relief)
        impact += self.p["impact_noise"] * float(self.rng.normal())
        self.phase += 0.3
        self.t += 1
        c1 = float(abs(self.tilt) > self.p["theta_tilt"])
        c2 = float(impact > self.p["theta_impact"])
        rew = float(self.vel) - 0.01 * float(np.dot(a, a))
        return self._obs(), rew, np.array([c1, c2]), bool(self.t >= HORIZON), {}

    def fallback_action(self, obs):
        return np.array([-1.0, 0.0])            # stop pushing, stay upright


# --------------------------------------------------------------------------
# Family 3: maintenance -- "Intervene when persistent degradation becomes
# serious."
#   psi_1 : remaining useful life below theta_R while still running
#   psi_2 : anomaly sustained above theta_A for k steps while still running
# RUL and the anomaly process share only `rho` of their innovation, so the
# two triggers fire on different episodes. Servicing on one signal alone
# leaves the other reading violated.
# --------------------------------------------------------------------------
class Maintenance(BaseEnv):
    """Two maintenance channels that each fix only ONE reading.

    `service` restores remaining useful life but does nothing for the anomaly
    process; `clean` suppresses the anomaly but does not restore life. This
    separation is what puts the two readings in tension. With a single
    combined action -- the first version of this environment -- one control
    satisfied both readings at once, the unconstrained optimum was already
    safe, and no crossing existed: the readings were effectively nested and
    the prescreen correctly found nothing.
    """
    obs_dim, act_dim = 6, 2

    def _reset(self):
        self.rul = 1.0
        self.anom = 0.2
        self.streak = 0
        return self._obs()

    def _obs(self):
        return np.array([self.rul, self.anom,
                         self.streak / max(self.p["k"], 1),
                         float(self.t) / HORIZON,
                         float(self.rul < self.p["theta_R"]),
                         float(self.anom > self.p["theta_A"])])

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=float).ravel(), -1, 1)
        service = 0.5 * (float(a[0]) + 1.0)     # restores life only
        clean = 0.5 * (float(a[1]) + 1.0)       # suppresses anomaly only
        rho = self.p["rho"]
        shared = float(self.rng.normal())
        wear = 0.020 + 0.006 * (rho * shared + math.sqrt(max(1 - rho ** 2, 0))
                                * float(self.rng.normal()))
        self.rul = float(np.clip(self.rul - max(wear, 0.0) + 0.055 * service,
                                 0.0, 1.0))
        drift = self.p["anom_drift"] * (rho * shared
                                        + math.sqrt(max(1 - rho ** 2, 0))
                                        * float(self.rng.normal()))
        self.anom = float(np.clip(0.95 * self.anom + 0.055 + drift
                                  - 0.30 * clean, 0.0, 1.0))
        self.streak = self.streak + 1 if self.anom > self.p["theta_A"] else 0
        self.t += 1
        c1 = float(self.rul < self.p["theta_R"] and service < 0.5)
        c2 = float(self.streak >= self.p["k"] and clean < 0.5)
        rew = 1.0 - self.p["w_service"] * service - self.p["w_clean"] * clean
        return self._obs(), rew, np.array([c1, c2]), bool(self.t >= HORIZON), {}

    def fallback_action(self, obs):
        return np.array([1.0, 1.0])             # full service and cleaning


FAMILIES = {"navigation": Navigation, "locomotion": Locomotion,
            "maintenance": Maintenance}


def make(cfg: Config, seed: int = 0) -> BaseEnv:
    return FAMILIES[cfg.family](cfg, seed)


# --------------------------------------------------------------------------
# WP-F.7: ~30 candidate configurations, frozen before the prescreen runs.
# --------------------------------------------------------------------------
def candidate_configs() -> list:
    cfgs = []
    for i, (sep, rp, rf, tv, tf) in enumerate([
            (0.42, 0.34, 0.34, 0.30, 0.22), (0.38, 0.32, 0.32, 0.26, 0.18),
            (0.46, 0.36, 0.36, 0.34, 0.26), (0.40, 0.30, 0.36, 0.28, 0.20),
            (0.44, 0.36, 0.30, 0.32, 0.24), (0.36, 0.34, 0.34, 0.24, 0.16),
            (0.48, 0.38, 0.38, 0.36, 0.28), (0.42, 0.32, 0.38, 0.30, 0.18),
            (0.40, 0.38, 0.32, 0.26, 0.26), (0.46, 0.30, 0.30, 0.34, 0.20)]):
        cfgs.append(Config("navigation", f"nav{i}", dict(
            zone_sep=sep, r_person=rp, r_fragile=rf, theta_v=tv, theta_F=tf),
            budget=0.02))
    for i, (lg, rel, tt, ti, nz) in enumerate([
            (0.55, 0.55, 0.35, 0.55, 0.05), (0.65, 0.60, 0.30, 0.50, 0.05),
            (0.50, 0.50, 0.40, 0.60, 0.04), (0.60, 0.65, 0.35, 0.45, 0.06),
            (0.70, 0.55, 0.25, 0.55, 0.05), (0.45, 0.60, 0.45, 0.50, 0.05),
            (0.60, 0.45, 0.30, 0.65, 0.04), (0.55, 0.70, 0.35, 0.40, 0.06),
            (0.65, 0.50, 0.28, 0.58, 0.05), (0.50, 0.65, 0.42, 0.48, 0.05)]):
        cfgs.append(Config("locomotion", f"loco{i}", dict(
            lean_gain=lg, relief=rel, theta_tilt=tt, theta_impact=ti,
            impact_noise=nz), budget=0.05))
    for i, (rho, tr, ta, k, ad, ws, wc) in enumerate([
            (0.10, 0.35, 0.55, 5, 0.10, 0.55, 0.45),
            (0.00, 0.30, 0.50, 4, 0.12, 0.50, 0.50),
            (0.20, 0.40, 0.60, 6, 0.10, 0.60, 0.40),
            (0.10, 0.25, 0.50, 3, 0.14, 0.45, 0.55),
            (0.00, 0.35, 0.45, 5, 0.12, 0.55, 0.50),
            (0.30, 0.30, 0.55, 4, 0.10, 0.50, 0.45),
            (0.15, 0.45, 0.60, 7, 0.09, 0.65, 0.40),
            (0.05, 0.30, 0.52, 5, 0.13, 0.50, 0.55),
            (0.25, 0.38, 0.48, 4, 0.11, 0.55, 0.45),
            (0.10, 0.42, 0.58, 6, 0.10, 0.60, 0.50)]):
        cfgs.append(Config("maintenance", f"maint{i}", dict(
            rho=rho, theta_R=tr, theta_A=ta, k=k, anom_drift=ad,
            w_service=ws, w_clean=wc), budget=0.05))
    return cfgs


def region_of(c: np.ndarray) -> str:
    """Deterministic disagreement-region label for one transition (plan F.3)."""
    c1, c2 = float(c[0]), float(c[1])
    if c1 > 0 and c2 == 0:
        return "D1_only"
    if c2 > 0 and c1 == 0:
        return "D2_only"
    if c1 > 0 and c2 > 0:
        return "both"
    return "neither"
