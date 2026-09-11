"""A replay MDP grounded in REAL C-MAPSS FD001 degradation (Domain 1, Phase B).

cmapss_demo grounds the *construction* of U_alpha in real turbofan features. This
module grounds the *control problem* too: instead of the synthetic MaintenanceMDP
generative model, transitions are replayed from actual engine run-to-failure
trajectories. The latent state is (engine, position-along-its-real-life); the agent
sees the same real feature dict (rul_hat/q05/anom) the DSL predicates read.

    continue       advance one real cycle along this engine; running off the end
                   of the recorded life is a run-to-failure (-c_fail), then a fresh
                   engine is drawn;
    minor_repair   pay c_minor, step `repair_gain` cycles back up the real curve;
    replace        pay c_replace, swap in a fresh engine at cycle 0.

So degradation is never simulated by a hand-written model -- every (rul_hat, anom)
the learner conditions on, and every step of wear, comes from NASA's data. The
economics are the same TENSION regime as the synthetic kill test (costly
precautionary replacement), so honoring the rule still trades against return.

This lets the offline-RL kill test (saorl.offline_rl) run on real degradation:
a learner constrained on only the liberal interpretation hides a worst-case
violation over the full U_alpha that constraining on all of U_alpha removes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np

from .dsl import State
from .offline import OfflineDataset

EngineState = Tuple[int, int]  # (engine index, position along its real life)


@dataclass
class ReplayMDP:
    """Maintenance control replayed over real C-MAPSS engine trajectories.

    Economics default to the synthetic kill test's TENSION regime so results are
    comparable: costly precautionary replacement (c_replace) makes honoring the
    rule sacrifice operating revenue (r_op).
    """

    engines: List[List[State]]
    horizon: int = 120
    repair_gain: int = 30          # cycles of life recovered by a minor repair
    r_op: float = 4.0
    c_minor: float = 6.0
    c_replace: float = 40.0
    c_fail: float = 20.0

    def n_engines(self) -> int:
        return len(self.engines)

    def reset(self, rng: np.random.Generator) -> EngineState:
        return (int(rng.integers(self.n_engines())), 0)

    def start(self, rng: np.random.Generator, frac_range: Tuple[float, float] = (0.0, 0.0)) -> EngineState:
        """Draw a start state; frac_range picks the position as a fraction of the
        engine's real life (so deep starts land in the degraded band)."""
        ei = int(rng.integers(self.n_engines()))
        L = len(self.engines[ei])
        lo, hi = frac_range
        a, b = int(lo * L), int(hi * L)
        pos = int(rng.integers(a, b + 1)) if b > a else a
        return (ei, min(pos, L - 1))

    def observe(self, st: EngineState) -> State:
        ei, pos = st
        return self.engines[ei][pos]

    def step(
        self, st: EngineState, action: str, rng: np.random.Generator
    ) -> Tuple[float, EngineState, bool]:
        ei, pos = st
        if action == "continue":
            if pos + 1 >= len(self.engines[ei]):
                return -self.c_fail, self.reset(rng), True  # ran real life to failure
            return self.r_op, (ei, pos + 1), False
        if action == "minor_repair":
            return -self.c_minor, (ei, max(0, pos - self.repair_gain)), False
        if action == "replace":
            return -self.c_replace, self.reset(rng), False
        raise ValueError(f"unknown action: {action}")


def generate_replay_dataset(
    env: ReplayMDP,
    behavior: Callable,
    n_episodes: int = 80,
    seed: int = 0,
    start_frac: Tuple[float, float] = (0.0, 0.0),
) -> OfflineDataset:
    """Roll out a behavior policy in the replay MDP. `start_frac` controls where on
    the real degradation curve each episode begins (see ReplayMDP.start)."""
    rng = np.random.default_rng(seed)
    trajs, acts, rews = [], [], []
    for _ in range(n_episodes):
        st = env.start(rng, start_frac)
        traj, a_list, r_list = [], [], []
        for t in range(env.horizon):
            s = env.observe(st)
            traj.append(s)
            a = behavior(traj, t, rng)
            r, st, _ = env.step(st, a, rng)
            a_list.append(a)
            r_list.append(r)
        trajs.append(traj)
        acts.append(a_list)
        rews.append(r_list)
    return OfflineDataset(trajs, acts, rews)


def make_replay_rl_dataset(env: ReplayMDP, seed: int = 0) -> OfflineDataset:
    """Heterogeneous logged dataset over real degradation (mirrors
    offline.make_offline_rl_dataset): descending episodes to exercise the rule,
    plus broad random-action coverage (whole-life and deep starts) so the honoring
    action has support exactly where the rule fires."""
    from .offline import concat_datasets, coverage_behavior, epsilon_threshold_behavior

    d1 = generate_replay_dataset(
        env, epsilon_threshold_behavior(), n_episodes=150, seed=seed
    )
    d2 = generate_replay_dataset(
        env, coverage_behavior(), n_episodes=250, seed=seed + 1, start_frac=(0.0, 0.9)
    )
    d3 = generate_replay_dataset(
        env, coverage_behavior(), n_episodes=200, seed=seed + 2, start_frac=(0.55, 0.9)
    )
    return concat_datasets(d1, d2, d3)


def evaluate_replay_return(
    env: ReplayMDP, policy, n_episodes: int = 40, seed: int = 123
) -> float:
    """Mean episodic return of a target policy over real-degradation rollouts."""
    rng = np.random.default_rng(seed)
    totals = []
    for _ in range(n_episodes):
        st = env.reset(rng)
        traj, total = [], 0.0
        for t in range(env.horizon):
            traj.append(env.observe(st))
            a = policy(traj, t)
            r, st, _ = env.step(st, a, rng)
            total += r
        totals.append(total)
    return float(np.mean(totals))
