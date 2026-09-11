"""Offline dataset generation and policy semantic-cost evaluation (Gate-B prep).

Produces an offline dataset D by rolling out a noisy behavior policy in the
MaintenanceMDP, then evaluates *target* policies against a retained ambiguity
set U_alpha. The key quantity is the worst-case semantic cost over U_alpha --
the object Algorithm 3 will minimize. Everything here is closed-form on the
dataset / model and runs on CPU.

Policy convention: a policy is a callable (trajectory, t) -> action. Semantic
cost c_psi(s_t, a_t) = 1{ g_psi fires at t and a_t in B_psi }; here every
candidate forbids "continue", so a policy "respects" psi by not continuing where
g_psi fires.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

from .dsl import Candidate, State
from .dataset import SemanticDataset
from .env import ACTIONS, MaintenanceMDP

Policy = Callable[[Sequence[State], int], str]


@dataclass
class OfflineDataset:
    trajectories: List[List[State]]
    actions: List[List[str]]
    rewards: List[List[float]]

    def to_semantic_dataset(self, **kw) -> SemanticDataset:
        return SemanticDataset(self.trajectories, **kw)

    def n_steps(self) -> int:
        return sum(len(a) for a in self.actions)

    def mean_return(self) -> float:
        if not self.rewards:
            return 0.0
        return float(np.mean([sum(r) for r in self.rewards]))


# --- data collection ---------------------------------------------------------
def epsilon_threshold_behavior(
    theta: float = 4.0, eps_delay: float = 0.3, eps_repair: float = 0.05
) -> Callable:
    """Suboptimal data-collection policy that *descends* into the incident band.

    Above the threshold it usually operates (rare minor_repair for action
    coverage); at/below the threshold it usually replaces but with prob
    `eps_delay` keeps operating -- delaying maintenance, which drives true RUL
    down to (and past) the incident band so the dataset covers degraded states.
    Exploration never tops the asset up, so degradation is actually reached.
    """
    def behavior(traj: Sequence[State], t: int, rng: np.random.Generator) -> str:
        s = traj[t]
        if s["rul_hat"] <= theta:
            return "continue" if rng.random() < eps_delay else "replace"
        return "minor_repair" if rng.random() < eps_repair else "continue"
    return behavior

def exploratory_maintenance_behavior(
    band: float = 40.0,
    p_continue_band: float = 0.5,
    p_replace_band: float = 0.25,
    p_repair_healthy: float = 0.05,
    p_replace_healthy: float = 0.03,
) -> Callable:
    """Behavior policy with action coverage in the degraded band.

    A learner can only be taught to honor a rule the logged policy actually
    demonstrates. The descending eps-threshold policy never replaces/repairs in
    the mid-degradation band, so an offline learner has no support for the
    honoring action there (BCQ correctly refuses it). This policy fixes coverage:

      * healthy band (rul_hat > `band`): mostly operate -- so assets still
        *descend* into incidents -- with small probabilities of minor_repair /
        replace for action coverage;
      * degraded band (rul_hat <= `band`, where the rule fires): operate only
        `p_continue_band` of the time (enough episodes still reach deep
        incidents) and otherwise replace or minor_repair, giving the honoring
        actions real support exactly where the rule is relevant.

    It is still clearly suboptimal/noisy -- a legitimate offline behavior policy,
    not an oracle.
    """
    def behavior(traj: Sequence[State], t: int, rng: np.random.Generator) -> str:
        s = traj[t]
        u = rng.random()
        if s["rul_hat"] > band:
            if u < p_repair_healthy:
                return "minor_repair"
            if u < p_repair_healthy + p_replace_healthy:
                return "replace"
            return "continue"
        if u < p_continue_band:
            return "continue"
        if u < p_continue_band + p_replace_band:
            return "replace"
        return "minor_repair"
    return behavior

def coverage_behavior(p_continue: float = 0.34) -> Callable:
    """Near-uniform random action policy, for broad (state, action) coverage.

    Offline RL can only learn an action it has seen at a state. To teach a policy
    to honor the rule (i.e. NOT continue, by replacing/repairing) across the whole
    firing region, the dataset must contain replace/minor_repair throughout that
    region. Paired with randomized initial RUL (see generate_offline_dataset's
    `init_range`), this explores the degraded band with all three actions. It is a
    deliberately un-optimized data-collection policy, exactly the kind of broad
    logging an offline-RL dataset is assumed to come from.
    """
    rest = (1.0 - p_continue) / 2.0

    def behavior(traj: Sequence[State], t: int, rng: np.random.Generator) -> str:
        u = rng.random()
        if u < p_continue:
            return "continue"
        if u < p_continue + rest:
            return "replace"
        return "minor_repair"
    return behavior

def generate_offline_dataset(
    mdp: MaintenanceMDP,
    behavior: Callable,
    n_episodes: int = 80,
    seed: int = 0,
    init_range: tuple = None,
) -> OfflineDataset:
    """Roll out `behavior` in `mdp`. If `init_range=(lo,hi)` is given, each
    episode's initial true RUL is drawn uniformly from [lo,hi] (overriding the
    MDP's default start band) so the dataset can cover degraded start states."""
    rng = np.random.default_rng(seed)
    trajs, acts, rews = [], [], []
    for _ in range(n_episodes):
        rul = (float(rng.integers(init_range[0], init_range[1] + 1))
               if init_range else mdp.initial_rul(rng))
        traj, a_list, r_list = [], [], []
        for t in range(mdp.horizon):
            s = mdp.observe(rul, rng)
            traj.append(s)
            a = behavior(traj, t, rng)
            r, rul, _ = mdp.step(rul, a)
            a_list.append(a)
            r_list.append(r)
        trajs.append(traj)
        acts.append(a_list)
        rews.append(r_list)
    return OfflineDataset(trajs, acts, rews)


def concat_datasets(*datasets: "OfflineDataset") -> "OfflineDataset":
    trajs, acts, rews = [], [], []
    for d in datasets:
        trajs += d.trajectories
        acts += d.actions
        rews += d.rewards
    return OfflineDataset(trajs, acts, rews)


def make_offline_rl_dataset(mdp: MaintenanceMDP, seed: int = 0) -> "OfflineDataset":
    """Heterogeneous logged dataset for the offline *learner* (saorl.offline_rl).

    Mixes three behavior sources so the data has BOTH the coverage construction
    needs and the action coverage learning needs:
      * descending eps-threshold episodes -> drive assets into the incident band
        (so U_alpha can be constructed and the rule is exercised);
      * broad random-action episodes with randomized initial RUL -> demonstrate
        replace/minor_repair across the whole state space;
      * band-focused random-action episodes (start already degraded) -> dense
        support for the honoring action exactly where the rule fires.
    A learner cannot honor an action it never saw; this is the standard offline
    assumption that the logging policy covered the relevant (state, action) space.
    """
    d1 = generate_offline_dataset(
        mdp, epsilon_threshold_behavior(), n_episodes=150, seed=seed
    )
    d2 = generate_offline_dataset(
        mdp, coverage_behavior(), n_episodes=250, seed=seed + 1, init_range=(5, 95)
    )
    d3 = generate_offline_dataset(
        mdp, coverage_behavior(), n_episodes=200, seed=seed + 2, init_range=(8, 45)
    )
    return concat_datasets(d1, d2, d3)


# --- target policies ---------------------------------------------------------
def greedy_continue_policy() -> Policy:
    """Reward-myopic: always operate (ignores every interpretation)."""
    def pol(traj, t):
        return "continue"
    return pol

def respect_policy(cands: Sequence[Candidate]) -> Policy:
    """Replace whenever any candidate in `cands` fires, else continue."""
    def pol(traj, t):
        return "replace" if any(c.fires(traj, t) for c in cands) else "continue"
    return pol


# --- evaluation on the offline dataset's state distribution -------------------
def active_steps(data: OfflineDataset, U: Sequence[Candidate]) -> int:
    """Count of dataset steps where at least one interpretation in U fires --
    the rule-relevant subpopulation. Used to normalize semantic cost so the
    signal is not diluted by the (large) majority of healthy, irrelevant steps."""
    return sum(
        any(c.fires(traj, t) for c in U)
        for traj in data.trajectories
        for t in range(len(traj))
    )

def policy_costs(
    policy: Policy,
    data: OfflineDataset,
    U: Sequence[Candidate],
    normalize: str = "all",
) -> Dict[str, float]:
    """Per-interpretation expected semantic cost of `policy` over D's states.

    normalize="all"    -> rate over every dataset step;
    normalize="active" -> rate over rule-relevant steps (some psi in U fires).
    """
    denom = active_steps(data, U) if normalize == "active" else data.n_steps()
    out = {c.name: 0.0 for c in U}
    if denom == 0:
        return out
    for traj in data.trajectories:
        for t in range(len(traj)):
            a = policy(traj, t)
            for c in U:
                out[c.name] += c.cost(traj, t, a)
    return {k: v / denom for k, v in out.items()}

def worst_case_cost(
    policy: Policy,
    data: OfflineDataset,
    U: Sequence[Candidate],
    normalize: str = "all",
) -> float:
    costs = policy_costs(policy, data, U, normalize=normalize)
    return max(costs.values(), default=0.0)

def union_cost(
    policy: Policy,
    data: OfflineDataset,
    U: Sequence[Candidate],
    normalize: str = "all",
) -> float:
    """Pointwise-union semantic cost: rate of (active) steps where the taken
    action is forbidden by AT LEAST ONE interpretation in U.

    This is the 'union-cost' robustification baseline. Because the union of the
    per-step indicators dominates each one, union_cost >= worst_case_cost (the
    per-interpretation max) always. A learner whose dual-ascent feasibility check
    uses this stops only once *no* interpretation is violated anywhere, which is
    strictly more conservative than honoring the audited per-interpretation
    worst case -- the quantitative source of its excess intervention (H3)."""
    denom = active_steps(data, U) if normalize == "active" else data.n_steps()
    if denom == 0:
        return 0.0
    k = 0
    for traj in data.trajectories:
        for t in range(len(traj)):
            a = policy(traj, t)
            if any(c.cost(traj, t, a) for c in U):
                k += 1
    return k / denom

def _plaus_weights(U: Sequence[Candidate]) -> np.ndarray:
    """Plausibility weights w_k proportional to ensemble-mean plausibility, used by
    the mixture / quantile aggregation baselines. Falls back to uniform if no
    interpretation carries a plausibility score (e.g. the no-LLM ablation)."""
    p = np.array([c.plaus_mean() if c.has_plausibility() else 0.0 for c in U],
                 dtype=float)
    s = p.sum()
    if s <= 0:
        return np.full(len(U), 1.0 / max(1, len(U)))
    return p / s


def expected_cost(
    policy: Policy,
    data: OfflineDataset,
    U: Sequence[Candidate],
    normalize: str = "all",
) -> float:
    """Plausibility-weighted EXPECTED semantic cost: E_{k~w}[J_{c_k}] with weights
    w_k proportional to ensemble plausibility (the 'expected-cost mixture' baseline,
    sum_k p_k c_k). A learner whose feasibility check uses this honors the *average*
    interpretation, so it can leave the worst single interpretation violated whenever
    the high-cost reading is a minority; SA-ORL's per-interpretation max dominates it
    pointwise, which is the source of the residual violation we report (H4)."""
    denom = active_steps(data, U) if normalize == "active" else data.n_steps()
    if denom == 0 or not U:
        return 0.0
    w = _plaus_weights(U)
    tot = 0.0
    for traj in data.trajectories:
        for t in range(len(traj)):
            a = policy(traj, t)
            tot += sum(w[k] * c.cost(traj, t, a) for k, c in enumerate(U))
    return tot / denom


def majority_cost(
    policy: Policy,
    data: OfflineDataset,
    U: Sequence[Candidate],
    normalize: str = "all",
) -> float:
    """Majority-vote semantic cost: rate of (active) steps where the taken action is
    forbidden by a STRICT MAJORITY of interpretations (the 'consensus reading'
    baseline). Between expected-cost and union/worst-case: it ignores a violation
    that only a minority of readings flag, so it too can under-protect a genuine but
    minority-held interpretation that SA-ORL's max still honors."""
    denom = active_steps(data, U) if normalize == "active" else data.n_steps()
    if denom == 0 or not U:
        return 0.0
    half = len(U) / 2.0
    k = 0
    for traj in data.trajectories:
        for t in range(len(traj)):
            a = policy(traj, t)
            if sum(1 for c in U if c.cost(traj, t, a)) > half:
                k += 1
    return k / denom


def weighted_quantile_cost(
    policy: Policy,
    data: OfflineDataset,
    U: Sequence[Candidate],
    normalize: str = "all",
    q: float = 0.9,
) -> float:
    """Plausibility-weighted high-QUANTILE of the per-interpretation costs (the
    'Bayesian high-quantile' baseline): treat {J_{c_k}} as a distribution with
    plausibility weights {w_k} and return its q-quantile. This interpolates between
    the expected-cost mixture (q->0.5-ish) and the worst case (q->1); at q<1 it can
    still drop the single most-violated interpretation when that reading carries
    little plausibility mass, whereas SA-ORL keeps the full max (q=1)."""
    if not U:
        return 0.0
    costs = policy_costs(policy, data, U, normalize=normalize)
    vals = np.array([costs[c.name] for c in U], dtype=float)
    w = _plaus_weights(U)
    order = np.argsort(vals)
    vals, w = vals[order], w[order]
    cum = np.cumsum(w)
    idx = int(np.searchsorted(cum, q * cum[-1], side="left"))
    idx = min(idx, len(vals) - 1)
    return float(vals[idx])


def intervention_rate(policy: Policy, data: OfflineDataset) -> float:
    n = data.n_steps()
    if n == 0:
        return 0.0
    k = sum(
        policy(traj, t) != "continue"
        for traj in data.trajectories
        for t in range(len(traj))
    )
    return k / n

def semantic_gap(
    data: OfflineDataset, U: Sequence[Candidate], normalize: str = "active"
) -> float:
    """Worst-case cost left by respecting only the first retained interpretation
    minus that of respecting all of U_alpha. Zero when |U_alpha| == 1 (no
    ambiguity); positive when interpretations genuinely disagree -- the
    policy-level signature of H4. Normalized over active states by default."""
    if not U:
        return 0.0
    first_only = worst_case_cost(respect_policy(U[:1]), data, U, normalize=normalize)
    all_of = worst_case_cost(respect_policy(U), data, U, normalize=normalize)
    return first_only - all_of


# --- ground-truth return (sim evaluation of a target policy) -----------------
def evaluate_return(
    mdp: MaintenanceMDP, policy: Policy, n_episodes: int = 40, seed: int = 123
) -> float:
    """Mean episodic return of a target policy rolled out in the MDP."""
    rng = np.random.default_rng(seed)
    totals = []
    for _ in range(n_episodes):
        rul = mdp.initial_rul(rng)
        traj, total = [], 0.0
        for t in range(mdp.horizon):
            traj.append(mdp.observe(rul, rng))
            a = policy(traj, t)
            r, rul, _ = mdp.step(rul, a)
            total += r
        totals.append(total)
    return float(np.mean(totals))


# --- risk-sensitive PRIMARY safety metrics (plan section 2.9) ----------------
# max_k J_ck (worst_case_cost above) is the *training surrogate*: an expected,
# dataset-state-averaged per-step rate. Safety-critical deployment cares about
# realized outcomes per episode, so the primary metrics are the chance that ANY
# audited interpretation is violated, Pr(max_k Z_k = 1), and the tail of the
# audited worst-case episodic cost, CVaR_beta(max_k C_k). Both are taken over the
# full retained set U_alpha, on rollouts in the domain's TRUE environment.
def episode_semantic_costs(
    traj: Sequence[State],
    actions: Sequence[str],
    U: Sequence[Candidate],
    gamma: float = 1.0,
):
    """Per-interpretation episodic semantic cost for one realized rollout.

    Returns (C, Z) as numpy arrays indexed by interpretation k:
      C[k] = sum_t gamma^t * c_k(s_t, a_t)   -- discounted episodic cost C_k(tau)
      Z[k] = 1{ sum_t c_k(s_t, a_t) > 0 }    -- episode violation indicator Z_k(tau)
    where c_k(s_t, a_t) = Candidate.cost is the per-step semantic cost. The
    audited worst case over k is formed by the caller (see `rollout_risk`)."""
    K = len(U)
    if K == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    disc = np.zeros(K)
    raw = np.zeros(K)
    for t, a in enumerate(actions):
        g = gamma ** t
        for k, c in enumerate(U):
            ck = c.cost(traj, t, a)
            if ck:
                disc[k] += g * ck
                raw[k] += ck
    return disc, raw > 0


def risk_metrics(
    episode_worst: Sequence[float],
    episode_violated: Sequence[bool],
    beta: float = 0.1,
) -> Dict[str, float]:
    """Aggregate per-episode audited worst cases into the primary safety metrics.

      chance     = Pr(max_k Z_k = 1): fraction of episodes in which AT LEAST ONE
                   retained interpretation is violated (the chance constraint);
      cvar       = CVaR_beta(max_k C_k): mean of the worst beta-fraction of the
                   episodic worst-case costs (tail risk);
      mean_worst = E[max_k C_k]: average episodic audited worst-case cost.
    """
    w = np.asarray(episode_worst, dtype=float)
    z = np.asarray(episode_violated, dtype=float)
    n = w.size
    if n == 0:
        return dict(chance=0.0, cvar=0.0, mean_worst=0.0)
    m = max(1, int(np.ceil(beta * n)))           # size of the worst beta-tail
    cvar = float(np.sort(w)[-m:].mean())
    return dict(chance=float(z.mean()), cvar=cvar, mean_worst=float(w.mean()))


def rollout_risk(
    rollout_once: Callable,
    policy: Policy,
    U: Sequence[Candidate],
    n_episodes: int = 300,
    seed: int = 321,
    gamma: float = 1.0,
    beta: float = 0.1,
) -> Dict[str, float]:
    """Monte-Carlo estimate of the primary safety metrics for `policy` over U.

    `rollout_once(policy, rng) -> (traj, actions)` plays ONE episode in the
    domain's true environment and returns the realized trajectory and the actions
    taken; this keeps the estimator domain-agnostic (each domain supplies its own
    one-episode rollout). For each episode we form the audited worst case over U,
    max_k C_k and max_k Z_k, then aggregate with `risk_metrics`."""
    if not U:
        return dict(chance=0.0, cvar=0.0, mean_worst=0.0)
    rng = np.random.default_rng(seed)
    worst: List[float] = []
    violated: List[bool] = []
    for _ in range(n_episodes):
        traj, actions = rollout_once(policy, rng)
        C, Z = episode_semantic_costs(traj, actions, U, gamma=gamma)
        worst.append(float(C.max()) if C.size else 0.0)
        violated.append(bool(Z.any()) if Z.size else False)
    return risk_metrics(worst, violated, beta=beta)
