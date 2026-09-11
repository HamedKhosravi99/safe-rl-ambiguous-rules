"""Paper-grade experiment harness: multi-seed, multi-model, with statistics.

This is the infrastructure that turns the kill-test proof-of-concept into the
experimental table. It runs a configurable matrix and serializes raw results so a
cluster job can crank the seed count without changing code:

  domains   x  model classes              x  constraint mode       x  seeds
  --------     -----------------------       -----------------        -----
  synthetic    fqi  = tabular BCQ-FQI         unconstrained (greedy)   0..N-1
  real C-MAPSS cql  = neural CQL (raw feats)  single-interp (most plausible psi)
  gridworld    (gridworld: tabular only)      SA-ORL        (honor U)

Domains 1's two variants (synthetic maintenance + real C-MAPSS replay) carry the
threshold ambiguity; the gridworld is Domain 2, the genuinely temporal warning-
window ambiguity (offset-indexed tabular FQI). Each domain supplies its own
unconstrained baseline and the model classes that apply to it (neural CQL is a
raw-continuous-feature learner, so it does not apply to the discrete offset MDP).

For every (domain, model, mode, seed) it logs return / honored cost / TRUE
worst-case cost over U_alpha. It then reports, per (domain, model):
  * mean +/- 95% t-CI for each metric,
  * the paired single-vs-SA-ORL gap in true worst-case cost with a Wilcoxon
    signed-rank test across seeds (the kill-test effect, as a statistic).

Raw per-seed records + the summary are written to results/<run>.json so runs
accumulate and can be re-aggregated offline.

Run:
  python3 -m saorl.experiments --seeds 10                  # all domains, both models
  python3 -m saorl.experiments --seeds 30 --domains gridworld --learners fqi
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from .build_plaus_cache import RULE_SETS
from .construct import construct_U_alpha
from .env import MaintenanceMDP
from .judge import attach_plausibility, load_llm_cache
from .offline import (
    evaluate_return,
    expected_cost,
    greedy_continue_policy,
    majority_cost,
    make_offline_rl_dataset,
    rollout_risk,
    union_cost,
    weighted_quantile_cost,
    worst_case_cost,
)
from .offline_rl import learn_fqi_constrained
from .neural_rl import learn_cql_constrained
from .sota_learners import learn_cpq_constrained, learn_pid_lagrangian_constrained

TENSION = MaintenanceMDP(r_op=4.0, c_fail=20.0, c_replace=40.0)
EPS = 0.05
LAM_GRID = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0)

# model classes. fqi/cql vary the function approximation under ONE constraint
# mechanism (Lagrangian shaped reward + dual ascent); cpq/pid additionally vary the
# constraint MECHANISM itself (CPQ cost-critic + safe-action restriction; PID dual),
# so the single-vs-SA-ORL kill test spans how offline-safe RL actually enforces cost.
LEARNERS: Dict[str, Callable] = {
    "fqi": learn_fqi_constrained,             # tabular BCQ-FQI, Lagrangian shaped reward
    "cql": learn_cql_constrained,             # neural CQL, Lagrangian shaped reward
    "cpq": learn_cpq_constrained,             # CPQ: cost-critic + safe-action restriction
    "pid": learn_pid_lagrangian_constrained,  # PID-Lagrangian dual on shaped reward
}
LEARNER_LABEL = {
    "fqi": "tabular BCQ-FQI", "cql": "neural CQL",
    "cpq": "CPQ", "pid": "PID-Lagrangian",
}


@dataclass
class Record:
    domain: str
    model: str       # unconstrained | <learner>:{single,saorl,raw,union}
    seed: int
    ret: float
    honored: Optional[float]
    true_worst: float
    lam: Optional[float]
    # primary safety metrics over U_alpha (plan section 2.9), true-env rollouts
    chance: Optional[float] = None       # Pr(max_k Z_k = 1)
    cvar: Optional[float] = None         # CVaR_beta(max_k C_k)
    mean_worst: Optional[float] = None   # E[max_k C_k]


# --- domain specs ------------------------------------------------------------
# A domain supplies: build(seed) -> (data, env, U_alpha, return_fn); a greedy
# baseline policy; and the model classes that apply to it. Keeping these per
# domain lets Domain 2 (no learner env, learned greedy baseline, tabular-only)
# share the same multi-seed/statistics machinery as Domain 1.
@dataclass
class DomainSpec:
    build: Callable                # seed -> (data, env, U, return_fn, raw_pool)
    greedy: Callable               # (data, env, U, return_fn) -> Policy
    learners: Dict[str, Callable]  # applicable model classes for this domain
    rollout: Callable              # env -> (rollout_once: (policy, rng) -> (traj, actions))
    # raw_pool (the full un-audited candidate set, 5th build output) is None when
    # baselines are uninformative -- e.g. the gridworld retains its whole pool, so
    # raw-template == union == SA-ORL by construction. `rollout` supplies the
    # domain's true-env one-episode rollout used to estimate the primary safety
    # metrics (Pr(violation), CVaR) -- see saorl.offline.rollout_risk.


def _cons_cands():
    raw = [c for c, _ in RULE_SETS["conservative"]["items"]]
    return attach_plausibility(raw, load_llm_cache("conservative"))


def most_plausible(U: Sequence) -> list:
    """The single most plausible retained interpretation: argmax mean ensemble
    plausibility over U_alpha. This is the principled, order-independent choice for
    the single-translation baseline -- the most *faithful* reading a practitioner
    would commit to, not an arbitrary input-order pick. Returns a 1-element list so
    it slots in wherever ``U[:1]`` was used. (Empirically this coincides with
    ``U[:1]`` in every domain here, so the headline numbers are unchanged; the point
    is to make the baseline robust to the rejected 'you cherry-picked the order'
    objection -- even the most plausible single reading hides the gap.)"""
    if not U:
        return list(U)
    def mean_plaus(c) -> float:
        p = getattr(c, "plausibility", None)
        return float(np.mean(p)) if p else 0.0
    return [max(U, key=mean_plaus)]


def widest_firing(U: Sequence, data) -> list:
    """The single MOST CONSERVATIVE retained reading = the one whose firing region
    is widest over the data (the strong single-translation baseline a careful
    practitioner would pick if forced to commit to one rule). Under our pre-
    registered intended-reading rule (omit_intended._intended_idx, 'most protective
    reading'), this single pick *coincides* with the oracle/ask-clarification
    interpretation psi-dagger -- so a column for it doubles as the KnowNo-style
    clarification oracle. The point of the kill test is that even this strongest
    single reading under-protects: another retained interpretation fires on steps it
    misses, so SA-ORL's per-interpretation max still removes violations it leaves."""
    if not U:
        return list(U)
    def fcount(c) -> int:
        return sum(c.fires(traj, t)
                   for traj in data.trajectories for t in range(len(traj)))
    return [max(U, key=fcount)]


def _maint_greedy(data, env, U, ret_fn):
    return greedy_continue_policy()


# --- per-domain one-episode rollouts (for the primary safety metrics) ---------
# Each returns rollout_once(policy, rng) -> (traj, actions): one episode in the
# domain's TRUE environment, mirroring its evaluate_*_return but recording the
# realized states and actions so rollout_risk can score the audited cost over U.
def _maint_rollout(env):
    def rollout_once(policy, rng):
        rul = env.initial_rul(rng)
        traj, acts = [], []
        for t in range(env.horizon):
            traj.append(env.observe(rul, rng))
            a = policy(traj, t)
            acts.append(a)
            _, rul, _ = env.step(rul, a)
        return traj, acts
    return rollout_once


def _replay_rollout(env):
    def rollout_once(policy, rng):
        st = env.reset(rng)
        traj, acts = [], []
        for t in range(env.horizon):
            traj.append(env.observe(st))
            a = policy(traj, t)
            acts.append(a)
            _, st, _ = env.step(st, a, rng)
        return traj, acts
    return rollout_once


def _gw_rollout(env):
    def rollout_once(policy, rng):
        state = env.initial(rng)
        traj, acts = [], []
        for t in range(env.horizon):
            latent, obs = env.signals(state, rng)
            traj.append(dict(pos=state["pos"], **obs))
            a = policy(traj, t)
            acts.append(a)
            _, pos = env.step(state, a, obs)
            state = dict(pos=pos, dleft=latent["dleft"], cool=latent["cool"])
        return traj, acts
    return rollout_once


def _budget_rollout(env):
    def rollout_once(policy, rng):
        state = env.initial(rng)
        traj, acts = [], []
        for t in range(env.horizon):
            traj.append(env.observe(state))
            a = policy(traj, t)
            acts.append(a)
            _, state = env.step(state, a)
        return traj, acts
    return rollout_once


def _synthetic_seed(seed: int):
    mdp = TENSION
    data = make_offline_rl_dataset(mdp, seed=seed)
    pool = _cons_cands()
    U = construct_U_alpha(pool, data.to_semantic_dataset()).U_alpha
    return data, mdp, U, (lambda pol: evaluate_return(mdp, pol, n_episodes=40)), pool


def _synthetic_spec(learners: Sequence[str]) -> DomainSpec:
    avail = {k: LEARNERS[k] for k in learners if k in LEARNERS}
    return DomainSpec(_synthetic_seed, _maint_greedy, avail, _maint_rollout)


def _real_spec(learners: Sequence[str], subset: str = "FD001") -> DomainSpec:
    from .cmapss import load_features, to_trajectories
    from .cmapss_env import ReplayMDP, evaluate_replay_return, make_replay_rl_dataset
    from .dataset import SemanticDataset

    # subset selects the C-MAPSS difficulty tier (FD001..FD004). The predicate
    # meanings (and therefore the frozen plausibility cache) transfer unchanged
    # because every subset exposes the same rul/anom keys in the same units, so
    # the SAME audited candidate pool is scored against each subset's degradation.
    trajs = to_trajectories(load_features(subset))
    env = ReplayMDP(trajs)
    pool = _cons_cands()
    U_fixed = construct_U_alpha(pool, SemanticDataset(trajs)).U_alpha

    def build(seed: int):
        data = make_replay_rl_dataset(env, seed=seed)
        return (data, env, U_fixed,
                (lambda pol: evaluate_replay_return(env, pol, n_episodes=40)), pool)

    avail = {k: LEARNERS[k] for k in learners if k in LEARNERS}
    return DomainSpec(build, _maint_greedy, avail, _replay_rollout)


def _gridworld_spec(learners: Sequence[str]) -> DomainSpec:
    from .gridworld import (
        CrossingGridworld,
        RULE_SETS_T,
        gridworld_semantic_dataset,
        make_gridworld_dataset,
    )
    from .gridworld_rl import evaluate_gridworld_return, learn_gw_constrained

    cache = str(Path(__file__).with_name("plausibility_cache_temporal.json"))
    if not Path(cache).exists():
        raise FileNotFoundError(cache)
    raw = [c for c, _ in RULE_SETS_T["window_conservative"]["items"]]
    judge = load_llm_cache("window_conservative", cache)

    def build(seed: int):
        env = CrossingGridworld(c_accident=2.0)  # calibrated: tail is return-tempting
        data = make_gridworld_dataset(env, seed=seed)
        dsem = gridworld_semantic_dataset(data)
        U = construct_U_alpha(attach_plausibility(raw, judge), dsem).U_alpha
        # raw_pool=None: the gridworld retains its full pool (no rejected
        # interpretation), so raw-template/union baselines equal SA-ORL trivially.
        return data, env, U, (lambda pol: evaluate_gridworld_return(env, pol, n_episodes=40)), None

    def greedy(data, env, U, ret_fn):
        return learn_gw_constrained(data, honor=[], U_eval=U, eps=EPS,
                                    return_fn=ret_fn, lam_grid=(0.0,)).policy

    def gw_learner(data, env, honor, U_eval, eps, return_fn, **kw):
        # offset-indexed tabular FQI takes no learner env; absorb it for a uniform call.
        return learn_gw_constrained(data, honor=honor, U_eval=U_eval, eps=eps,
                                    return_fn=return_fn, **kw)

    # tabular only: neural CQL is a raw-continuous-feature learner, N/A on the offset MDP.
    return DomainSpec(build, greedy, {"fqi": gw_learner} if "fqi" in learners else {},
                      _gw_rollout)


def _budget_spec(learners: Sequence[str]) -> DomainSpec:
    from .budget import (
        BudgetAgentMDP,
        RULE_SETS_B,
        budget_semantic_dataset,
        make_budget_dataset,
    )
    from .budget_rl import evaluate_budget_return, learn_budget_constrained

    cache = str(Path(__file__).with_name("plausibility_cache_budget.json"))
    if not Path(cache).exists():
        raise FileNotFoundError(cache)
    # Use the full ambiguous cap pool ($20..$80) as the raw set and let the frozen
    # construction trim it: the data gates discard $20 (overblocks clearly-fine
    # spend) and $80 (never binds), and the LANGUAGE ensemble decides which of the
    # surviving caps are faithful to "keep spending modest". Less hand-curation
    # than a pre-trimmed set, and it exhibits the construction actually filtering.
    raw = [c for c, _ in RULE_SETS_B["budget_ambiguous"]["items"]]
    judge = load_llm_cache("budget_ambiguous", cache)

    def build(seed: int):
        env = BudgetAgentMDP()
        data = make_budget_dataset(env, seed=seed)
        dsem = budget_semantic_dataset(env, data)
        U = construct_U_alpha(attach_plausibility(raw, judge), dsem).U_alpha
        # raw_pool=None: the rejected caps ($20/$80) are absurd readings, not a
        # competing "un-audited template", so the raw-template/union H3 baselines
        # are uninformative here -- the kill test is single-reading vs SA-ORL.
        return (data, env, U,
                (lambda pol: evaluate_budget_return(env, pol, n_episodes=40)), None)

    def greedy(data, env, U, ret_fn):
        return learn_budget_constrained(data, honor=[], U_eval=U, eps=EPS,
                                        return_fn=ret_fn, lam_grid=(0.0,)).policy

    def budget_learner(data, env, honor, U_eval, eps, return_fn, **kw):
        # bill-bucket tabular FQI takes no learner env; absorb it for a uniform call.
        return learn_budget_constrained(data, honor=honor, U_eval=U_eval, eps=eps,
                                        return_fn=return_fn, **kw)

    # tabular only: neural CQL is a raw-continuous-feature learner, N/A on the bill MDP.
    return DomainSpec(build, greedy,
                      {"fqi": budget_learner} if "fqi" in learners else {},
                      _budget_rollout)


def _domain_specs(domains: Sequence[str], learners: Sequence[str]) -> Dict[str, DomainSpec]:
    out: Dict[str, DomainSpec] = {}
    for d in domains:
        if d == "synthetic":
            out[d] = _synthetic_spec(learners)
        elif d == "real" or d.startswith("real_"):
            # "real" -> FD001 (back-compat); "real_FD002".."real_FD004" -> that
            # subset (reviewer #4: generalize the kill test across C-MAPSS tiers).
            subset = d.split("_", 1)[1] if "_" in d else "FD001"
            try:
                out[d] = _real_spec(learners, subset)
            except (FileNotFoundError, ImportError) as exc:
                print(f"  (real C-MAPSS {subset} skipped: {exc})")
        elif d == "gridworld":
            try:
                out[d] = _gridworld_spec(learners)
            except (FileNotFoundError, ImportError):
                print("  (gridworld domain skipped: temporal plausibility cache not built)")
        elif d == "budget":
            try:
                out[d] = _budget_spec(learners)
            except (FileNotFoundError, ImportError):
                print("  (budget domain skipped: budget plausibility cache not built)")
    return out


def _learner_kwargs(name: str, seed: int, device: str) -> dict:
    if name == "cql":
        return dict(seed=seed, device=device)
    if name in ("cpq", "pid"):
        return dict(bcq_tau=0.05)            # distinct mechanism: no lambda grid to sweep
    return dict(bcq_tau=0.05, lam_grid=LAM_GRID)  # fqi


# --- run ---------------------------------------------------------------------
def run(
    seeds: Sequence[int],
    domains: Sequence[str],
    learners: Sequence[str],
    eps: float = EPS,
    device: str = "cpu",
    extra_baselines: bool = False,
    beta: float = 0.1,
) -> List[Record]:
    specs = _domain_specs(domains, learners)
    records: List[Record] = []
    for dname, spec in specs.items():
        for seed in seeds:
            data, env, U, ret_fn, raw_pool = spec.build(seed)
            rollout_once = spec.rollout(env)
            # CVaR tail level beta is swept for the risk Pareto (reviewer #8); the
            # chance/mean-worst metrics are beta-invariant, only CVaR_beta moves.
            risk = lambda pol: rollout_risk(rollout_once, pol, U, beta=beta)  # over U_alpha
            g = spec.greedy(data, env, U, ret_fn)
            gr = risk(g)
            records.append(Record(
                dname, "unconstrained", seed, ret_fn(g), None,
                worst_case_cost(g, data, U, normalize="active"), None,
                gr["chance"], gr["cvar"], gr["mean_worst"],
            ))
            n_new = 1
            for lname, fn in spec.learners.items():
                kw = _learner_kwargs(lname, seed, device)
                single = fn(data, env, honor=most_plausible(U), U_eval=U, eps=eps,
                            return_fn=ret_fn, **kw)
                robust = fn(data, env, honor=U, U_eval=U, eps=eps, return_fn=ret_fn, **kw)
                sr, rr = risk(single.policy), risk(robust.policy)
                records.append(Record(dname, f"{lname}:single", seed, single.ret,
                                      single.honored_cost, single.true_worst, single.lam,
                                      sr["chance"], sr["cvar"], sr["mean_worst"]))
                records.append(Record(dname, f"{lname}:saorl", seed, robust.ret,
                                      robust.honored_cost, robust.true_worst, robust.lam,
                                      rr["chance"], rr["cvar"], rr["mean_worst"]))
                n_new += 2
                # H3 conservatism baselines (tabular FQI only, where raw_pool is
                # informative): raw-template honors the un-audited pool; union-cost
                # honors U_alpha but stops dual ascent on the pointwise-union cost.
                if raw_pool is not None and lname == "fqi":
                    raw = fn(data, env, honor=raw_pool, U_eval=U, eps=eps,
                             return_fn=ret_fn, **kw)
                    union = fn(data, env, honor=U, U_eval=U, eps=eps, return_fn=ret_fn,
                               feas_fn=lambda p, d, h: union_cost(p, d, h, normalize="active"),
                               **kw)
                    rawr, unir = risk(raw.policy), risk(union.policy)
                    records.append(Record(dname, f"{lname}:raw", seed, raw.ret,
                                          raw.honored_cost, raw.true_worst, raw.lam,
                                          rawr["chance"], rawr["cvar"], rawr["mean_worst"]))
                    records.append(Record(dname, f"{lname}:union", seed, union.ret,
                                          union.honored_cost, union.true_worst, union.lam,
                                          unir["chance"], unir["cvar"], unir["mean_worst"]))
                    n_new += 2
                # Reviewer-requested STRONGER single-reading / alternative-aggregation
                # baselines (fqi only, where cost is exact): the kill test must beat
                # more than an arbitrary single pick. Each honors U_alpha but collapses
                # it differently -- conservative=widest single reading (== oracle
                # intended pick), majority=consensus, expected=plausibility-weighted
                # mean, quantile=plausibility-weighted high quantile -- and SA-ORL's
                # per-interpretation max should still strictly reduce realized violation.
                if extra_baselines and raw_pool is not None and lname == "fqi":
                    variants = [
                        ("conservative", dict(honor=widest_firing(U, data))),
                        ("majority", dict(honor=U, feas_fn=lambda p, d, h:
                                          majority_cost(p, d, h, normalize="active"))),
                        ("expected", dict(honor=U, feas_fn=lambda p, d, h:
                                          expected_cost(p, d, h, normalize="active"))),
                        ("quantile", dict(honor=U, feas_fn=lambda p, d, h:
                                          weighted_quantile_cost(p, d, h, normalize="active", q=0.9))),
                    ]
                    for mname, extra in variants:
                        b = fn(data, env, U_eval=U, eps=eps, return_fn=ret_fn, **extra, **kw)
                        br = risk(b.policy)
                        records.append(Record(dname, f"{lname}:{mname}", seed, b.ret,
                                              b.honored_cost, b.true_worst, b.lam,
                                              br["chance"], br["cvar"], br["mean_worst"]))
                        n_new += 1
            print(f"  ran {dname} seed={seed}: "
                  + ", ".join(f"{r.model}:tw={r.true_worst:.3f}"
                              for r in records[-n_new:]))
    return records


# --- statistics --------------------------------------------------------------
def _ci95(xs: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    n = len(a)
    m = float(a.mean()) if n else 0.0
    if n < 2:
        return m, 0.0
    se = a.std(ddof=1) / np.sqrt(n)
    h = float(stats.t.ppf(0.975, n - 1) * se)
    return m, h


def summarize(records: List[Record], eps: float, learners: Sequence[str]) -> dict:
    domains = sorted({r.domain for r in records})
    summary: dict = {"per_model": {}, "paired": {}}
    for dom in domains:
        drecs = [r for r in records if r.domain == dom]
        # Enumerate models in a stable order but include whatever modes are present
        # (single/saorl always; raw/union only where the H3 baselines were run).
        order = ["unconstrained"] + [
            f"{l}:{m}" for l in learners for m in
            ("single", "saorl", "raw", "union",
             "conservative", "majority", "expected", "quantile")
        ]
        present = {r.model for r in drecs}
        models = [m for m in order if m in present]
        print(f"\n  === {dom}  (n={len({r.seed for r in drecs})} seeds) ===")
        print(f"  {'model':20s} {'return':>12s} {'TRUE worst*':>14s} "
              f"{'Pr(viol)':>10s} {'CVaR_b':>10s}")
        for mdl in models:
            rs = [r for r in drecs if r.model == mdl]
            if not rs:
                continue
            rm, rh = _ci95([r.ret for r in rs])
            hh = [r.honored for r in rs if r.honored is not None]
            hm, hci = _ci95(hh) if hh else (None, None)
            tm, tci = _ci95([r.true_worst for r in rs])
            ch = [r.chance for r in rs if r.chance is not None]
            cv = [r.cvar for r in rs if r.cvar is not None]
            cm, cci = _ci95(ch) if ch else (None, None)
            vm, vci = _ci95(cv) if cv else (None, None)
            cs = f"{cm:.3f}" if cm is not None else "--"
            vs = f"{vm:.3f}" if vm is not None else "--"
            print(f"  {mdl:20s} {rm:6.1f}+/-{rh:4.1f} {tm:6.3f}+/-{tci:5.3f} "
                  f"{cs:>10s} {vs:>10s}")
            summary["per_model"][f"{dom}/{mdl}"] = dict(
                ret=(rm, rh), honored=(hm, hci), true_worst=(tm, tci),
                chance=(cm, cci), cvar=(vm, vci), n=len(rs))
        # paired kill-test effect per learner: single vs saorl true worst by seed
        for l in learners:
            byseed = {r.seed: r for r in drecs if r.model == f"{l}:single"}
            byseed2 = {r.seed: r for r in drecs if r.model == f"{l}:saorl"}
            seeds_c = sorted(set(byseed) & set(byseed2))
            if not seeds_c:
                continue
            s_tw = [byseed[s].true_worst for s in seeds_c]
            r_tw = [byseed2[s].true_worst for s in seeds_c]
            diff = [a - b for a, b in zip(s_tw, r_tw)]
            dm, dci = _ci95(diff)
            try:
                w = stats.wilcoxon(s_tw, r_tw, alternative="greater")
                pval = float(w.pvalue)
            except ValueError:
                pval = float("nan")  # e.g. all-identical (degenerate)
            n_sep = sum(a > eps and b <= eps for a, b in zip(s_tw, r_tw))
            print(f"    [{LEARNER_LABEL[l]}] kill-test gap (single-SA-ORL true worst) "
                  f"= {dm:+.3f}+/-{dci:.3f}, Wilcoxon p={pval:.3g}, "
                  f"separates on {n_sep}/{len(seeds_c)} seeds")
            entry = dict(gap=(dm, dci), wilcoxon_p=pval,
                         n_separate=n_sep, n=len(seeds_c))
            # primary-metric reduction: realized Pr(violation) and CVaR over U,
            # single-interpretation policy minus SA-ORL's, paired by seed.
            ch_ok = [s for s in seeds_c
                     if byseed[s].chance is not None and byseed2[s].chance is not None]
            if ch_ok:
                ch_d = [byseed[s].chance - byseed2[s].chance for s in ch_ok]
                cv_d = [byseed[s].cvar - byseed2[s].cvar for s in ch_ok]
                chm, chci = _ci95(ch_d)
                cvm, cvci = _ci95(cv_d)
                s_ch_m, _ = _ci95([byseed[s].chance for s in ch_ok])
                r_ch_m, _ = _ci95([byseed2[s].chance for s in ch_ok])
                print(f"      primary safety, single->SA-ORL: Pr(viol) "
                      f"{s_ch_m:.3f}->{r_ch_m:.3f} (drop {chm:+.3f}+/-{chci:.3f}), "
                      f"CVaR drop {cvm:+.3f}+/-{cvci:.3f}")
                entry["chance"] = dict(single=s_ch_m, saorl=r_ch_m, drop=(chm, chci))
                entry["cvar_drop"] = (cvm, cvci)
            summary["paired"][f"{dom}/{l}"] = entry
        # Reviewer-requested kill test against STRONGER single-reading / alternative-
        # aggregation baselines: each should still leave higher realized worst-case
        # cost than SA-ORL's per-interpretation max. Paired by seed, same statistic.
        for base in ("conservative", "majority", "expected", "quantile"):
            for l in learners:
                sa = {r.seed: r for r in drecs if r.model == f"{l}:saorl"}
                bb = {r.seed: r for r in drecs if r.model == f"{l}:{base}"}
                seeds_c = sorted(set(sa) & set(bb))
                if not seeds_c:
                    continue
                b_tw = [bb[s].true_worst for s in seeds_c]
                r_tw = [sa[s].true_worst for s in seeds_c]
                dm, dci = _ci95([a - b for a, b in zip(b_tw, r_tw)])
                try:
                    pval = float(stats.wilcoxon(b_tw, r_tw, alternative="greater").pvalue)
                except ValueError:
                    pval = float("nan")
                n_sep = sum(a > eps and b <= eps for a, b in zip(b_tw, r_tw))
                ch_ok = [s for s in seeds_c
                         if bb[s].chance is not None and sa[s].chance is not None]
                drop = (None, None)
                if ch_ok:
                    drop = _ci95([bb[s].chance - sa[s].chance for s in ch_ok])
                print(f"    [{LEARNER_LABEL[l]}] vs {base} (true worst) "
                      f"= {dm:+.3f}+/-{dci:.3f}, Wilcoxon p={pval:.3g}, "
                      f"separates {n_sep}/{len(seeds_c)}, Pr(viol) drop {drop[0]}")
                summary["paired"][f"{dom}/{l}/kt_{base}"] = dict(
                    gap=(dm, dci), wilcoxon_p=pval, n_separate=n_sep,
                    chance_drop=drop, n=len(seeds_c))
        # H3 conservatism: paired RETURN gain of SA-ORL over each robustification
        # baseline, at matched safety (both keep true worst <= eps). Positive =>
        # SA-ORL is less conservative (higher return) than the baseline.
        for l in learners:
            sa = {r.seed: r for r in drecs if r.model == f"{l}:saorl"}
            for base in ("raw", "union"):
                bb = {r.seed: r for r in drecs if r.model == f"{l}:{base}"}
                seeds_c = sorted(set(sa) & set(bb))
                if not seeds_c:
                    continue
                sa_ret = [sa[s].ret for s in seeds_c]
                bs_ret = [bb[s].ret for s in seeds_c]
                gain = [a - b for a, b in zip(sa_ret, bs_ret)]
                gm, gci = _ci95(gain)
                try:
                    w = stats.wilcoxon(sa_ret, bs_ret, alternative="greater")
                    pval = float(w.pvalue)
                except ValueError:
                    pval = float("nan")  # all-identical (baseline degenerate)
                both_safe = sum(sa[s].true_worst <= eps and bb[s].true_worst <= eps
                                for s in seeds_c)
                print(f"    [{LEARNER_LABEL[l]}] H3 return gain over {base} "
                      f"= {gm:+.2f}+/-{gci:.2f}, Wilcoxon p={pval:.3g}, "
                      f"both-safe {both_safe}/{len(seeds_c)}")
                summary["paired"][f"{dom}/{l}/h3_{base}"] = dict(
                    ret_gain=(gm, gci), wilcoxon_p=pval,
                    both_safe=both_safe, n=len(seeds_c))
    return summary


def main():
    ap = argparse.ArgumentParser(description="SA-ORL multi-seed, multi-model benchmark")
    ap.add_argument("--seeds", type=int, default=10, help="number of seeds (0..N-1)")
    ap.add_argument("--seed-list", type=str, default=None,
                    help="explicit comma list of seeds (overrides --seeds); lets a "
                         "SLURM array shard one seed per element and merge offline")
    ap.add_argument("--domains", type=str, default="synthetic,real,gridworld",
                    help="comma list of: synthetic, real, gridworld")
    ap.add_argument("--learners", type=str, default="fqi,cql")
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--beta", type=float, default=0.1,
                    help="CVaR tail level for the risk Pareto (#8); sweep over runs")
    ap.add_argument("--device", type=str, default="cpu", help="cpu|cuda (neural learner)")
    ap.add_argument("--extra-baselines", action="store_true",
                    help="also run the stronger single-reading / alternative-aggregation "
                         "baselines (conservative, majority, expected, quantile; fqi only)")
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    if args.seed_list:
        seeds = [int(s) for s in args.seed_list.split(",") if s.strip() != ""]
    else:
        seeds = list(range(args.seeds))
    domains = [d for d in args.domains.split(",") if d]
    learners = [l for l in args.learners.split(",") if l]
    print(f"SA-ORL experiments: seeds={seeds} domains={domains} "
          f"learners={learners} device={args.device}")
    t0 = time.time()
    records = run(seeds, domains, learners, eps=args.eps, device=args.device,
                  extra_baselines=args.extra_baselines, beta=args.beta)
    summary = summarize(records, args.eps, learners)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"experiments_{stamp}.json"
    out_path.write_text(json.dumps(dict(
        config=dict(seeds=len(seeds), seed_list=seeds, domains=domains,
                    learners=learners, eps=args.eps, beta=args.beta,
                    extra_baselines=args.extra_baselines, device=args.device),
        records=[asdict(r) for r in records],
        summary=summary,
        elapsed_s=time.time() - t0,
    ), indent=2))
    print(f"\n  wrote {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
