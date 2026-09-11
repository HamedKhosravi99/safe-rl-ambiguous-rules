"""W5: audit of the deployment certificate Theorem 2's SHIP rule actually uses.

The proof in the paper instantiates the evaluator with a union-Hoeffding
bound over the w maximal readings, each per-episode cost Z^psi in [0, H].
The IMPLEMENTATION behind every 'ships' count (expected_cost_certificate.py,
registered D2) is a different, tighter evaluator:

    E[W] <= B * CP_{1 - delta_ev/2}(k, n),

with W = max_psi C_psi / Z_psi the per-episode worst FIXED-NORMALIZED cost
(C_psi = raw charged steps of reading psi in the episode, Z_psi the fixed
greedy activation mass of exact_lp.fixed_psi_Z), B = max_psi (almost-sure
max of C_psi by DP) / Z_psi its deterministic range, k the number of the
n = 300 evaluation episodes with ANY retained violation and CP the exact
one-sided Clopper-Pearson bound.  The justification is that {W > 0} is
exactly the violation event, so E[W] <= B Pr[W > 0]; the w maximal
readings enter through the pathwise reduction (one event, no union).

This module (i) reproduces every per-seed certificate from the archived
counts and asserts the archived ships totals, (ii) states the units and
ranges in one place, (iii) computes the sample complexity of the gate, and
(iv) re-evaluates two seeds end to end (retrain + 300 rollouts) to recover
the per-episode W the archive does not store, reporting the Hoeffding and
empirical-Bernstein routes on the same episodes.

Run: SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.certificate_audit [--no-reeval]
Writes results/conformal/lp/certificate_audit.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

import numpy as np
from scipy.stats import beta

from .alt_defense import GREEDY_ACTION
from .exact_lp import (_ROOT, build_budget, build_gridworld, build_real,
                       build_synthetic, fixed_psi_Z)
from .expected_cost_certificate import (DELTA_EV, D_BUDGET, LEVEL, _newest,
                                        cp_upper, max_cost_dp)

_OUT = _ROOT / "results/conformal" / "lp" / "certificate_audit.json"
_ARCH = _ROOT / "results/conformal" / "lp" / "expected_cost_certificate.json"
_DEPLOY = _ROOT / "results/conformal" / "lp" / "deploy_certificate.json"
DOM_LABEL = dict(synthetic="Synthetic maintenance", real="C-MAPSS replay MDP",
                 gridworld="Warning-window", budget="Budget agent")
BUILDER = dict(synthetic=build_synthetic, real=build_real,
               gridworld=build_gridworld, budget=build_budget)


def hoeffding_radius(B: float, n: int, w: int, delta: float) -> float:
    return B * math.sqrt(math.log(w / delta) / (2.0 * n))


def bernstein_bound(x: np.ndarray, B: float, w: int, delta: float) -> float:
    """Maurer-Pontil empirical Bernstein upper bound on the mean of x in [0,B],
    at confidence 1 - delta/w (union over the w maximal readings kept for
    comparability with the Hoeffding route)."""
    n = x.size
    d = delta / w
    v = float(x.var(ddof=1)) if n > 1 else 0.0
    return float(x.mean()) + math.sqrt(2.0 * v * math.log(2.0 / d) / n) \
        + 7.0 * B * math.log(2.0 / d) / (3.0 * (n - 1))


def n_zero_violation(B: float, d: float, level: float) -> int:
    """Smallest n at which k = 0 violations certifies: B (1 - (1-level)^{1/n}) <= d."""
    if d >= B:
        return 1
    return int(math.ceil(math.log(1.0 - level) / math.log(1.0 - d / B)))


def n_for_rate(B: float, d: float, p: float, level: float, nmax: int = 10_000_000) -> int | None:
    """Smallest n at which k = round(p n) violations still certify (None if never below nmax)."""
    if B * p >= d:
        return None
    lo, hi = 1, nmax
    if B * cp_upper(int(round(p * hi)), hi, level) > d:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if B * cp_upper(int(round(p * mid)), mid, level) <= d:
            hi = mid
        else:
            lo = mid + 1
    return lo


def reproduce_from_archive() -> dict:
    arch = json.load(open(_ARCH))
    dep = json.load(open(_DEPLOY))
    out = {}
    for dom, blob_name in (("synthetic", "risk50"), ("real", "risk50"),
                           ("gridworld", "risk50"), ("budget", "budget50")):
        a = arch["domains"][dom]
        blob = _newest(blob_name)
        n = a["n_episodes"]
        B = a["B"]
        w = dep["domains"][dom]["w"]
        arms = {}
        for model, label in (("fqi:saorl", "corset"), ("fqi:single", "single")):
            recs = sorted([r for r in blob["records"] if r["domain"] == dom and r["model"] == model],
                          key=lambda r: r["seed"])
            rows = []
            for r in recs:
                k = int(round(float(r["chance"]) * n))
                cp = cp_upper(k, n)
                bound = B * cp
                rows.append(dict(seed=r["seed"], k=k, n=n, p_hat=k / n, cp_upper=cp,
                                 bound=bound, ships=bool(bound <= D_BUDGET),
                                 archived_mean_worst_raw=r["mean_worst"],
                                 archived_true_worst_active_norm=r["true_worst"]))
            n_ship = sum(r["ships"] for r in rows)
            assert n_ship == a["arms"][label]["n_certified_at_budget"], (dom, label, n_ship)
            arms[label] = dict(n_seeds=len(rows), ships=n_ship, rows=rows,
                               max_k_that_ships=max([r["k"] for r in rows if r["ships"]], default=None))
        # the largest k for which B*CP(k,n) <= d, at this n
        kmax = -1
        for k in range(n + 1):
            if B * cp_upper(k, n) <= D_BUDGET:
                kmax = k
            else:
                break
        out[dom] = dict(label=DOM_LABEL[dom], B=B, w_maximal=w, n=n, level=LEVEL, delta_ev=DELTA_EV,
                        budget=D_BUDGET, per_reading=a["per_reading"],
                        k_max_certifiable_at_n=kmax, arms=arms,
                        hoeffding_radius_if_used=hoeffding_radius(B, n, w, DELTA_EV),
                        sample_complexity=dict(
                            n_zero_violation=n_zero_violation(B, D_BUDGET, LEVEL),
                            max_certifiable_violation_rate=D_BUDGET / B,
                            n_at_rate_half=n_for_rate(B, D_BUDGET, 0.5 * D_BUDGET / B, LEVEL),
                            n_at_rate_full=n_for_rate(B, D_BUDGET, D_BUDGET / B, LEVEL),
                            hoeffding_n_mean_zero=int(math.ceil((B / D_BUDGET) ** 2 * math.log(w / DELTA_EV) / 2)),
                            hoeffding_n_mean_half=int(math.ceil((B / (0.5 * D_BUDGET)) ** 2 * math.log(w / DELTA_EV) / 2)),
                            hoeffding_n_mean_full=None))
    return out


def reevaluate(domain: str, seed: int, arch_dom: dict) -> dict:
    """Retrain the set-protected FQI policy for one seed and re-roll the 300
    evaluation episodes, recovering per-episode W = max_psi C_psi / Z_psi."""
    from .experiments import _domain_specs, _learner_kwargs
    from .offline import episode_semantic_costs
    spec = _domain_specs([domain], ["fqi"])[domain]
    data, env, U, ret_fn, _raw = spec.build(seed)
    rollout_once = spec.rollout(env)
    fn = spec.learners["fqi"]
    kw = _learner_kwargs("fqi", seed, "cpu")
    robust = fn(data, env, honor=U, U_eval=U, eps=D_BUDGET, return_fn=ret_fn, **kw)
    names = [c.name for c in U]
    m = BUILDER[domain](env, list(U))
    a_greedy = m.action_names.index(GREEDY_ACTION[domain])
    Z = fixed_psi_Z(m, names, a_greedy)
    Zv = np.array([float(Z[k]) if float(Z[k]) > 0 else 1.0 for k in names])
    B = arch_dom["B"]
    rng = np.random.default_rng(321)          # rollout_risk's fixed evaluation seed
    n = 300
    W, viol, Craw = [], [], []
    for _ in range(n):
        traj, actions = rollout_once(robust.policy, rng)
        C, Zind = episode_semantic_costs(traj, actions, U, gamma=1.0)
        Craw.append(C.tolist())
        W.append(float((C / Zv).max()) if C.size else 0.0)
        viol.append(bool(Zind.any()))
    W = np.array(W)
    k = int(sum(viol))
    w = len(json.load(open(_DEPLOY))["domains"][domain]["maximal"])
    cp = cp_upper(k, n)
    return dict(domain=domain, seed=seed, horizon=int(m.horizon), n=n,
                Z={kk: float(Z[kk]) for kk in names},
                B=B, w_maximal=w,
                chance_recomputed=k / n, k=k,
                mean_W=float(W.mean()), max_W_observed=float(W.max()),
                sd_W=float(W.std(ddof=1)),
                cp_route_bound=B * cp, cp_route_ships=bool(B * cp <= D_BUDGET),
                hoeffding_bound=float(W.mean() + hoeffding_radius(B, n, w, DELTA_EV)),
                hoeffding_ships=bool(W.mean() + hoeffding_radius(B, n, w, DELTA_EV) <= D_BUDGET),
                bernstein_bound=bernstein_bound(W, B, w, DELTA_EV),
                bernstein_ships=bool(bernstein_bound(W, B, w, DELTA_EV) <= D_BUDGET),
                learned_return=float(robust.ret),
                mean_raw_cost_per_reading={kk: float(np.mean([c[i] for c in Craw])) for i, kk in enumerate(names)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-reeval", action="store_true")
    args = ap.parse_args()
    rep = dict(
        registration=("W5 certificate audit, 2026-09-02: archived per-seed counts reproduced and asserted; "
                      "two seeds re-evaluated end to end for the quantities the archive does not store"),
        evaluator_as_implemented=dict(
            statistic="W = max_psi C_psi / Z_psi per evaluation episode",
            C_psi="raw number of charged steps of reading psi in the episode (c_psi(s,a) in {0,1}); C_psi in [0, H]",
            Z_psi="fixed per-reading normalizer: expected steps the return-greedy reference policy spends in psi-firing states (exact_lp.fixed_psi_Z), set-independent",
            range="W in [0, B], B = max_psi (almost-sure max of C_psi over any policy and positive-probability transitions, by DP) / Z_psi",
            violation_event="{W > 0} = {some retained reading charges at least one step} = the episode-violation event whose rate the tables call 'violation'",
            bound="E[W] <= B * CP_{1 - delta_ev/2}(k, n): exact one-sided Clopper-Pearson on the violation rate times the deterministic range; no union over readings (the pathwise maximum is one event)",
            ship="deploy iff B * CP <= d, d = 0.05 in the normalized units above",
            delta_split="delta_ev = 0.05 registered; delta_ev/2 spent on the CP bound",
            proof_in_paper_before_this_audit="union Hoeffding over the w maximal readings with Z^psi in [0, H] -- a valid but different evaluator; the paper text is corrected to describe the one used",
            units_caveat="the learners' training-time metric normalizes by active steps of the honored set (offline.py normalize='active'); the certificate normalizes by the fixed greedy activation mass; the two agree at d = 0 and differ otherwise, so d = 0.05 in the certificate is not the learner's d"),
        domains=reproduce_from_archive())
    if not args.no_reeval:
        reev = {}
        synth = rep["domains"]["synthetic"]["arms"]["corset"]["rows"]
        seed_ship = min(r["seed"] for r in synth if r["ships"])
        for dom, seed in (("synthetic", seed_ship), ("real", 0)):
            try:
                reev[dom] = reevaluate(dom, seed, rep["domains"][dom])
                arch_chance = next(r["p_hat"] for r in rep["domains"][dom]["arms"]["corset"]["rows"] if r["seed"] == seed)
                reev[dom]["archived_chance"] = arch_chance
                reev[dom]["reproduces_archived_count"] = bool(abs(arch_chance - reev[dom]["chance_recomputed"]) < 1e-12)
                print(json.dumps({k: v for k, v in reev[dom].items() if k != "Z"}, indent=1))
            except Exception as e:  # noqa: BLE001 - report, do not hide
                reev[dom] = dict(error=repr(e), seed=seed)
                print("re-evaluation failed:", dom, repr(e))
        rep["reevaluation"] = reev
    _OUT.write_text(json.dumps(rep, indent=1))
    for dom, v in rep["domains"].items():
        print(f"{dom:>10s}: B={v['B']:.2f} w={v['w_maximal']} kmax@300={v['k_max_certifiable_at_n']} "
              f"ships set/single={v['arms']['corset']['ships']}/{v['arms']['single']['ships']} "
              f"n0={v['sample_complexity']['n_zero_violation']} pmax={v['sample_complexity']['max_certifiable_violation_rate']:.5f} "
              f"hoeff_radius={v['hoeffding_radius_if_used']:.3f}")
    print("wrote", _OUT)


if __name__ == "__main__":
    main()
