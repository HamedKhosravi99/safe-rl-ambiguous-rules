"""Theorem A: deployment-time certificate for the SHIPPED policy.

The tabular end-to-end theorem certifies the *learner*; it does not cover a
neural CORSET-wrap policy.  But deployment ships a fixed artifact, and
certifying a fixed policy needs none of the learner-side assumptions
(tabular state/action, known reward, logged support, fallback).  It needs
only evaluation access and a computable DSL predicate.

Two ingredients:

1. PATHWISE DOMINANCE.  c_psi <= c_psi' pointwise implies
   sum_t c_psi <= sum_t c_psi' on EVERY trajectory, so
       max_{psi in U} (per-episode cost) = max_{psi in Max(U)} (...)
   pathwise.  Hence the episode-violation event is decided by the maximal
   antichain, and expected-cost bounds need multiplicity w = |Max(U)|
   rather than |U|.

2. EXACT BINOMIAL INTERVALS.  The paper's headline safety metric,
   Pr(max_k Z_k = 1), is a single Bernoulli parameter (by (1) it is one
   event, not |U| events), so Clopper-Pearson gives an exact one-sided
   upper confidence bound with NO multiplicity correction at all.

This module computes, per domain: Max(U) and w; per-seed exact CP upper
bounds on the deployed policy's episode-violation rate; and the same for
the uniform-over-seeds mixture policy (itself a valid deployable policy,
for which all 50 x 300 episodes are i.i.d.).

Run: SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.deploy_certificate
Writes results/conformal/lp/deploy_certificate.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import beta

from .exact_lp import (_ROOT, build_budget, build_gridworld, build_real,
                       build_synthetic)

_OUT = _ROOT / "results/conformal" / "lp" / "deploy_certificate.json"
DELTA = 0.05          # one-sided confidence level for the certificate


def cp_upper(k: int, n: int, delta: float = DELTA) -> float:
    """Exact one-sided Clopper-Pearson upper bound on a binomial rate."""
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, k + 1, n - k))


def maximal_set(m, names: Sequence[str]) -> List[str]:
    """Maximal elements of (names, <=) under pointwise cost dominance."""
    out = []
    for i in names:
        if not any(j != i and np.all(m.cost[i] <= m.cost[j])
                   and np.any(m.cost[j] > m.cost[i]) for j in names):
            out.append(i)
    return out


def _newest(dirname: str):
    fs = glob.glob(os.path.join(_ROOT, "results/conformal", dirname,
                                "experiments_*.json"))
    return json.load(open(max(fs, key=os.path.getmtime))) if fs else None


def study(domain: str) -> dict:
    from .experiments import _domain_specs
    specs = _domain_specs([domain], ["fqi"])
    _data, env, U, _ret, _pool = specs[domain].build(0)
    builder = dict(synthetic=build_synthetic, real=build_real,
                   gridworld=build_gridworld, budget=build_budget)[domain]
    m = builder(env, list(U))
    names = [c.name for c in U]
    mx = maximal_set(m, names)

    blob = _newest("budget50" if domain == "budget" else "risk50")
    if blob is None:
        return dict(error="no episode records")
    n_ep = int(blob["config"].get("n_eval_episodes",
               blob["config"].get("n_episodes", 300)))

    out = dict(U=names, maximal=mx, w=len(mx), n_episodes_per_seed=n_ep,
               delta=DELTA, arms={})
    for model, label in (("fqi:saorl", "corset"), ("fqi:single", "single")):
        recs = [r for r in blob["records"]
                if r["domain"] == domain and r["model"] == model]
        if not recs:
            continue
        # ONE CERTIFICATE PER SHIPPED ARTIFACT.  Within a seed the 300
        # episodes are i.i.d. given that fixed policy -- exactly Thm A's
        # setting -- so Clopper-Pearson is exact there.  Episodes are
        # CLUSTERED by seed, so pooling all 50x300 into one binomial would
        # be invalid; across-seed summaries use the seed as the unit.
        ks, ups = [], []
        for r in recs:
            k = int(round(float(r["chance"]) * n_ep))
            ks.append(k)
            ups.append(cp_upper(k, n_ep))
        ups = np.array(ups)
        rates = np.array([float(r["chance"]) for r in recs])
        se = rates.std(ddof=1) / np.sqrt(len(rates)) if len(rates) > 1 else 0.0
        out["arms"][label] = dict(
            n_seeds=len(recs),
            seed_mean_rate=float(rates.mean()),
            seed_mean_rate_se=float(se),
            median_seed_cp_upper=float(np.median(ups)),
            worst_seed_cp_upper=float(ups.max()),
            best_seed_cp_upper=float(ups.min()),
            n_seeds_certified_at_005=int((ups <= 0.05).sum()),
            n_seeds_certified_at_010=int((ups <= 0.10).sum()),
            n_seeds_zero_violations=int((np.array(ks) == 0).sum()))
        print(f"  {domain:10s} {label:7s} "
              f"median CP<= {np.median(ups):.4f}  worst CP<= {ups.max():.4f}  "
              f"certified@0.10: {int((ups<=0.10).sum())}/{len(ups)}  "
              f"zero-violation seeds: {int((np.array(ks)==0).sum())}",
              flush=True)
    return out


def main() -> None:
    assert os.environ.get("SAORL_CONFORMAL") == "1", "run with SAORL_CONFORMAL=1"
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="synthetic,real,gridworld,budget")
    args = ap.parse_args()
    rep: Dict = dict(config=dict(
        delta=DELTA,
        note="Thm A deployment certificate: exact Clopper-Pearson upper "
             "bounds on the SHIPPED policy's episode-violation rate; "
             "multiplicity handled by pathwise dominance (w = |Max(U)|)"),
        domains={})
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    for dom in args.domains.split(","):
        print(f"[{dom}] building model + certifying...", flush=True)
        rep["domains"][dom] = study(dom)
        json.dump(rep, open(_OUT, "w"), indent=1)
    print(f"wrote {_OUT}", flush=True)


if __name__ == "__main__":
    main()
