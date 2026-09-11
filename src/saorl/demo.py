"""End-to-end demo of the SA-ORL linchpin: ambiguity tracking (plan v12 H4).

Same frozen construction protocol (alpha=0.15, rho_inc=0.6, rho_over=0.2),
applied to two rules:

  * Rule L1 ("avoid continuing operation when degradation is severe") is
    genuinely ambiguous -- RUL / lower-quantile / anomaly / persistence
    interpretations are all language-plausible and all activate on incidents ->
    |U_alpha| > 1.

  * Rule L_sharp is near-unambiguous -- only one interpretation survives the
    frozen filters; the rest are knocked out by *different* stable mechanisms
    (language-implausibility, overblocking, low incident activation) -> |U_alpha| ~ 1.

Plausibility here is human-free: each candidate carries a 7-member language
ensemble (a cached LLM ensemble; see saorl.judge.CachedEnsembleJudge), not a
3-person survey. If the separation holds under one frozen protocol, the paper's
central non-circularity claim has a working, reproducible mechanism behind it. Run:

    python3 -m saorl.demo
"""
from __future__ import annotations

import numpy as np

from .construct import construct_U_alpha
from .dataset import SemanticDataset
from .dsl import Atom, Candidate, Persist
from .judge import CachedEnsembleJudge, attach_plausibility

FORBID = frozenset({"continue"})


def make_trajectories(n_assets=60, horizon=100, seed=0):
    """Synthetic run-to-failure trajectories with prognostic features.

    Lives are long enough that early life is genuinely healthy (RUL_true >= 80,
    the Normal regime) and end-of-life is an incident (RUL_true <= 20), so both
    incident-activation and overblocking terms are exercised. Deployed features
    (rul_hat, q05, anom) are noisy/biased functions of the true RUL, which is a
    construction-only label.
    """
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(n_assets):
        life = rng.integers(horizon - 12, horizon + 1)
        traj = []
        for t in range(horizon):
            rul_true = max(0.0, life - t)
            noise = rng.normal(0, 3.0)
            rul_hat = max(0.0, rul_true + noise)
            # lower-tail quantile sits below the point prediction
            q05 = max(0.0, rul_hat - rng.uniform(4, 8))
            # anomaly score rises as health falls: RUL_true=20 -> ~0.8
            anom = float(np.clip(1.0 - rul_true / 100.0 + rng.normal(0, 0.04), 0, 1))
            traj.append(dict(rul_true=rul_true, rul_hat=rul_hat, q05=q05, anom=anom))
        trajs.append(traj)
    return trajs


# --- frozen 7-member language ensembles (a cached LLM-ensemble audit) ---------
# Scores in [0,1] = per-member plausibility that the predicate faithfully reads
# the rule. These stand in for an offline LLM ensemble serialized to disk.
_AMBIGUOUS_ENSEMBLE = {
    # all four are reasonable readings of "degradation is severe"...
    "psi1: RUL_hat<20": (1.0, 1.0, 0.9, 1.0, 0.8, 0.9, 0.9),       # mean ~0.93
    "psi3: anom>0.8": (0.9, 1.0, 0.8, 0.9, 0.9, 0.9, 0.9),         # mean ~0.90
    # ...but Q05<10 is judged too tight a lower-quantile reading -> implausible
    "psi2: Q05<10": (0.4, 0.5, 0.3, 0.4, 0.5, 0.3, 0.4),           # mean ~0.40
    # ...and the persistence reading is only weakly endorsed
    "psi4: Persist3(RUL<25)": (0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.55),  # mean ~0.55
}
_SHARP_ENSEMBLE = {
    "phi1: RUL_hat<15": (0.9, 1.0, 0.9, 0.9, 0.9, 0.9, 0.9),       # mean ~0.91
    "phi2: anom>0.3": (0.2, 0.1, 0.2, 0.3, 0.1, 0.2, 0.2),         # mean ~0.19
    "phi3: RUL_hat<85": (0.6, 0.7, 0.5, 0.6, 0.6, 0.6, 0.6),       # mean ~0.60
    "phi4: RUL_hat<3": (0.6, 0.5, 0.6, 0.6, 0.5, 0.6, 0.6),        # mean ~0.57
}
# A *conservatively-ordered* ambiguous rule: the retained interpretations are not
# near-equivalent (as psi1/psi3 above are), but nested in strictness -- a liberal
# reading (replace only deep in the incident band) and a conservative one (act
# earlier, on a milder anomaly). A learner can satisfy the liberal reading cheaply
# while leaving a hidden violation of the conservative one. This is the substrate
# that makes the Algorithm-3 robustness claim non-vacuous (see learn_demo.py).
_CONSERVATIVE_ENSEMBLE = {
    "liberal: RUL<12": (0.9, 0.9, 0.8, 0.9, 0.9, 0.85, 0.9),       # mean ~0.88
    "mid: RUL<22": (0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9),            # mean ~0.90
    "consv: anom>0.6": (0.7, 0.75, 0.7, 0.75, 0.7, 0.75, 0.75),    # mean ~0.73
    "vconsv: RUL<40": (0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6),         # mean ~0.56
}


def ambiguous_rule_candidates():
    """L1 candidates (plan 8.3): four formalizations, language-scored by ensemble."""
    raw = [
        Candidate("psi1: RUL_hat<20", Atom("rul_hat", "le", 20), FORBID),
        Candidate("psi2: Q05<10", Atom("q05", "le", 10), FORBID),
        Candidate("psi3: anom>0.8", Atom("anom", "ge", 0.8), FORBID),
        Candidate("psi4: Persist3(RUL<25)", Persist(3, Atom("rul_hat", "le", 25)), FORBID),
    ]
    return attach_plausibility(raw, CachedEnsembleJudge(_AMBIGUOUS_ENSEMBLE))


def sharp_rule_candidates():
    """A near-unambiguous rule: only one interpretation survives the audit.

    The competitors are knocked out by *different* frozen filters -- language
    implausibility (phi2), overblocking healthy assets (phi3), or barely firing
    on incidents (phi4) -- not by tuning a single knob.
    """
    raw = [
        Candidate("phi1: RUL_hat<15", Atom("rul_hat", "le", 15), FORBID),
        Candidate("phi2: anom>0.3", Atom("anom", "ge", 0.3), FORBID),
        Candidate("phi3: RUL_hat<85", Atom("rul_hat", "le", 85), FORBID),
        Candidate("phi4: RUL_hat<3", Atom("rul_hat", "le", 3), FORBID),
    ]
    return attach_plausibility(raw, CachedEnsembleJudge(_SHARP_ENSEMBLE))


def conservative_rule_candidates():
    """A conservatively-ordered ambiguous rule (nested strictness).

    `liberal` and `vconsv` are knocked out by stable, *different* filters (low
    incident activation / outside the alpha band); the survivors `mid` (RUL<22,
    fires deep in the band) and `consv` (anom>0.6, fires earlier ~RUL<40) are a
    liberal/conservative pair. A learner honoring only `mid` can leave a hidden
    violation of `consv`, which SA-ORL removes -- see saorl.learn_demo.
    """
    raw = [
        Candidate("liberal: RUL<12", Atom("rul_hat", "le", 12), FORBID),
        Candidate("mid: RUL<22", Atom("rul_hat", "le", 22), FORBID),
        Candidate("consv: anom>0.6", Atom("anom", "ge", 0.6), FORBID),
        Candidate("vconsv: RUL<40", Atom("rul_hat", "le", 40), FORBID),
    ]
    return attach_plausibility(raw, CachedEnsembleJudge(_CONSERVATIVE_ENSEMBLE))


def _print_result(title, res):
    print(f"\n=== {title} ===")
    print(f"  L_min = {res.l_min:.3f}   |U_alpha| = {res.size()}")
    print(f"  {'candidate':24s} {'Lsem':>6s} {'pinc':>5s} {'pnorm':>6s} "
          f"{'plaus':>5s}  status")
    for r in res.reports:
        status = "KEPT" if r.retained else f"rej: {r.reason}"
        print(f"  {r.name:24s} {r.l_sem:6.3f} {r.p_inc:5.2f} {r.p_norm:6.2f} "
              f"{r.plaus:5.2f}  {status}")
    print(f"  -> U_alpha = {res.names()}")


def main():
    trajs = make_trajectories()
    dsem = SemanticDataset(trajs)

    amb = construct_U_alpha(ambiguous_rule_candidates(), dsem)
    sharp = construct_U_alpha(sharp_rule_candidates(), dsem)

    _print_result("Rule L1 (ambiguous)", amb)
    _print_result("Rule L_sharp (near-unambiguous)", sharp)

    print("\n=== Ambiguity-tracking check (H4) ===")
    print(f"  |U_alpha| ambiguous rule  = {amb.size()}  (expect > 1)")
    print(f"  |U_alpha| sharp rule      = {sharp.size()}  (expect ~ 1)")
    ok = amb.size() > 1 and sharp.size() == 1
    print(f"  separation holds: {ok}")
    return ok


if __name__ == "__main__":
    main()
