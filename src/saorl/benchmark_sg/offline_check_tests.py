"""Sanity tests for the robust Offline-CHECK evaluator (spec section 5).

Run: PYTHONPATH=. python3 -m saorl.benchmark_sg.offline_check_tests
Writes results/offline_check_experiment/evaluator_tests.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .control_mdp import GAMMA
from .offline_check import robust_eval, l1_radius, M_SUPP, HeldOut

_OUT = Path(__file__).resolve().parents[3] / "results" / "offline_check_experiment"


def exact_occupancy(P, mu0, pi):
    """Normalized discounted occupancy; identical convention to safe_face_offline."""
    nS, nA = pi.shape
    Ppi = np.einsum("sa,san->sn", pi, P)
    x_s = (1.0 - GAMMA) * np.linalg.solve(np.eye(nS) - GAMMA * Ppi.T, mu0)
    return x_s[:, None] * pi


def rand_mdp(rng, nS=4, nA=3):
    P = rng.dirichlet(np.ones(nS), size=(nS, nA))
    mu0 = rng.dirichlet(np.ones(nS))
    pi = rng.dirichlet(np.ones(nA), size=nS)
    c = rng.random((nS, nA))
    return P, mu0, pi, c


def main():
    rng = np.random.default_rng(20260909)
    res = {}

    # T1: zero radius == exact plug-in evaluation (evaluator is correct at b=0)
    err = []
    for _ in range(200):
        P, mu0, pi, c = rand_mdp(rng)
        exact = float((exact_occupancy(P, mu0, pi) * c).sum())
        got = robust_eval(P, np.zeros(pi.shape), pi, c, mu0)
        err.append(abs(exact - got))
    res["T1_zero_radius_matches_exact"] = dict(
        max_abs_err=max(err), passed=bool(max(err) < 1e-7))

    # T2: monotone in the radius (enlarging the set cannot lower the bound)
    viol = 0
    for _ in range(200):
        P, mu0, pi, c = rand_mdp(rng)
        prev = -np.inf
        for beta in (0.0, 0.05, 0.2, 0.5, 1.0, 2.0):
            v = robust_eval(P, np.full(pi.shape, beta), pi, c, mu0)
            if v < prev - 1e-12:
                viol += 1
            prev = v
    res["T2_monotone_in_radius"] = dict(violations=viol, passed=bool(viol == 0))

    # T3: SOUNDNESS. Sample a log from P*, build the set at delta, and check the
    # bound dominates the TRUE cost whenever P* is in the set. Reported both as
    # the coverage rate of the confidence event and the domination rate.
    dom_fail, cover_fail, trials = 0, 0, 300
    for _ in range(trials):
        P, mu0, pi, c = rand_mdp(rng)
        nS, nA = pi.shape
        n = int(rng.integers(50, 400))
        N = rng.multinomial(n, np.full(nS * nA, 1.0 / (nS * nA))).reshape(nS, nA)
        Phat = np.zeros_like(P)
        for s in range(nS):
            for a in range(nA):
                if N[s, a] > 0:
                    Phat[s, a] = rng.multinomial(N[s, a], P[s, a]) / N[s, a]
                else:
                    Phat[s, a, s] = 1.0
        b = l1_radius(N, nS, 0.05)
        inset = bool(np.all(np.abs(Phat - P).sum(axis=2) <= b + 1e-12))
        true_c = float((exact_occupancy(P, mu0, pi) * c).sum())
        bound = robust_eval(Phat, b, pi, c, mu0)
        if not inset:
            cover_fail += 1
        elif bound < true_c - 1e-9:
            dom_fail += 1
    res["T3_domination_when_covered"] = dict(
        trials=trials, coverage_failures=cover_fail,
        domination_failures=dom_fail, passed=bool(dom_fail == 0))

    # T4: unsupported (s,a) is maximally conservative, never optimistic.
    P, mu0, pi, c = rand_mdp(rng, nS=5, nA=3)
    N = np.full(pi.shape, 100); N[0, 0] = 0
    b = l1_radius(N, 5, 0.05)
    res["T4_unsupported_is_full_simplex"] = dict(
        radius_at_unsupported=float(b[0, 0]),
        max_supported_radius=float(b[N >= M_SUPP].max()),
        passed=bool(b[0, 0] == 2.0))

    # T5: max-over-readings — HeldOut.check reports the max of the per-cost
    # bounds computed at the SAME radius it builds internally.
    P, mu0, pi, _ = rand_mdp(rng)
    nS, nA = pi.shape
    C = rng.random((4,) + pi.shape)
    n = 600
    S = rng.integers(0, nS, n); A = rng.integers(0, nA, n)
    SP = np.array([rng.choice(nS, p=P[S[i], A[i]]) for i in range(n)])
    ho = HeldOut(dict(S=S, A=A, SP=SP, nS=nS, nA=nA, mu0=mu0))
    got = ho.check(pi, C, d=1e9)
    b_int = l1_radius(ho.N, nS, 0.05)
    per = [robust_eval(ho.Phat, b_int, pi, C[k], mu0) for k in range(4)]
    res["T5_max_over_readings"] = dict(
        per_reading=[float(v) for v in per],
        check_bound=float(got["bound"]),
        passed=bool(abs(got["bound"] - max(per)) < 1e-12
                    and got["bound"] >= max(per) - 1e-12))

    # T6: bound respects the trivial range [0, max c] for costs in [0,1]
    bad = 0
    for _ in range(100):
        P, mu0, pi, c = rand_mdp(rng)
        v = robust_eval(P, np.full(pi.shape, 0.3), pi, c, mu0)
        if v < -1e-12 or v > c.max() + 1e-9:
            bad += 1
    res["T6_range"] = dict(violations=bad, passed=bool(bad == 0))

    # T7: FULL-SIMPLEX rows must stay finite and bounded by max c. This is the
    # case the b=2 unsupported rows produce; the unclipped closed form diverges.
    bad, vals = 0, []
    for _ in range(100):
        P, mu0, pi, c = rand_mdp(rng)
        v = robust_eval(P, np.full(pi.shape, 2.0), pi, c, mu0)
        vals.append(v)
        if not np.isfinite(v) or v < -1e-9 or v > c.max() + 1e-7:
            bad += 1
    res["T7_full_simplex_finite"] = dict(
        violations=bad, max_bound=float(np.max(vals)),
        all_finite=bool(np.all(np.isfinite(vals))), passed=bool(bad == 0))

    # T8: soundness under FULL simplex on unsupported rows -- the true model is
    # trivially inside a b=2 ball, so the bound must dominate the true cost.
    dom_fail = 0
    for _ in range(200):
        P, mu0, pi, c = rand_mdp(rng)
        nS, nA = pi.shape
        N = rng.integers(0, 30, (nS, nA))          # many rows below M_SUPP
        Phat = np.zeros_like(P)
        for s in range(nS):
            for a in range(nA):
                if N[s, a] > 0:
                    Phat[s, a] = rng.multinomial(N[s, a], P[s, a]) / N[s, a]
                else:
                    Phat[s, a, s] = 1.0
        b = l1_radius(N, nS, 0.05)
        if not np.all(np.abs(Phat - P).sum(axis=2) <= b + 1e-12):
            continue
        true_c = float((exact_occupancy(P, mu0, pi) * c).sum())
        if robust_eval(Phat, b, pi, c, mu0) < true_c - 1e-9:
            dom_fail += 1
    res["T8_domination_with_unsupported"] = dict(
        domination_failures=dom_fail, passed=bool(dom_fail == 0))

    res["ALL_PASSED"] = all(v["passed"] for k, v in res.items()
                            if isinstance(v, dict) and "passed" in v)
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "evaluator_tests.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
