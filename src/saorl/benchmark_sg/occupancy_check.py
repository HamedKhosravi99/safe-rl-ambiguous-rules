"""Occupancy-based (density-ratio) Offline-CHECK. ADDITIVE; touches nothing else.

The model-based route failed because (s,a)-rectangular L1 robustness charges
pairs with zero true occupancy. This module asks whether the discounted
occupancy ratio w = d^pi / d^mu can be estimated from HELD-OUT DATA ALONE and
turned into a valid one-sided upper bound.

WHAT THE LOG GIVES US (see report section A).  `sample_log` restarts from mu0
with probability 1-gamma at every step, so the marginal law of the recorded
(s_i,a_i) is EXACTLY the normalized discounted occupancy d^mu of the behaviour
policy.  d^mu is therefore estimable by counting, with no model.

TABULAR DICE == PLUG-IN FLOW (proved numerically in `check_equivalence`).
With indicator test functions the empirical DICE moment condition

    (1/n) sum_i w(s_i,a_i) [ 1{(s_i,a_i)=(s,a)}
                             - gamma sum_{a'} pi(a'|s'_i) 1{(s'_i,a')=(s,a)} ]
        = (1-gamma) mu0(s) pi(a|s)

is algebraically the discounted-flow equation of the EMPIRICAL kernel. So in the
tabular case the "model-free" DICE point estimate and the plug-in model-based
occupancy are the same number. The occupancy route is therefore NOT a different
point estimate; it differs only in how uncertainty is quantified. This is
reported rather than glossed over.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from .control_mdp import GAMMA


# ---------------------------------------------------------------- logging
def sample_log_traj(m: dict, pi_b: np.ndarray, n: int, rng) -> dict:
    """Identical generative process to `sample_log`, additionally RECORDING the
    episode id and the within-episode time index.

    This instruments the logger; it uses no information `sample_log` did not
    already have. Episode boundaries are only PARTIALLY recoverable from the
    un-instrumented log (a restart landing on s'_i is indistinguishable from a
    continuation), so we record them rather than guess. Reported as an
    assumption in section A.
    """
    P, mu0, C = m["P"], m["mu0"], m["C"]
    nS, nA, K = m["nS"], m["nA"], C.shape[0]
    S = np.empty(n, np.int64); A = np.empty(n, np.int64)
    SP = np.empty(n, np.int64); EP = np.empty(n, np.int64); TT = np.empty(n, np.int64)
    s = rng.choice(nS, p=mu0); ep = 0; t = 0
    for i in range(n):
        a = rng.choice(nA, p=pi_b[s])
        sp = rng.choice(nS, p=P[s, a])
        S[i], A[i], SP[i], EP[i], TT[i] = s, a, sp, ep, t
        if rng.random() < GAMMA:
            s, t = sp, t + 1
        else:
            s, ep, t = rng.choice(nS, p=mu0), ep + 1, 0
    return dict(S=S, A=A, SP=SP, EP=EP, T=TT, nS=nS, nA=nA, K=K, mu0=mu0)


# ------------------------------------------------------- occupancy helpers
def true_occupancy(P, mu0, pi, gamma=GAMMA):
    """Oracle d^pi. DIAGNOSTIC ONLY -- never enters the certificate."""
    nS, nA = pi.shape
    Ppi = np.einsum("sa,san->sn", pi, P)
    x_s = (1.0 - gamma) * np.linalg.solve(np.eye(nS) - gamma * Ppi.T, mu0)
    return x_s[:, None] * pi


def dice_ratio(lg: dict, pi: np.ndarray, gamma=GAMMA, ridge=1e-9):
    """Tabular DICE: solve the empirical moment condition for d^pi, return
    w_hat = d^pi_hat / d^mu_hat on the observed support.

    Uses ONLY: held-out samples (s,a,s'), the frozen target policy pi, and mu0.
    No transition model is formed explicitly and the true kernel is never read.
    """
    S, A, SP, mu0 = lg["S"], lg["A"], lg["SP"], lg["mu0"]
    nS, nA, n = lg["nS"], lg["nA"], len(S)
    d = nS * nA
    idx = S * nA + A

    # empirical d^mu by counting (exact: the log's marginal IS d^mu)
    Nsa = np.bincount(idx, minlength=d).astype(float)
    dmu = Nsa / n

    # M[j, k] = (1/n) sum_i [ 1{(s_i,a_i)=k} - gamma pi(a_k|s'_i) 1{s'_i=s_k} ]
    # solved for the vector dpi over all (s,a): dpi = (1-g) mu0 x pi + g * (flow)
    M = np.zeros((d, d))
    np.add.at(M, (idx, idx), 1.0)
    for a2 in range(nA):
        np.add.at(M, (idx, SP * nA + a2), -gamma * pi[SP, a2])
    M /= n
    b = (1.0 - gamma) * (mu0[:, None] * pi).reshape(-1)
    # dpi solves M^T dpi_over_dmu ... work directly in the occupancy variable:
    # sum_i w(s_i,a_i) [phi(s_i,a_i) - g E_pi phi(s'_i,.)] / n = (1-g) E_mu0,pi phi
    w = np.linalg.solve(M.T + ridge * np.eye(d), b)
    dpi_hat = w * dmu
    with np.errstate(divide="ignore", invalid="ignore"):
        w_hat = np.where(dmu > 0, w, 0.0)
    return dict(w=w_hat.reshape(nS, nA), dpi=dpi_hat.reshape(nS, nA),
                dmu=dmu.reshape(nS, nA), N=Nsa.reshape(nS, nA))


def plugin_flow_ratio(lg: dict, pi: np.ndarray, gamma=GAMMA):
    """Plug-in: build Phat, solve the exact flow equation, divide by dmu_hat.
    Used ONLY to demonstrate the tabular DICE == plug-in equivalence."""
    S, A, SP, mu0 = lg["S"], lg["A"], lg["SP"], lg["mu0"]
    nS, nA, n = lg["nS"], lg["nA"], len(S)
    Phat = np.zeros((nS, nA, nS))
    np.add.at(Phat, (S, A, SP), 1.0)
    tot = Phat.sum(axis=2, keepdims=True)
    Phat = np.where(tot > 0, Phat / np.maximum(tot, 1), 0.0)
    for s in range(nS):
        for a in range(nA):
            if tot[s, a, 0] == 0:
                Phat[s, a, s] = 1.0
    dpi = true_occupancy(Phat, mu0, pi, gamma)
    Nsa = np.bincount(S * nA + A, minlength=nS * nA).reshape(nS, nA).astype(float)
    dmu = Nsa / n
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(dmu > 0, dpi / np.maximum(dmu, 1e-300), 0.0)
    return dict(w=w, dpi=dpi, dmu=dmu, N=Nsa)
