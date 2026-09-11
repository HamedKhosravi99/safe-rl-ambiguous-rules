"""Certified-PROTECT: DECIDE + certified planner, with NO post-hoc CHECK.

ADDITIVE. Does not replace PROTECT and does not touch any archived result.

COMPOSITION UNDER TEST
  (1) offline DECIDE certifies   F_psi^{eps'}(d) subset Pi_{U_T}(d)
  (2) Certified-PROTECT returns  pi_hat in F_psi^{eps'}(d)
  => pi_hat in Pi_{U_T}(d), so psi^dagger in U_T gives J_{c_psidagger}(pi_hat) <= d.

WHY THE REWARD HALF OF THE CONTRACT IS FREE HERE.  On the `irrelevant`
instances the exact sufficiency ceiling eps*_max saturates at 1.0, and rewards
are normalised to [0,1], so the constraint J_r >= V_psi(d) - eps' is vacuous:
F_psi^{eps'}(d) = Pi_psi(d). Certified-PROTECT therefore only has to certify
ANCHOR FEASIBILITY -- no reward lower bound, no upper bound on V_psi(d), and no
beta_r term. (On instances where eps*_max < 1 the reward half would bind; that
case is reported separately, not assumed away.)

CERTIFIED FEASIBILITY BY CONSTRUCTION.  The planner solves the occupancy LP in
the estimated model with the simulation-lemma penalty
    c_pen(s,a) = c_psi(s,a) + gamma * (1/(1-gamma)) * b(s,a),
b the Weissman L1 radius. On the confidence event {||Phat-P||_1 <= b}, any LP
solution with penalised cost <= d has TRUE anchor cost <= d. Nothing is
evaluated post hoc; feasibility is a property of the program that produced the
policy. The always-intervene column is exempt (b=0) because its contract is
mechanical, so a pathwise-safe fallback is always representable.

TASK 5 COMPLIANCE. The planner constrains ONLY the anchor cost c_psi. Costs of
the other surviving readings are never evaluated to authorise deployment; they
are computed with the true model AFTERWARD purely to score the experiment.

EXACT SAMPLING SHORTCUT. To reach N up to 1e10 we do not step a simulator. The
per-pair counts are drawn multinomially from the true behaviour occupancy and
each Phat(.|s,a) row is drawn as Multinomial(n(s,a), P(.|s,a))/n(s,a). That is
the exact law of the sufficient statistics the planner consumes.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.optimize import linprog

from .control_mdp import GAMMA

DELTA_TR = 0.10       # confidence level of the transition event
M_SUPP = 5            # support threshold
SAFE_A = 1            # the "intervene" action (mechanically certified, cost 0)


def flow_matrices(P, mu0):
    nS, nA, _ = P.shape
    A_eq = np.zeros((nS, nS * nA))
    for s in range(nS):
        for a in range(nA):
            A_eq[s, s * nA + a] += 1.0
    A_eq -= GAMMA * np.transpose(P, (2, 0, 1)).reshape(nS, nS * nA)
    return A_eq, (1.0 - GAMMA) * mu0


def occupancy_of(P, mu0, pi):
    nS, nA, _ = P.shape
    Ppi = np.einsum("sap,sa->sp", P, pi)
    d_s = np.linalg.solve(np.eye(nS) - GAMMA * Ppi.T, (1 - GAMMA) * mu0)
    return d_s[:, None] * pi


def eval_policy(P, f, mu0, pi):
    nS, nA, _ = P.shape
    Ppi = np.einsum("sap,sa->sp", P, pi)
    fpi = np.einsum("sa,sa->s", f, pi)
    v = np.linalg.solve(np.eye(nS) - GAMMA * Ppi, fpi)
    return float((1.0 - GAMMA) * (mu0 @ v))


def draw_statistics(P, mu0, N, rng, pi_b=None):
    """Exact law of (counts, Phat) for a log of N transitions."""
    nS, nA, _ = P.shape
    if pi_b is None:
        pi_b = np.full((nS, nA), 1.0 / nA)          # generic uniform behaviour
    dmu = occupancy_of(P, mu0, pi_b).reshape(-1)
    dmu = np.maximum(dmu, 0); dmu /= dmu.sum()
    counts = rng.multinomial(int(min(N, 2**62)), dmu).reshape(nS, nA)
    Phat = np.zeros_like(P)
    for s in range(nS):
        for a in range(nA):
            n = counts[s, a]
            if n > 0:
                Phat[s, a] = rng.multinomial(int(n), P[s, a]) / n
            else:
                Phat[s, a, s] = 1.0
    return counts, Phat


def radii(counts, nS, nA, N):
    S_ln2 = nS * np.log(2.0)
    b = np.full((nS, nA), 2.0)
    nz = counts > 0
    b[nz] = np.minimum(2.0, np.sqrt(
        2 * (S_ln2 + np.log(nS * nA * max(N, 1) / DELTA_TR)) / counts[nz]))
    return b


def certified_protect(P_true, mu0, r, c_anchor, d, N, rng, pi_b=None):
    """Return (pi_hat, info). Certified anchor-feasible on the confidence event."""
    nS, nA, _ = P_true.shape
    counts, Phat = draw_statistics(P_true, mu0, N, rng, pi_b)
    b = radii(counts, nS, nA, N)

    # mechanically certified fallback column: intervene -> safe state, cost 0
    Phat = Phat.copy()
    Phat[:, SAFE_A] = 0.0
    Phat[:, SAFE_A, 0] = 1.0
    b[:, SAFE_A] = 0.0

    mask = (counts >= M_SUPP)
    mask[:, SAFE_A] = True

    Gam = 1.0 / (1.0 - GAMMA)
    Vr = float(r.max() - r.min()) * Gam
    r_pen = r - GAMMA * Vr * b                       # pessimistic reward
    c_pen = c_anchor + GAMMA * Gam * b               # optimistic (upper) cost

    A_eq, b_eq = flow_matrices(Phat, mu0)
    bounds = [(0, None) if m else (0, 0) for m in mask.reshape(-1)]
    res = linprog(-r_pen.reshape(-1), A_ub=np.array([c_pen.reshape(-1)]),
                  b_ub=np.array([d]), A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if res.status != 0:
        pi = np.zeros((nS, nA)); pi[:, SAFE_A] = 1.0
        return pi, dict(certified=True, used_fallback=True,
                        b_med=float(np.median(b[counts >= M_SUPP])) if mask.any() else 2.0,
                        supp_frac=float(mask.mean()))
    x = res.x.reshape(nS, nA)
    pi = np.zeros((nS, nA)); tot = x.sum(1)
    for s in range(nS):
        pi[s] = x[s] / tot[s] if tot[s] > 1e-12 else np.eye(nA)[SAFE_A]
    nontrivial = bool((pi[:, SAFE_A] < 0.999).any())
    return pi, dict(certified=True, used_fallback=not nontrivial,
                    b_med=float(np.median(b[counts >= M_SUPP])) if (counts >= M_SUPP).any() else 2.0,
                    supp_frac=float(mask.mean()))
