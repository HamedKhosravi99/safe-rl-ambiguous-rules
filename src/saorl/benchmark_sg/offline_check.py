"""Held-out Offline-CHECK: certify a frozen policy from held-out logged data.

ADDITIVE. Nothing here modifies the existing CHECK (`check_ship` in
safe_face_offline.py), which samples fresh episodes from the true model.

Setting.  The compiled instances have KNOWN per-(s,a) costs c_phi and an UNKNOWN
transition kernel P, so all evaluation uncertainty is in the occupancy that P
induces.  Offline-CHECK therefore builds an (s,a)-rectangular L1 confidence set
around the held-out empirical kernel and computes, for the frozen policy, a
robust (worst-case) policy evaluation over that set.

Confidence set.  Identical construction to the offline DECIDE certificate
(saorl/certified_at_scale.py, the T0.1-synchronized Weissman form):

    b(s,a) = min{2, sqrt( 2 (S ln 2 + ln(S A N / delta)) / n(s,a) )}

with a union bound over the S*A pairs, so P* lies in the set with probability at
least 1 - delta simultaneously.  Pairs with n(s,a) < M_SUPP get b = 2, i.e. the
full simplex: no imputation, no optimism, no silent extrapolation.

Robust evaluation.  For a FIXED policy pi and (s,a)-rectangular uncertainty the
worst case is a gamma-contraction, so robust value iteration converges:

    V(s) = sum_a pi(a|s) [ (1-gamma) c(s,a) + gamma * max_{p in P(s,a)} p . V ]
    max_{p in P(s,a)} p . V = phat(s,a) . V + (b(s,a)/2) (max V - min V)

which is the exact support function of an L1 ball intersected with the simplex,
so the bound is tight for the set and never underestimates cost.  The reported
bound is B_phi(pi) = sum_s mu0(s) V(s), matching the normalization of
`occupancy()` (sum of occupancy = 1).
"""
from __future__ import annotations

import numpy as np

from .control_mdp import GAMMA

M_SUPP = 5          # held-out support threshold; below this the set is the simplex
VI_TOL = 1e-12
VI_MAX = 20000


def l1_radius(N: np.ndarray, nS: int, delta: float) -> np.ndarray:
    """Weissman L1 radius per (s,a), union-bounded over S*A pairs."""
    nSA = N.size
    Ntot = max(int(N.sum()), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.sqrt(2.0 * (nS * np.log(2.0) + np.log(nSA * Ntot / delta))
                    / np.maximum(N, 1))
    b = np.minimum(2.0, b)
    b[N < M_SUPP] = 2.0                      # unsupported: whole simplex
    return b


def robust_eval(Phat: np.ndarray, b: np.ndarray, pi: np.ndarray,
                c: np.ndarray, mu0: np.ndarray, gamma: float = GAMMA) -> float:
    """sup over the L1 set of J_c(pi), by robust value iteration (fixed pi)."""
    nS, nA = pi.shape
    V = np.zeros(nS)
    imm = (1.0 - gamma) * (pi * c).sum(axis=1)          # (1-g) sum_a pi c
    resid = 0.0
    for _ in range(VI_MAX):
        vmax, vmin = V.max(), V.min()
        spread = 0.5 * (vmax - vmin)
        # Worst-case next-state value for each (s,a). Two valid upper bounds on
        # the support function sup{p.V : ||p-phat||_1 <= b, p in simplex}:
        #   (i) phat.V + (b/2)(max V - min V)   [Holder / total-variation]
        #  (ii) max V                           [p is a distribution]
        # Taking the min keeps the bound sound and, crucially, keeps the
        # operator a gamma-contraction: without (ii) the b=2 rows used for
        # unsupported (s,a) make the iteration diverge.
        nxt = np.minimum(Phat @ V + b * spread, vmax)   # (nS,nA)
        Vn = imm + gamma * (pi * nxt).sum(axis=1)
        resid = float(np.max(np.abs(Vn - V)))
        V = Vn
        if resid < VI_TOL:
            break
    # Costs are nonnegative and V_0 = 0, so the iterates increase to the fixed
    # point FROM BELOW: an early stop would underestimate. Add the standard
    # contraction correction so the returned number is a certified upper bound
    # on the fixed point, not merely a converged approximation of it.
    return float(mu0 @ V) + gamma * resid / (1.0 - gamma)


def plugin_eval(Phat: np.ndarray, pi: np.ndarray, c: np.ndarray,
                mu0: np.ndarray, gamma: float = GAMMA) -> float:
    """UNSAFE control: plug-in evaluation with no uncertainty correction."""
    return robust_eval(Phat, np.zeros_like(pi), pi, c, mu0, gamma)


class HeldOut:
    """Sufficient statistics of the held-out split only."""

    def __init__(self, lg: dict):
        nS, nA = lg["nS"], lg["nA"]
        self.nS, self.nA, self.mu0 = nS, nA, lg["mu0"]
        idx = lg["S"] * nA + lg["A"]
        self.N = np.bincount(idx, minlength=nS * nA).reshape(nS, nA)
        self.Phat = np.zeros((nS, nA, nS))
        np.add.at(self.Phat, (lg["S"], lg["A"], lg["SP"]), 1.0)
        tot = self.Phat.sum(axis=2, keepdims=True)
        self.Phat = np.where(tot > 0, self.Phat / np.maximum(tot, 1), 0.0)
        for s in range(nS):
            for a in range(nA):
                if self.N[s, a] == 0:
                    self.Phat[s, a, s] = 1.0      # placeholder; b=2 there anyway

    def check(self, pi: np.ndarray, C: np.ndarray, d: float,
              delta_ev: float = 0.05) -> dict:
        """Offline-CHECK: robust bound over every reading, ship iff <= d."""
        b = l1_radius(self.N, self.nS, delta_ev)
        bounds = [robust_eval(self.Phat, b, pi, C[k], self.mu0)
                  for k in range(C.shape[0])]
        B = float(max(bounds))
        # coverage diagnostics for the abstention taxonomy
        used = pi > 1e-12
        unsup = bool((used & (self.N < M_SUPP)).any())
        frac_unsup = float((used & (self.N < M_SUPP)).sum() / max(used.sum(), 1))
        plug = float(max(plugin_eval(self.Phat, pi, C[k], self.mu0)
                         for k in range(C.shape[0])))
        return dict(bound=B, ship=bool(B <= d), plugin=plug,
                    plugin_ship=bool(plug <= d),
                    uses_unsupported=unsup, frac_unsupported=frac_unsup,
                    median_radius=float(np.median(b[used])) if used.any() else None,
                    n_check=int(self.N.sum()))
