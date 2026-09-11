#!/usr/bin/env python3
"""Finite-data epsilon-face certificate for discounted CMDPs in the paper's normalized occupancy convention.

Model: P[s,a,s'], mu0[s], reward r[s,a] in [0, r_max], costs c_k[s,a] in [0, c_max], discount gamma.
Normalized occupancy x(s,a) >= 0 with sum_a x(s',a) - gamma sum_{s,a} P(s'|s,a) x(s,a) = (1-gamma) mu0(s'),
so sum x = 1 and J_f(pi) = f . x in [0, f_max].  Simulation lemma in these units:
|J_f^P(pi) - J_f^{Phat}(pi)| <= f_max * (gamma/(1-gamma)) * max_{s,a} ||P(.|s,a) - Phat(.|s,a)||_1.
"""
import math
import numpy as np
from scipy.optimize import linprog

class DiscountedCMDP:
    def __init__(self, P, mu0, r, costs, gamma):
        self.P, self.mu0, self.r, self.costs, self.gamma = np.asarray(P, float), np.asarray(mu0, float), np.asarray(r, float), [np.asarray(c, float) for c in costs], float(gamma)
        self.S, self.A = self.r.shape
        self._eq_true = None
    def occ(self, P=None):
        if P is None and self._eq_true is not None: return self._eq_true
        Pm = self.P if P is None else P
        S, A, g = self.S, self.A, self.gamma
        Aeq = np.zeros((S, S * A))
        for s2 in range(S):
            for s in range(S):
                for a in range(A):
                    Aeq[s2, s * A + a] = (1.0 if s == s2 else 0.0) - g * Pm[s, a, s2]
        beq = (1 - g) * self.mu0
        if P is None: self._eq_true = (Aeq, beq)
        return Aeq, beq
    def lp(self, obj, ineqs=(), P=None, maximize=True, support=None):
        Aeq, beq = self.occ(P)
        A_ub = np.array([v.reshape(-1) for v, _ in ineqs]) if len(ineqs) else None
        b_ub = np.array([b for _, b in ineqs]) if len(ineqs) else None
        bounds = [(0, None)] * (self.S * self.A)
        if support is not None:                      # restrict to policies supported on covered pairs
            for s in range(self.S):
                for a in range(self.A):
                    if not support[s, a]: bounds[s * self.A + a] = (0, 0)
        res = linprog(-obj.reshape(-1) if maximize else obj.reshape(-1), A_ub=A_ub, b_ub=b_ub, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
        if res.status != 0: return None
        return -res.fun if maximize else res.fun

def V_single(m, k, d, P=None, support=None):
    return m.lp(m.r, [(m.costs[k], d)], P, support=support)
def V_set(m, d, P=None, support=None):
    return m.lp(m.r, [(c, d) for c in m.costs], P, support=support)
def dominant(m, k):
    return all(np.all(m.costs[k] >= m.costs[j] - 1e-12) for j in range(len(m.costs)) if j != k)
def W_eps(m, k, d, eps, P=None, support=None):
    """Worst competing cost over the epsilon-face of reading k. None if reading k infeasible."""
    Vk = V_single(m, k, d, P, support)
    if Vk is None: return None
    best = -np.inf
    for j in range(len(m.costs)):
        if j == k: continue
        w = m.lp(m.costs[j], [(m.costs[k], d), (-m.r, -(Vk - eps))], P, support=support)
        if w is not None: best = max(best, w)
    return best if best > -np.inf else 0.0
def exact_verdict(m, k, d, eps=0.0):
    w = W_eps(m, k, d, eps)
    return None if w is None else (w <= d + 1e-9, w - d)

# ---------- concentration for the per-pair L1 error ----------
def l1_weissman(n, S, delta_pair):
    return math.sqrt(2.0 * (S * math.log(2.0) + math.log(1.0 / delta_pair)) / n) if n > 0 else 2.0
def l1_bernstein(phat_row, n, delta_pair):
    """Sum over outcomes of empirical-Bernstein deviations (Maurer-Pontil), union over the S outcomes."""
    S = len(phat_row)
    if n < 2: return 2.0
    L = math.log(2.0 * S / delta_pair)
    dev = np.sqrt(2.0 * phat_row * (1.0 - phat_row) * L / (n - 1)) + 7.0 * L / (3.0 * (n - 1))
    return float(min(2.0, dev.sum()))
def sample_generative(P, n, rng):
    Phat = np.zeros_like(P); counts = np.full(P.shape[:2], n)
    for s in range(P.shape[0]):
        for a in range(P.shape[1]):
            Phat[s, a] = rng.multinomial(n, P[s, a]) / n
    return Phat, counts
def sample_trajectories(m, behavior, n_steps, rng, restart_prob=None):
    """Log n_steps transitions under a behavior policy behavior[s,a]; restarts from mu0 w.p. 1-gamma (discounted episodes)."""
    S, A = m.S, m.A
    counts = np.zeros((S, A), int); next_counts = np.zeros((S, A, S), int)
    rp = (1 - m.gamma) if restart_prob is None else restart_prob
    s = rng.choice(S, p=m.mu0)
    for _ in range(n_steps):
        a = rng.choice(A, p=behavior[s]); s2 = rng.choice(S, p=m.P[s, a])
        counts[s, a] += 1; next_counts[s, a, s2] += 1
        s = rng.choice(S, p=m.mu0) if rng.uniform() < rp else s2
    Phat = np.where(counts[..., None] > 0, next_counts / np.maximum(counts[..., None], 1), 0.0)
    return Phat, counts

def certificate(m, Phat, counts, k, d, eps, delta, bound="weissman", n_min=1, Nmax=None):
    """High-confidence Decide for reading k. Returns dict(pass, What, beta, covered_frac)."""
    S, A, g = m.S, m.A, m.gamma
    support = counts >= max(1, n_min)
    if not support.any(): return {"pass": False, "What": None, "beta": None, "covered": 0.0}
    npairs = S * A
    delta_pair = delta / npairs if Nmax is None else delta / (npairs * Nmax)   # union over pairs (and visit counts for trajectory data)
    errs = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            if support[s, a]:
                errs[s, a] = l1_weissman(int(counts[s, a]), S, delta_pair) if bound == "weissman" else l1_bernstein(Phat[s, a], int(counts[s, a]), delta_pair)
    Lsim = g / (2.0 * (1.0 - g))          # centered simulation lemma: |(p-phat).V| <= ||p-phat||_1 * span(V)/2, span(V) <= f_max/(1-gamma)
    beta = Lsim * float(errs[support].max())
    r_max = float(m.r.max()); c_max = float(max(c.max() for c in m.costs))
    beta_r, beta_c = r_max * beta, c_max * beta
    Vm = V_single(m, k, d - beta_c, Phat, support)
    if Vm is None: return {"pass": False, "What": None, "beta": beta, "covered": float(support.mean())}
    Vm -= beta_r
    best = -np.inf
    for j in range(len(m.costs)):
        if j == k: continue
        w = m.lp(m.costs[j], [(m.costs[k], d + beta_c), (-m.r, -(Vm - beta_r - eps))], Phat, support=support)
        if w is not None: best = max(best, w)
    What = (best if best > -np.inf else 0.0) + beta_c
    return {"pass": bool(What <= d + 1e-9), "What": float(What), "beta": beta, "covered": float(support.mean())}

if __name__ == "__main__":
    # self-test on a random sparse discounted instance
    rng = np.random.default_rng(1)
    S, A, g = 12, 2, 0.9
    P = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            nxt = rng.choice(S, size=3, replace=False); P[s, a, nxt] = rng.dirichlet(np.ones(3))
    r = rng.uniform(0, 1, (S, A)); c0 = (rng.uniform(size=(S, A)) < 0.3).astype(float); c1 = (rng.uniform(size=(S, A)) < 0.3).astype(float)
    m = DiscountedCMDP(P, rng.dirichlet(np.ones(S)), r, [c0, c1], g)
    d, eps = 0.15, 0.02
    for k in range(2):
        ex = exact_verdict(m, k, d, eps)
        c0chk = certificate(m, m.P.copy(), np.full((S, A), 10**9), k, d, eps, 0.05)   # beta ~ 0 with huge counts
        print(f"reading {k}: exact eps-verdict {ex}, certificate at true model with huge counts: pass={c0chk['pass']} What-d={c0chk['What']-d:+.4f} beta={c0chk['beta']:.2e}")
    Phat, counts = sample_generative(m.P, 200000, rng)
    for b in ("weissman", "bernstein"):
        print(b, certificate(m, Phat, counts, 0, d, eps, 0.05, bound=b))
    Phat, counts = sample_trajectories(m, np.full((S, A), 0.5), 400000, rng)
    print("trajectory data, min count", counts.min(), certificate(m, Phat, counts, 0, d, eps, 0.05, bound="bernstein", n_min=100, Nmax=400000))
