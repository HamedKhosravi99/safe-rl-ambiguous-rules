#!/usr/bin/env python3
"""Numerical checks for the ARROW theory-extension note (finite-horizon tabular CMDPs, occupancy LPs).

Checks: (A) Clopper-Pearson bound for delta_gen from the archived E3 counts;
(B) exact-face knife edge vs epsilon-face stability; (C) soundness/completeness of the
model-based high-confidence Decide certificate on random CMDPs; (D) 1/gamma^2 scaling of the
sample size in the two-state instance; (E) price-of-ambiguity bounds; (F) budget structure of W(d).
"""
import json, math, time
import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta as beta_dist

rng = np.random.default_rng(20260903)
T0 = time.time()
OUT = {}

class CMDP:
    """Finite-horizon tabular CMDP with time-homogeneous P[s,a,s'], reward r[s,a], costs[k][s,a]."""
    def __init__(self, P, mu0, r, costs, H):
        self.P, self.mu0, self.r, self.costs, self.H = P, mu0, r, costs, H
        self.S, self.A = r.shape
        self._eq = {}
    def occ(self, P=None):
        key = "true" if P is None else None          # cache only the true model (sampled models are rebuilt)
        if key is not None and key in self._eq: return self._eq[key]
        P = self.P if P is None else P
        S, A, H = self.S, self.A, self.H
        n = H * S * A
        Aeq = np.zeros((H * S, n)); beq = np.zeros(H * S)
        idx = lambda h, s, a: h * S * A + s * A + a
        for s in range(S):
            for a in range(A): Aeq[s, idx(0, s, a)] = 1.0
            beq[s] = self.mu0[s]
        for h in range(1, H):
            for s2 in range(S):
                row = h * S + s2
                for a in range(A): Aeq[row, idx(h, s2, a)] = 1.0
                for s in range(S):
                    for a in range(A): Aeq[row, idx(h - 1, s, a)] -= P[s, a, s2]
        if key is not None: self._eq[key] = (Aeq, beq)
        return Aeq, beq
    def vec(self, f): return np.tile(np.asarray(f).reshape(-1), self.H)
    def lp(self, obj, ineqs=(), P=None, maximize=True):
        Aeq, beq = self.occ(P)
        A_ub = np.array([v for v, _ in ineqs]) if len(ineqs) else None
        b_ub = np.array([b for _, b in ineqs]) if len(ineqs) else None
        res = linprog(-obj if maximize else obj, A_ub=A_ub, b_ub=b_ub, A_eq=Aeq, b_eq=beq,
                      bounds=(0, None), method="highs")
        if res.status != 0: return None
        return -res.fun if maximize else res.fun

def V(m, psi, d, P=None):
    return m.lp(m.vec(m.r), [(m.vec(m.costs[psi]), d)], P)
def V_set(m, d, P=None):
    return m.lp(m.vec(m.r), [(m.vec(c), d) for c in m.costs], P)
def W_eps(m, psi, d, eps, P=None):
    """max_phi sup{J_phi : J_psi<=d, J_r>=V_psi(d)-eps}; None if psi infeasible."""
    Vp = V(m, psi, d, P)
    if Vp is None: return None
    best = -np.inf
    for phi in range(len(m.costs)):
        if phi == psi: continue
        w = m.lp(m.vec(m.costs[phi]), [(m.vec(m.costs[psi]), d), (-m.vec(m.r), -(Vp - eps))], P)
        if w is not None: best = max(best, w)
    return best
def certificate(m, Phat, psi, d, eps, beta_c, beta_r):
    """High-confidence Decide: returns (pass, What). Conservative empirical epsilon-face."""
    Vm = V(m, psi, d - beta_c, Phat)
    if Vm is None: return False, None
    Vm -= beta_r                       # lower confidence bound on V_psi(d)
    best = -np.inf
    for phi in range(len(m.costs)):
        if phi == psi: continue
        w = m.lp(m.vec(m.costs[phi]), [(m.vec(m.costs[psi]), d + beta_c), (-m.vec(m.r), -(Vm - beta_r - eps))], Phat)
        if w is not None: best = max(best, w)
    What = best + beta_c
    return bool(What <= d + 1e-9), What
def beta_n(S, npairs, n, H, delta):
    eps_n = math.sqrt(2.0 * (S * math.log(2.0) + math.log(npairs / delta)) / n)   # Weissman et al. L1 bound, union over pairs
    return H * (H - 1) / 2.0 * eps_n                                              # finite-horizon simulation lemma
def sample_Phat(P, n, rng):
    Phat = np.zeros_like(P)
    for s in range(P.shape[0]):
        for a in range(P.shape[1]):
            Phat[s, a] = rng.multinomial(n, P[s, a]) / n
    return Phat

# ---------------- (A) delta_gen upper confidence bound (one-sided Clopper-Pearson) ----------------
def cp_upper(x, m, b): return float(beta_dist.ppf(1 - b, x + 1, m - x)) if x < m else 1.0
A = {}
for name, m, gen in [("faithful", 173, 111), ("expressible", 315, 111)]:
    x = m - gen
    A[name] = {"m": m, "misses": x, "point": x / m, "ucb_95": cp_upper(x, m, 0.05), "ucb_90": cp_upper(x, m, 0.10)}
OUT["A_delta_gen"] = A
print("[A] delta_gen:", json.dumps(A, indent=1))

# ---------------- (B) exact-face knife edge vs epsilon-face ----------------
# H=2; s0 -> a1: reward state w.p. q1, a2: w.p. q2. c_phi(s0,a1)=0.8 known; c_psi=0. d=0.3.
def knife(q1, q2, eps):
    S, A_, H = 3, 2, 2                  # s0, sR (reward 1), s0' (nothing); absorbing
    P = np.zeros((S, A_, S)); P[0, 0] = [0, q1, 1 - q1]; P[0, 1] = [0, q2, 1 - q2]
    P[1, :, 1] = 1; P[2, :, 2] = 1
    r = np.zeros((S, A_)); r[1, :] = 1.0
    cpsi = np.zeros((S, A_)); cphi = np.zeros((S, A_)); cphi[0, 0] = 0.8
    m = CMDP(P, np.array([1.0, 0, 0]), r, [cpsi, cphi], H)
    d = 0.3
    return {"Gamma_exact": W_eps(m, 0, d, 0.0) - d, "Gamma_eps": W_eps(m, 0, d, eps) - d}
def knife2(eta, eps, c0=0.5, d=0.3, q=0.5):
    S, A_, H = 3, 2, 2
    P = np.zeros((S, A_, S)); P[0, 0] = [0, q, 1 - q]; P[0, 1] = [0, q + eta, 1 - q - eta]; P[1, :, 1] = 1; P[2, :, 2] = 1
    r = np.zeros((S, A_)); r[1, :] = 1.0
    cpsi = np.zeros((S, A_)); cpsi[0, 1] = c0          # psi charges a2
    cphi = np.zeros((S, A_)); cphi[0, 0] = c0          # phi charges a1
    m = CMDP(P, np.array([1.0, 0, 0]), r, [cpsi, cphi], H)
    return {"Gamma_star_exact": min(W_eps(m, k, d, 0.0) for k in range(2)) - d,
            "Gamma_star_eps": min(W_eps(m, k, d, eps) for k in range(2)) - d}
B = {"one_reading": {"tie": knife(0.5, 0.5, 0.05), "perturbed": knife(0.5, 0.51, 0.05)},
     "two_readings_Gamma_star": {"world_A_eta=0": knife2(0.0, 0.05), "world_B_eta=0.01": knife2(0.01, 0.05)}, "eps": 0.05, "eta": 0.01}
OUT["B_knife_edge"] = B
print("[B] knife edge:", json.dumps(B, indent=1))

# ---------------- (C) soundness / completeness on random CMDPs ----------------
def random_instance(rng, S=5, A_=3, H=4, K=2):
    P = rng.dirichlet(np.full(S, 0.5), size=(S, A_))
    r = rng.uniform(0, 1, size=(S, A_))
    costs = []
    for k in range(K):
        mask = rng.uniform(size=(S, A_)) < 0.35
        costs.append(np.where(mask, rng.uniform(0.5, 1.0, size=(S, A_)), 0.0))
    mu0 = rng.dirichlet(np.ones(S))
    return CMDP(P, mu0, r, costs, H)
def dominated(m, psi):
    return any(np.all(m.costs[psi] >= m.costs[phi]) for phi in range(len(m.costs)) if phi != psi)
eps = 0.1; delta = 0.05
suff, prot = [], []
tries = 0
while (len(suff) < 3 or len(prot) < 3) and tries < 400:
    tries += 1
    m = random_instance(rng)
    for psi in range(len(m.costs)):
        if dominated(m, psi): continue
        for d in np.linspace(0.05, 1.5, 30):
            w = W_eps(m, psi, d, eps)
            if w is None: continue
            g = w - d
            if -0.15 <= g <= -0.02 and len(suff) < 3: suff.append((m, psi, float(d), float(g))); break
            if 0.05 <= g <= 0.5 and len(prot) < 3: prot.append((m, psi, float(d), float(g))); break
print(f"[C] found {len(suff)} sufficient and {len(prot)} protect instances in {tries} tries")
selftest = all(certificate(m, m.P.copy(), psi, d, eps, 0.0, 0.0)[0] == (g <= 0) for (m, psi, d, g) in suff + prot)
print("[C] self-test (beta=0, true model reproduces exact verdict):", selftest); assert selftest
npairs = 5 * 3
C = {"eps": eps, "delta": delta, "n_grid": [10**3, 10**4, 10**5, 10**6, 10**7], "sufficient": [], "protect": []}
# sensitivity diagnostic: relaxed-exact certificate at the TRUE model, W_hat(beta) as a function of the slack beta
def relaxed_exact(m, psi, d, eps, b):
    return certificate(m, m.P.copy(), psi, d, eps, b, b)[1]
for label, lst in [("sufficient", suff), ("protect", prot)]:
    for (m, psi, d, g) in lst:
        rec = {"margin_W_minus_d": g, "d": d, "pass_rate": {}}
        if label == "sufficient":
            curve = {b: relaxed_exact(m, psi, d, eps, b) - d for b in [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]}
            rec["relaxed_exact_margin_vs_beta"] = curve
            # largest slack still passing at the true model, and the sample size at which beta_n reaches half of it
            bstar = max([b for b, v in curve.items() if v <= 0], default=0.0)
            rec["beta_star"] = bstar
            if bstar > 0:
                n_need = int((m.H * (m.H - 1) / 2) ** 2 * 2 * (m.S * math.log(2) + math.log(npairs / delta)) / (bstar / 2) ** 2)
                rec["n_needed_for_beta_half_bstar"] = n_need
                for n in [n_need, 4 * n_need]:
                    if n > 2 * 10**9: continue
                    b = beta_n(m.S, npairs, n, m.H, delta)
                    passes = sum(certificate(m, sample_Phat(m.P, n, rng), psi, d, eps, b, b)[0] for _ in range(20))
                    rec["pass_rate"][str(n)] = passes / 20
            print(f"[C] sensitivity: margin={g:+.3f} curve={ {k: round(v,4) for k,v in curve.items()} } beta*={bstar} n_needed={rec.get('n_needed_for_beta_half_bstar')}")
        for n in C["n_grid"]:
            b = beta_n(m.S, npairs, n, m.H, delta)
            passes = 0; reps = 40
            for _ in range(reps):
                Phat = sample_Phat(m.P, n, rng)
                ok, _ = certificate(m, Phat, psi, d, eps, b, b)
                passes += ok
            rec["pass_rate"][str(n)] = passes / reps
            rec.setdefault("beta_n", {})[str(n)] = b
        C[label].append(rec)
        print(f"[C] {label}: margin={g:+.3f} d={d:.3f} pass rates {rec['pass_rate']}")
OUT["C_certificate"] = C

# ---------------- (D) 1/gamma^2 scaling in the two-state instance ----------------
def two_state(p):
    S, A_, H = 3, 2, 2                  # s0 -> a1: bad w.p. p (reward 1 either way), a2: good, reward 0.5
    P = np.zeros((S, A_, S)); P[0, 0] = [0, 1 - p, p]; P[0, 1] = [0, 1, 0]; P[1, :, 1] = 1; P[2, :, 2] = 1
    r = np.zeros((S, A_)); r[0, 0] = 1.0; r[0, 1] = 0.5
    cpsi = np.zeros((S, A_)); cphi = np.zeros((S, A_)); cphi[2, :] = 1.0   # phi charges the bad state
    return CMDP(P, np.array([1.0, 0, 0]), r, [cpsi, cphi], H)
d = 0.3; D = {"d": d, "eps": 0.05, "delta": 0.05, "rows": []}
for gamma in [0.08, 0.04, 0.02, 0.01]:
    m = two_state(d - gamma)
    w = W_eps(m, 0, d, 0.05); assert abs((w - d) + gamma) < 1e-9, (w, gamma)
    n = 200; nstar = None
    while n < 1e8:
        b = beta_n(3, 4, n, 2, 0.05)
        passes = sum(certificate(m, sample_Phat(m.P, n, rng), 0, d, 0.05, b, b)[0] for _ in range(100))
        if passes >= 90: nstar = n; break
        n = int(n * 1.15) + 1
    pB = d + gamma
    n_lb = pB * (1 - pB) * math.log(1 / (4 * 0.05)) / (4 * gamma ** 2)
    D["rows"].append({"gamma": gamma, "n_star_certificate": nstar, "n_star_times_gamma2": nstar * gamma ** 2 if nstar else None,
                      "n_lower_bound": n_lb, "n_lb_times_gamma2": n_lb * gamma ** 2})
    print(f"[D] gamma={gamma}: n*={nstar}  n*gamma^2={(nstar*gamma**2) if nstar else float('nan'):.3f}  LB={n_lb:.1f}")
OUT["D_scaling"] = D

# ---------------- (E) price of ambiguity: PoA <= V_1(d) - V_1(d - rho_1) <= rho_1 * shadow price ----------------
E = {"instances": 0, "violations_exact_bound": 0, "violations_shadow_bound": 0, "examples": []}
E["nonvacuous"] = 0
for _ in range(60):
    m = random_instance(rng, K=3)
    base = m.costs[0]
    for k in (1, 2):                                   # close readings: perturb the first reading on a random support
        mask = rng.uniform(size=base.shape) < 0.5
        m.costs[k] = np.clip(base + np.where(mask, rng.uniform(-0.15, 0.15, size=base.shape), 0.0), 0.0, 1.0)
    for d in np.linspace(0.3, 1.5, 6):
        Vs = [V(m, k, d) for k in range(3)]
        if any(v is None for v in Vs): continue
        Vset = V_set(m, d)
        if Vset is None: continue
        poa = max(Vs) - Vset
        k1 = int(np.argmax(Vs))
        rho = max(m.lp(m.vec(m.costs[phi] - m.costs[k1])) for phi in range(3) if phi != k1)   # one-sided disagreement
        rho = max(rho, 0.0)
        Vlow = V(m, k1, d - rho)
        E["instances"] += 1
        if Vlow is None: continue          # bound vacuous when d - rho is infeasible
        E["nonvacuous"] += 1
        bound1 = Vs[k1] - Vlow
        if poa > bound1 + 1e-7: E["violations_exact_bound"] += 1
        h = 1e-5; slope = (V(m, k1, d - rho + h) - Vlow) / h   # right derivative = shadow price at d - rho
        bound2 = rho * slope
        if poa > bound2 + 1e-6: E["violations_shadow_bound"] += 1
        if len(E["examples"]) < 4: E["examples"].append({"d": float(d), "PoA": float(poa), "rho": float(rho), "bound_V": float(bound1), "shadow_price": float(slope), "bound_rho_lambda": float(bound2)})
OUT["E_poa"] = E
print("[E] PoA bounds:", json.dumps(E, indent=1))

# ---------------- (F) budget structure: D_psi as a union of intervals ----------------
F = []
for _ in range(30):
    m = random_instance(rng, K=2)
    psi = 0
    if dominated(m, psi): continue
    grid = np.linspace(0.02, 1.6, 240)
    signs = []
    for d in grid:
        w = W_eps(m, psi, d, 0.0)
        signs.append(None if w is None else (w <= d + 1e-9))
    seq = [s for s in signs if s is not None]
    intervals = sum(1 for i, s in enumerate(seq) if s and (i == 0 or not seq[i - 1]))
    F.append({"sufficient_intervals": intervals, "frac_sufficient": float(np.mean(seq)) if seq else None})
OUT["F_budget_structure"] = F
# explicit non-monotone instance: one step, actions z (r=0), b (r=0.6, c_psi=0.2, c_phi=0.15), a (r=1, c_psi=0.5, c_phi=1)
Pe = np.zeros((1, 3, 1)); Pe[:, :, 0] = 1.0
re = np.array([[0.0, 0.6, 1.0]]); cps = np.array([[0.0, 0.2, 0.5]]); cph = np.array([[0.0, 0.15, 1.0]])
me = CMDP(Pe, np.array([1.0]), re, [cps, cph], 1)
grid = np.round(np.arange(0.02, 1.3001, 0.02), 4)
suff_d = [float(d) for d in grid if W_eps(me, 0, float(d), 0.0) <= d + 1e-9]
runs = []
for d in suff_d:
    if runs and abs(d - runs[-1][-1] - 0.02) < 1e-6: runs[-1].append(d)
    else: runs.append([d])
Fx = {"sufficient_budgets_intervals": [[r[0], r[-1]] for r in runs], "n_intervals": len(runs),
      "margins": {str(d): float(W_eps(me, 0, d, 0.0) - d) for d in [0.1, 0.2, 0.3, 0.7, 1.0, 1.2]}}
OUT["F_explicit_nonmonotone"] = Fx
print("[F-explicit] sufficiency intervals:", Fx["sufficient_budgets_intervals"], "margins:", {k: round(v, 3) for k, v in Fx["margins"].items()})
print("[F] sufficiency intervals per instance:", [f["sufficient_intervals"] for f in F])

OUT["runtime_s"] = time.time() - T0
json.dump(OUT, open("check_theory_results.json", "w"), indent=1, default=float)
print("done in %.1fs" % OUT["runtime_s"])
