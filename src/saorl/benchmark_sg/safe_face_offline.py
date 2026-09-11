"""V29: offline learning on the optimizer-resolvable compiled instances.

V28 showed with exact LPs that a full-set-safe optimum exists on the anchor's
face and that an exact selector finds it. This module asks whether a learner
with only logged transitions can, and compares four constrained offline
learners trained from the SAME logs:

  single    Lagrangian on the anchor cost only            (what a practitioner
                                                           who trusts one
                                                           compilation does)
  fullset   K multipliers, one per retained reading       (full-set protection)
  safeface  alpha r - sum_phi q_phi c_phi - lambda c_psi  (the paper's minimax,
            with q by exponentiated gradient               Eqs. 13-14)
  uniform_q safeface with q pinned uniform                (mean, not max)

All are tabular fitted-Q iteration with a behaviour-support restriction; the
true transition model is never shown to the learner. Training-side value
estimates come from a count-based model fitted on the same log. The returned
policy is scored EXACTLY on the true model against the complete retained set,
so evaluation carries no Monte-Carlo noise.

Registration: results/e2e/REGISTRATION_V28.md (V29 addendum).
Run: PYTHONPATH=. python3 -m saorl.benchmark_sg.safe_face_offline
Writes results/e2e/safe_face_offline.json
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

from .control_mdp import GAMMA
from .control_suite import _eligible, _flow, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SEL = os.path.join(ROOT, "results/e2e", "safe_face_select.json")
OUT = os.path.join(ROOT, "results/e2e", "safe_face_offline.json")

BUDGET = 0.005
N_GRID = (2000, 20000)
SEEDS = tuple(range(30))
EPS_GRID = (0.0, 0.01)
KAPPA_FRAC = (0.0, 0.1, 0.2, 0.3, 0.5)   # registered V31 grid, kappa/d
N_EV = 20000                              # Check evaluation samples
DELTA_EV = 0.05
BEHAV_MIX = 0.3          # uniform mass in the logging policy
N_MIN = 3                # behaviour-support threshold
FQI_ITERS = 200
DUAL_ITERS = 60
ETA_Q = 2.0
ETA_DUAL = 400.0
TOL = 1e-9
SAFE_TOL = 1e-6


# --------------------------------------------------------------------------
# exact evaluation on the TRUE model
# --------------------------------------------------------------------------
def occupancy(P: np.ndarray, mu0: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Normalized discounted state-action occupancy of pi under P.

    Solves d = (1-g) mu0 + g P_pi' d and returns x[s,a] = d[s] pi[s,a], so
    that sum(x) = 1 and r'x is the normalized return used by the LPs.
    """
    nS, nA = pi.shape
    P_pi = np.einsum("sa,sat->st", pi, P)
    d = np.linalg.solve(np.eye(nS) - GAMMA * P_pi.T, (1.0 - GAMMA) * mu0)
    return d[:, None] * pi


def score(P, mu0, r, C, pi, d_budget) -> dict:
    x = occupancy(P, mu0, pi)
    cmax = float(max((C[k] * x).sum() for k in range(C.shape[0])))
    return dict(J_r=float((r * x).sum()), C_max=cmax,
                margin=float(d_budget - cmax),
                safe=bool(cmax <= d_budget + SAFE_TOL))


def check_ship(P, mu0, C, pi, d_budget, n_ev, rng, delta_ev=0.05):
    """The deployment check of Eq. (14): an independent empirical-Bernstein
    upper bound on max_phi J_{c_phi}(pi), Bonferroni over the K readings.

    Samples n_ev fresh state-action pairs from the learned policy's discounted
    occupancy under the TRUE model, which is what an operator's shadow run
    would produce, and ships only if the bound clears the budget.
    """
    K = C.shape[0]
    x = occupancy(P, mu0, pi).reshape(-1)
    x = np.maximum(x, 0)
    x = x / x.sum()
    idx = rng.choice(x.size, size=n_ev, p=x)
    dpp = delta_ev / K
    L = np.log(3.0 / dpp)
    bound = -np.inf
    for k in range(K):
        z = C[k].reshape(-1)[idx]
        mean, var = float(z.mean()), float(z.var(ddof=1))
        rng_c = float(C[k].max()) or 1.0
        b = mean + np.sqrt(2.0 * var * L / n_ev) + 3.0 * rng_c * L / n_ev
        bound = max(bound, b)
    return dict(bound=float(bound), ship=bool(bound <= d_budget))


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------
def behaviour_policy(m: dict, anchor: int, d_budget: float) -> np.ndarray:
    """0.7 x the single-reading optimal policy + 0.3 x uniform."""
    nS, nA = m["nS"], m["nA"]
    A_eq, b_eq = _flow(m)
    r = m["r"].reshape(-1)
    c = m["C"][anchor].reshape(-1)
    res = linprog(-r, A_ub=np.array([c]), b_ub=np.array([d_budget]),
                  A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    pi = np.full((nS, nA), 1.0 / nA)
    if res.status == 0:
        x = res.x.reshape(nS, nA)
        tot = x.sum(axis=1)
        vis = tot > 1e-12
        pi[vis] = x[vis] / tot[vis, None]
    return (1.0 - BEHAV_MIX) * pi + BEHAV_MIX / nA


def sample_log(m: dict, pi_b: np.ndarray, n: int, rng) -> dict:
    """Sample n transitions from the discounted occupancy of pi_b.

    Restarting with probability 1-gamma makes the visitation distribution the
    normalized discounted occupancy, matching the LP normalization.
    """
    P, mu0, r, C = m["P"], m["mu0"], m["r"], m["C"]
    nS, nA, K = m["nS"], m["nA"], C.shape[0]
    S = np.empty(n, dtype=np.int64)
    A = np.empty(n, dtype=np.int64)
    SP = np.empty(n, dtype=np.int64)
    R = np.empty(n)
    Cc = np.empty((K, n))
    s = rng.choice(nS, p=mu0)
    for i in range(n):
        a = rng.choice(nA, p=pi_b[s])
        sp = rng.choice(nS, p=P[s, a])
        S[i], A[i], SP[i], R[i] = s, a, sp, r[s, a]
        for k in range(K):
            Cc[k, i] = C[k, s, a]
        s = sp if rng.random() < GAMMA else rng.choice(nS, p=mu0)
    return dict(S=S, A=A, SP=SP, R=R, C=Cc, nS=nS, nA=nA, K=K, mu0=mu0)


# --------------------------------------------------------------------------
# tabular pieces the learners share
# --------------------------------------------------------------------------
class Log:
    """Sufficient statistics of a log: counts, per-(s,a) means, empirical P."""

    def __init__(self, lg: dict):
        nS, nA, K = lg["nS"], lg["nA"], lg["K"]
        self.nS, self.nA, self.K, self.mu0 = nS, nA, K, lg["mu0"]
        idx = lg["S"] * nA + lg["A"]
        self.N = np.bincount(idx, minlength=nS * nA).reshape(nS, nA)
        self.r = (np.bincount(idx, weights=lg["R"], minlength=nS * nA)
                  .reshape(nS, nA))
        self.c = np.stack([np.bincount(idx, weights=lg["C"][k],
                                       minlength=nS * nA).reshape(nS, nA)
                           for k in range(K)])
        nz = np.maximum(self.N, 1)
        self.r = self.r / nz
        self.c = self.c / nz
        self.Phat = np.zeros((nS, nA, nS))
        np.add.at(self.Phat, (lg["S"], lg["A"], lg["SP"]), 1.0)
        tot = self.Phat.sum(axis=2, keepdims=True)
        self.Phat = np.where(tot > 0, self.Phat / np.maximum(tot, 1), 0.0)
        # unsupported (s,a): self-loop, so the learned model does not invent
        # optimistic transitions there
        for s in range(nS):
            for a in range(nA):
                if self.N[s, a] == 0:
                    self.Phat[s, a, s] = 1.0
        self.support = self.N >= N_MIN
        # every state needs at least one admissible action
        for s in range(nS):
            if not self.support[s].any():
                self.support[s, int(np.argmax(self.N[s]))] = True

    def fqi(self, shaped: np.ndarray) -> np.ndarray:
        """Fitted-Q iteration on the log for a shaped per-(s,a) reward."""
        Q = np.zeros((self.nS, self.nA))
        for _ in range(FQI_ITERS):
            V = np.where(self.support, Q, -np.inf).max(axis=1)
            V = np.where(np.isfinite(V), V, 0.0)
            Qn = shaped + GAMMA * self.Phat @ V
            if np.max(np.abs(Qn - Q)) < 1e-10:
                Q = Qn
                break
            Q = Qn
        return Q

    def greedy(self, Q: np.ndarray) -> np.ndarray:
        pi = np.zeros((self.nS, self.nA))
        masked = np.where(self.support, Q, -np.inf)
        pi[np.arange(self.nS), masked.argmax(axis=1)] = 1.0
        return pi

    def V_hat(self, k: int, d_budget: float) -> float:
        """Estimate V_psi(d) from the log alone: the constrained LP on Phat,
        restricted to behaviour-supported state-action pairs."""
        nS, nA = self.nS, self.nA
        block = np.kron(np.eye(nS), np.ones((1, nA)))
        A_eq = block - GAMMA * np.transpose(self.Phat, (2, 0, 1)).reshape(nS, nS * nA)
        b_eq = (1.0 - GAMMA) * self.mu0
        bounds = [(0, None)] * (nS * nA)
        for s in range(nS):
            for a in range(nA):
                if not self.support[s, a]:
                    bounds[s * nA + a] = (0, 0)
        res = linprog(-self.r.reshape(-1),
                      A_ub=np.array([self.c[k].reshape(-1)]),
                      b_ub=np.array([d_budget]),
                      A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        return float(-res.fun) if res.status == 0 else 0.0

    def V_hat_set(self, d_budget: float) -> float:
        """Estimate V_U(d) from the log alone: the LP on Phat with EVERY
        retained cost constrained, restricted to supported pairs."""
        nS, nA = self.nS, self.nA
        block = np.kron(np.eye(nS), np.ones((1, nA)))
        A_eq = block - GAMMA * np.transpose(self.Phat, (2, 0, 1)).reshape(nS, nS * nA)
        b_eq = (1.0 - GAMMA) * self.mu0
        bounds = [(0, None)] * (nS * nA)
        for s in range(nS):
            for a in range(nA):
                if not self.support[s, a]:
                    bounds[s * nA + a] = (0, 0)
        res = linprog(-self.r.reshape(-1),
                      A_ub=np.array([self.c[k].reshape(-1)
                                     for k in range(self.K)]),
                      b_ub=np.array([d_budget] * self.K),
                      A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        return float(-res.fun) if res.status == 0 else 0.0

    def est(self, pi: np.ndarray) -> Tuple[float, np.ndarray]:
        """Training-side estimates of (return, per-reading costs) under Phat."""
        x = occupancy(self.Phat, self.mu0, pi)
        return float((self.r * x).sum()), np.array(
            [float((self.c[k] * x).sum()) for k in range(self.K)])


# --------------------------------------------------------------------------
# the four learners
# --------------------------------------------------------------------------
def learn_single(lg: Log, anchor: int, d_budget: float) -> np.ndarray:
    lam, best = 0.0, None
    for _ in range(DUAL_ITERS):
        pi = lg.greedy(lg.fqi(lg.r - lam * lg.c[anchor]))
        _jr, jc = lg.est(pi)
        if jc[anchor] <= d_budget + TOL:
            best = pi
            break
        lam += ETA_DUAL * (jc[anchor] - d_budget)
    return best if best is not None else pi


def learn_fullset(lg: Log, anchor: int, d_budget: float) -> np.ndarray:
    lam = np.zeros(lg.K)
    best = None
    for _ in range(DUAL_ITERS):
        shaped = lg.r - np.tensordot(lam, lg.c, axes=(0, 0))
        pi = lg.greedy(lg.fqi(shaped))
        _jr, jc = lg.est(pi)
        if (jc <= d_budget + TOL).all():
            best = pi
            break
        lam = np.maximum(0.0, lam + ETA_DUAL * (jc - d_budget))
    return best if best is not None else pi


def _minimax(lg: Log, anchor: int, d_face: float, d_target: float,
             floor: float, uniform_q: bool) -> np.ndarray:
    """Shared minimax loop: the face constraint stays at d_face, the retained
    costs are driven toward d_target, and the return is floored at `floor`."""
    q = np.full(lg.K, 1.0 / lg.K)
    alpha, lam = 1.0, 0.0
    cands: List[Tuple[float, float, np.ndarray]] = []
    for _ in range(DUAL_ITERS):
        shaped = (alpha * lg.r - np.tensordot(q, lg.c, axes=(0, 0))
                  - lam * lg.c[anchor])
        pi = lg.greedy(lg.fqi(shaped))
        jr, jc = lg.est(pi)
        cands.append((float(jc.max()), jr, pi))
        if not uniform_q:
            w = q * np.exp(ETA_Q * jc / max(d_target, 1e-12))
            q = w / w.sum()
        alpha = max(0.0, alpha + ETA_DUAL * (floor - jr))
        lam = max(0.0, lam + ETA_DUAL * (jc[anchor] - d_face))
    ok = [c for c in cands if c[1] >= floor - 1e-9]
    pool = ok if ok else cands
    return min(pool, key=lambda c: c[0])[2]


def learn_safeface(lg: Log, anchor: int, d_budget: float,
                   eps: float, uniform_q: bool = False,
                   kappa: Optional[float] = None):
    """alpha r - sum_phi q_phi c_phi - lambda c_anchor, q by exp. gradient.

    The reward floor uses V_hat_psi(d) estimated from the SAME log, never the
    true value: the learner sees only what the other learners see.
    """
    V_anchor = lg.V_hat(anchor, d_budget)
    if kappa is not None:
        # margin-aware arm: ask for Delta <= -kappa, which by the margin
        # theorem is V_psi(d) - V_U(d-kappa) <= eps, and floor the return at
        # the value that buffer actually allows.
        V_setlog = lg.V_hat_set(d_budget - kappa)
        if V_anchor - V_setlog > eps + 1e-9:
            return learn_fullset(lg, anchor, d_budget), True
        return _minimax(lg, anchor, d_budget, d_budget - kappa, V_setlog,
                        uniform_q), False
    # Algorithm 1, Decide: form the recoverable set from the SAME log and fall
    # back to full-set protection when it is empty. Omitting this step is what
    # made the first version of this experiment force an unattainable reward
    # floor; the fallback is part of the algorithm, not a repair.
    V_setlog = lg.V_hat_set(d_budget)
    if V_anchor - V_setlog > eps + 1e-9:
        return learn_fullset(lg, anchor, d_budget), True
    q = np.full(lg.K, 1.0 / lg.K)
    alpha, lam = 1.0, 0.0
    cands: List[Tuple[float, float, np.ndarray]] = []
    for _ in range(DUAL_ITERS):
        shaped = (alpha * lg.r - np.tensordot(q, lg.c, axes=(0, 0))
                  - lam * lg.c[anchor])
        pi = lg.greedy(lg.fqi(shaped))
        jr, jc = lg.est(pi)
        cands.append((float(jc.max()), jr, pi))
        if not uniform_q:
            w = q * np.exp(ETA_Q * jc / max(d_budget, 1e-12))
            q = w / w.sum()
        alpha = max(0.0, alpha + ETA_DUAL * (V_anchor - eps - jr))
        lam = max(0.0, lam + ETA_DUAL * (jc[anchor] - d_budget))
    # the paper's selection rule: among iterates meeting the estimated reward
    # floor, the smallest estimated worst retained cost
    ok = [c for c in cands if c[1] >= V_anchor - eps - 1e-9]
    pool = ok if ok else cands
    return min(pool, key=lambda c: c[0])[2], False


# --------------------------------------------------------------------------
def main() -> None:
    with open(SEL, encoding="utf8") as fh:
        sel = json.load(fh)
    mid = [r for r in sel["rows"]
           if r["regime"] == "optimizer_resolvable"
           and abs(r["budget"] - BUDGET) < 1e-12]
    print(f"{len(mid)} optimizer-resolvable instances at d={BUDGET}")

    pt, _ = parse_prometheus()
    bank = prom_threshold_bank(pt)
    compiled: Dict[str, dict] = {}
    for ti, t in enumerate(pt):
        ok, _w, readings = _eligible(build_prom_pool(t, bank))
        if ok:
            compiled[f'{getattr(t, "name", "?")}#{ti}'] = compile_instance(readings)

    rows: List[dict] = []
    for row in mid:
        m = compiled[row["rule_id"]]
        P, mu0, r, C = m["P"], m["mu0"], m["r"], m["C"]
        anchor = row["anchor"]
        V_anchor = row["V"][anchor]
        V_U = row["V_U"]
        pi_b = behaviour_policy(m, anchor, BUDGET)
        for n in N_GRID:
            for seed in SEEDS:
                rng = np.random.default_rng(
                    abs(hash((row["rule_id"], n, seed))) % (2 ** 32))
                lg = Log(sample_log(m, pi_b, n, rng))
                out, policies = {}, {}
                pi_s = learn_single(lg, anchor, BUDGET)
                pi_f = learn_fullset(lg, anchor, BUDGET)
                out["single"] = score(P, mu0, r, C, pi_s, BUDGET)
                out["fullset"] = score(P, mu0, r, C, pi_f, BUDGET)
                policies["single"], policies["fullset"] = pi_s, pi_f
                for e in EPS_GRID:
                    pi_sf, fb = learn_safeface(lg, anchor, BUDGET, e)
                    sc = score(P, mu0, r, C, pi_sf, BUDGET)
                    sc["fellback"] = bool(fb)
                    out[f"safeface_eps{e}"] = sc
                    pi_u, fbu = learn_safeface(lg, anchor, BUDGET, e,
                                               uniform_q=True)
                    scu = score(P, mu0, r, C, pi_u, BUDGET)
                    scu["fellback"] = bool(fbu)
                    out[f"uniform_q_eps{e}"] = scu
                    policies[f"safeface_eps{e}"] = pi_sf
                    policies[f"uniform_q_eps{e}"] = pi_u
                # margin-aware arm over the registered kappa grid
                for kf in KAPPA_FRAC:
                    pi_k, fbk = learn_safeface(lg, anchor, BUDGET, EPS_GRID[-1],
                                               kappa=kf * BUDGET)
                    sck = score(P, mu0, r, C, pi_k, BUDGET)
                    sck["fellback"] = bool(fbk)
                    out[f"margin_k{kf}"] = sck
                    policies[f"margin_k{kf}"] = pi_k
                # Check every arm on fresh samples from the true model
                crng = np.random.default_rng(
                    abs(hash((row["rule_id"], n, seed, "ev"))) % (2 ** 32))
                for name, pol in policies.items():
                    ck = check_ship(P, mu0, C, pol, BUDGET, N_EV, crng,
                                    DELTA_EV)
                    out[name]["ship"] = ck["ship"]
                    out[name]["bound"] = ck["bound"]
                    out[name]["unsafe_ship"] = bool(ck["ship"]
                                                    and not out[name]["safe"])
                rows.append(dict(rule_id=row["rule_id"], n=n, seed=seed,
                                 V_U=V_U, V_anchor=V_anchor, learners=out))
        print(f"  {row['rule_id']:38s} done")

    # ---- aggregate -----------------------------------------------------
    agg = {}
    for n in N_GRID:
        rs = [x for x in rows if x["n"] == n]
        a = {}
        names = (["single", "fullset"]
                 + [f"{b}_eps{e}" for e in EPS_GRID
                    for b in ("safeface", "uniform_q")]
                 + [f"margin_k{kf}" for kf in KAPPA_FRAC])
        for name in names:
            safe = [x["learners"][name]["safe"] for x in rs]
            ret = [x["learners"][name]["J_r"] / x["V_U"] if x["V_U"] > 0 else
                   np.nan for x in rs]
            cmax = [x["learners"][name]["C_max"] for x in rs]
            a[name] = dict(
                n_runs=len(rs),
                safe_frac=float(np.mean(safe)),
                return_frac_of_VU_med=float(np.nanmedian(ret)),
                return_frac_of_VU_mean=float(np.nanmean(ret)),
                C_max_med=float(np.median(cmax)),
                C_max_over_budget_med=float(np.median(cmax) / BUDGET),
                ship_frac=float(np.mean([x["learners"][name].get("ship", False)
                                         for x in rs])),
                unsafe_ship_frac=float(np.mean(
                    [x["learners"][name].get("unsafe_ship", False)
                     for x in rs])),
                fellback_frac=(float(np.mean([x["learners"][name]["fellback"]
                                              for x in rs]))
                               if "fellback" in rs[0]["learners"][name]
                               else None))
        # per-instance safety, to see whether failures concentrate
        a["_instances_ever_unsafe"] = {
            name: sorted({x["rule_id"] for x in rs
                          if not x["learners"][name]["safe"]})
            for name in names}
        agg[str(n)] = a

    res = dict(
        registration="V29 (REGISTRATION_V28.md addendum): four constrained "
                     "offline learners on the optimizer-resolvable compiled "
                     f"instances at d={BUDGET}; tabular FQI with behaviour "
                     f"support, {len(SEEDS)} seeds, N in {list(N_GRID)}; "
                     "policies scored exactly on the true model",
        budget=BUDGET, n_grid=list(N_GRID), n_seeds=len(SEEDS),
        n_instances=len(mid), aggregate=agg, rows=rows)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {OUT} ({len(rows)} runs)")
    for n in N_GRID:
        print(f"--- N={n} ---")
        for name in agg[str(n)]:
            if name.startswith("_"):
                continue
            v = agg[str(n)][name]
            print(f"  {name:18s} safe {v['safe_frac']:.3f}  "
                  f"SHIP {v['ship_frac']:.3f}  "
                  f"unsafeSHIP {v['unsafe_ship_frac']:.3f}  "
                  f"ret {v['return_frac_of_VU_med']:.3f}  "
                  f"Cmax {v['C_max_over_budget_med']:.3f}x")


if __name__ == "__main__":
    main()
