#!/usr/bin/env python3
"""V30: what does each certificate cost in data?  Value gap vs certified face.

The paper proves two finite-data statements about the same compiled instance:

  Gamma route: certify that EVERY eps-optimal policy under psi is full-set
    safe.  Needs a conservative empirical eps-face -- one LP per competing
    reading over a confidence-inflated polytope.  This is the certificate the
    archived study (real_rules_experiment.py) already measures.

  Delta route: certify that SOME eps-optimal policy under psi is full-set
    safe.  By the recoverability theorem this is exactly
        Vbar_psi - Vlow_U <= eps,
    two value bounds and no face at all.

Both are run here on the same 28 compiled rules with the SAME structured
load-chain estimator, the same Weissman/Bernstein radius and the same centered
simulation lemma, so any difference in data cost is due to what is certified,
not to how the confidence set is built.

Writes paper/theory_extension/delta_vs_gamma.json
"""
import json
import os
import math
import sys
import time

import numpy as np
from scipy.optimize import linprog

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(str(REPO), "src"))
from saorl.benchmark_sg import control_suite as cs  # noqa: E402

GAMMA = cs.GAMMA
OUT = REPO + "/results/theory_extension/"

LSIM = GAMMA / (2 * (1 - GAMMA))
SPAN_R, SPAN_C = 0.7, 1.0
DELTA = 0.05
EPS = 0.01
BUDGETS = [0.005, 0.02, 0.05]
N_GRID = [10 ** (x / 2) for x in range(8, 21)]      # 1e4 ... 1e10
REPS = 10
SEED = 20260904


class _R:
    def __init__(self, theta, for_s):
        self.threshold, self.for_s = theta, for_s


def compile_with(readings, M):
    orig = cs._load_matrix
    cs._load_matrix = lambda L: M
    try:
        return cs.compile_instance(readings)
    finally:
        cs._load_matrix = orig


def flow(m):
    nS, nA = m["nS"], m["nA"]
    block = np.kron(np.eye(nS), np.ones((1, nA)))
    return (block - GAMMA * np.transpose(m["P"], (2, 0, 1)).reshape(nS, nS * nA),
            (1 - GAMMA) * m["mu0"])


def lp(obj, rows, rhs, A, b):
    res = linprog(-obj, A_ub=np.array(rows) if rows else None,
                  b_ub=np.array(rhs) if rows else None,
                  A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    return None if res.status != 0 else float(obj @ res.x)


class Rule:
    def __init__(self, inst, idx=0):
        self.uid = f'{inst["uid"]}#{idx}'
        self.readings = [_R(x["theta"], x["for_s"]) for x in inst["readings"]]
        self.m = cs.compile_instance(self.readings)
        self.A, self.b = flow(self.m)
        self.r = self.m["r"].reshape(-1)
        self.c = [self.m["C"][k].reshape(-1) for k in range(self.m["C"].shape[0])]
        self.K = len(self.c)

    def V(self, k, d, A=None, b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        return lp(self.r, [self.c[k]], [d], A, b)

    def V_set(self, d, A=None, b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        return lp(self.r, self.c, [d] * self.K, A, b)

    def W(self, k, d, eps, A=None, b=None, Vk=None, dplus=None, thr=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        if Vk is None:
            Vk = self.V(k, d, A, b)
        if Vk is None:
            return None
        dplus = d if dplus is None else dplus
        thr = (Vk - eps) if thr is None else thr
        best = -np.inf
        for j in range(self.K):
            if j == k:
                continue
            w = lp(self.c[j], [self.c[k], -self.r], [dplus, -thr], A, b)
            if w is not None:
                best = max(best, w)
        return best


def radius(Mhat, nrow, bound):
    out = []
    for i in range(3):
        n = nrow[i]
        if bound == "weissman":
            out.append(math.sqrt(2 * (3 * math.log(2) + math.log(3 / DELTA)) / n))
        else:
            Lg = math.log(2 * 3 * 3 / DELTA)
            p = Mhat[i]
            out.append(min(2.0, float(np.sum(
                np.sqrt(2 * p * (1 - p) * n / (n - 1) * Lg / n)
                + 7 * Lg / (3 * (n - 1))))))
    return max(out)


def cert_gamma(R, mhat, k, d, eps, beta_r, beta_c):
    """The archived face certificate (real_rules_experiment.certificate)."""
    A, b = flow(mhat)
    Vm = R.V(k, d - beta_c, A, b)
    if Vm is None:
        return False, None
    Vm -= beta_r
    w = R.W(k, d, eps, A, b, Vk=Vm, dplus=d + beta_c, thr=Vm - beta_r - eps)
    if w is None:
        return False, None
    return (w + beta_c <= d + 1e-9), w + beta_c


def cert_delta(R, mhat, k, d, eps, beta_r, beta_c):
    """Value-gap certificate: an upper bound on V_psi minus a lower bound on V_U.

    Relaxing the estimated budget by beta_c makes the estimated feasible set
    contain the true one, so the estimated optimum plus beta_r upper-bounds
    V_psi; tightening it by beta_c makes the estimated set contained in the
    true one, so the estimated optimum minus beta_r lower-bounds V_U.
    """
    A, b = flow(mhat)
    V_hi = R.V(k, d + beta_c, A, b)
    V_lo = R.V_set(d - beta_c, A, b)
    if V_hi is None or V_lo is None:
        return False, None
    gap_hi = (V_hi + beta_r) - (V_lo - beta_r)
    return bool(gap_hi <= eps + 1e-12), float(gap_hi)


def main():
    suite = json.load(open(REPO + "/results/e2e/control_suite_uncapped.json"))
    rules = [Rule(i, j) for j, i in enumerate(suite["instances"])]
    assert len(rules) == 28
    M_TRUE = cs._load_matrix(3)
    w, v = np.linalg.eig(M_TRUE.T)
    pi = np.real(v[:, np.argmin(abs(w - 1))])
    pi = pi / pi.sum()

    # ---- exact truth for both questions --------------------------------
    truth = {}
    for R in rules:
        truth[R.uid] = {}
        for d in BUDGETS:
            V_U = R.V_set(d)
            rec = {}
            for k in range(R.K):
                Vk = R.V(k, d)
                if Vk is None or V_U is None:
                    rec[str(k)] = None
                    continue
                Wk = R.W(k, d, EPS, Vk=Vk)
                rec[str(k)] = dict(
                    V=Vk, V_U=V_U,
                    delta_true=bool(Vk - V_U <= EPS + 1e-9),
                    gamma_true=bool(Wk is not None and Wk <= d + 1e-9))
            truth[R.uid][str(d)] = rec
    for d in BUDGETS:
        nd = sum(1 for R in rules if any((truth[R.uid][str(d)][str(k)] or {})
                                         .get("delta_true") for k in range(R.K)))
        ng = sum(1 for R in rules if any((truth[R.uid][str(d)][str(k)] or {})
                                         .get("gamma_true") for k in range(R.K)))
        print(f"exact d={d}: instances with a Delta-recoverable reading {nd}/28,"
              f" with a Gamma-safe face {ng}/28")

    # ---- finite-data sweep ---------------------------------------------
    rng = np.random.default_rng(SEED)
    records = []
    t0 = time.time()
    for n_total in N_GRID:
        nrow = [max(2, int(round(n_total * pi[i]))) for i in range(3)]
        for rep in range(REPS):
            Mhat = np.array([rng.multinomial(nrow[i], M_TRUE[i]) / nrow[i]
                             for i in range(3)])
            rad = radius(Mhat, nrow, "bernstein")
            beta = LSIM * rad
            beta_r, beta_c = beta * SPAN_R, beta * SPAN_C
            for R in rules:
                mhat = compile_with(R.readings, Mhat)
                for d in BUDGETS:
                    for k in range(R.K):
                        t = truth[R.uid][str(d)][str(k)]
                        if t is None:
                            continue
                        gp, gw = cert_gamma(R, mhat, k, d, EPS, beta_r, beta_c)
                        dp, dg = cert_delta(R, mhat, k, d, EPS, beta_r, beta_c)
                        records.append(dict(
                            n=n_total, rep=rep, uid=R.uid, k=k, d=d,
                            beta=beta,
                            delta_true=t["delta_true"], gamma_true=t["gamma_true"],
                            delta_pass=bool(dp), gamma_pass=bool(gp),
                            gap_hi=dg, What=gw))
        print(f"n_total={n_total:.3g}: done ({time.time() - t0:.0f}s)")

    # ---- summary --------------------------------------------------------
    summary = {}
    for d in BUDGETS:
        per_n = {}
        for n_total in N_GRID:
            rr = [x for x in records if x["n"] == n_total and x["d"] == d]
            dt = [x for x in rr if x["delta_true"]]
            gt = [x for x in rr if x["gamma_true"]]
            per_n[f"{n_total:.3g}"] = dict(
                delta_recall=(float(np.mean([x["delta_pass"] for x in dt]))
                              if dt else None),
                gamma_recall=(float(np.mean([x["gamma_pass"] for x in gt]))
                              if gt else None),
                delta_false=int(sum(1 for x in rr
                                    if x["delta_pass"] and not x["delta_true"])),
                gamma_false=int(sum(1 for x in rr
                                    if x["gamma_pass"] and not x["gamma_true"])),
                n_delta_true=len(dt), n_gamma_true=len(gt), n_rows=len(rr))
        # smallest n at which each reaches 50% and 90% recall
        def first_at(key, thresh):
            for n_total in N_GRID:
                v = per_n[f"{n_total:.3g}"][key]
                if v is not None and v >= thresh:
                    return n_total
            return None
        summary[str(d)] = dict(
            per_n=per_n,
            delta_n50=first_at("delta_recall", 0.5),
            gamma_n50=first_at("gamma_recall", 0.5),
            delta_n90=first_at("delta_recall", 0.9),
            gamma_n90=first_at("gamma_recall", 0.9))
        s = summary[str(d)]
        print(f"d={d}: Delta reaches 50% recall at n={s['delta_n50']:.3g}"
              if s["delta_n50"] else f"d={d}: Delta never reaches 50%", end="")
        print(f", Gamma at n={s['gamma_n50']:.3g}"
              if s["gamma_n50"] else ", Gamma never reaches 50%")

    json.dump(dict(
        registration="V30: value-gap (Delta) vs certified-face (Gamma) "
                     "certificates on the 28 compiled rules, same structured "
                     f"load-chain estimator, empirical-Bernstein radius, "
                     f"eps={EPS}, delta={DELTA}, {REPS} reps per n",
        eps=EPS, delta=DELTA, budgets=BUDGETS, n_grid=N_GRID, reps=REPS,
        summary=summary), open(OUT + "delta_vs_gamma.json", "w"), indent=1)
    json.dump(records, open(OUT + "delta_vs_gamma_records.json", "w"))
    print("wrote", OUT + "delta_vs_gamma.json")


if __name__ == "__main__":
    main()
