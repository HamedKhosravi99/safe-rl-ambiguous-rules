"""V50: a tool-using agent under a SCOPE-ambiguous call-limit rule.

Rule (natural language): "Keep calls to the external service within the plan's
limit -- no more than N calls per session."  The agent works user requests that
decompose into sub-tasks, each resolved by ONE call to one of K external services
(search API, database, messaging, translation); a resolved auxiliary sub-task is worth 1, primary sub-tasks differ mildly in
importance (0.82 to 1.0, so the agent prioritises), a call with nothing left to
resolve is wasted (worth 0), and a step of local work
(the free fallback) is worth R_LOCAL.  The rule is ambiguous in SCOPE: which
service is "the external service"?  Reading psi_k charges one unit per call to
service k.  The readings are mutually exclusive interpretations, so their firing
regions are disjoint by action type and none dominates another (a crossing
antichain of width K).

Model.  Discounted continuing CMDP (gamma = GAMMA; the session ends with
probability 1-gamma per step, i.e. about 1/(1-gamma) steps).  State = remaining
sub-tasks per service (m_1, ..., m_K); at session start the demand profile is drawn
from a small product distribution (the primary service has many sub-tasks, the
auxiliaries a few).  Actions: call service k (k = 1..K) or local work.  The budget
d bounds the discounted occupancy rate of the charged calls.

Nothing is fitted to the budget: the demand profiles are a fixed primary/auxiliary
hierarchy (three variants: light, medium, heavy auxiliary demand), d is swept, and
the whole surface is reported, including the budgets at which no reading is
sufficient.

Exact analysis per (variant, K, d): V_U, V_surr (pointwise-max / union), V_k per
reading, Decide's face test (eps = EPS), surrogate price (V_U - V_surr)/V_U.
Learned arms (V49 protocol, the same single-signal learners) on the instances
Decide certifies.  Registration V50.

Run:  OMP_NUM_THREADS=1 PYTHONPATH=src python3 -m saorl.benchmark_sg.scope_agent [--exact-only]
Writes results/e2e/scope_agent.json
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import hashlib
import itertools
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

GAMMA = 0.97
R_LOCAL = 0.2             # value of a step of local work (the free fallback)
V_ITEM = 1.0              # value of a resolved auxiliary sub-task
PRIMARY_SLOPE = 0.02      # primary sub-tasks differ mildly in importance: with j remaining, the next is worth 0.8 + 0.02 j
SERVICES = ["search", "database", "messaging", "translation"]
PRIMARY_DEMAND = (6, 8, 10)                      # sub-tasks for the primary service, uniform
AUX_DEMAND = {"light": (1, 1, 1), "medium": (2, 2, 1), "heavy": (3, 3, 2)}   # aux k has s_k or s_k+1 sub-tasks, uniform
K_GRID = (2, 3, 4)
D_GRID = (0.05, 0.075, 0.10, 0.15)
EPS = 0.01

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
R = os.path.join(ROOT, "results/e2e")
OUT = os.path.join(R, "scope_agent.json")


def build(K: int, variant: str) -> dict:
    """Tabular CMDP in the compiled-instance format (S, nS, nA, P, r, C, mu0)."""
    aux = AUX_DEMAND[variant][:K - 1]
    maxes = [max(PRIMARY_DEMAND)] + [s + 1 for s in aux]
    S = list(itertools.product(*[range(mx + 1) for mx in maxes]))
    idx = {s: i for i, s in enumerate(S)}
    nS, nA = len(S), K + 1
    P = np.zeros((nS, nA, nS)); r = np.zeros((nS, nA)); C = np.zeros((K, nS, nA))
    for s in S:
        i = idx[s]
        for k in range(K):                               # call service k
            if s[k] > 0:
                s2 = list(s); s2[k] -= 1; P[i, k, idx[tuple(s2)]] = 1.0
                r[i, k] = (0.8 + PRIMARY_SLOPE * s[k]) if k == 0 else V_ITEM
            else:
                P[i, k, i] = 1.0; r[i, k] = 0.0          # wasted call
            C[k, i, k] = 1.0
        P[i, K, i] = 1.0; r[i, K] = R_LOCAL              # local work
    mu0 = np.zeros(nS)
    profiles = list(itertools.product(PRIMARY_DEMAND, *[(s, s + 1) for s in aux]))
    for pr in profiles:
        mu0[idx[tuple(pr)]] += 1.0 / len(profiles)
    return dict(S=S, nS=nS, nA=nA, P=P, r=r, C=C, mu0=mu0, K=K, names=SERVICES[:K], variant=variant)


def flow(m):
    if "_flow" not in m:
        nS, nA = m["nS"], m["nA"]
        rows, cols, vals = [], [], []
        for sp in range(nS):
            for a in range(nA):
                rows.append(sp); cols.append(sp * nA + a); vals.append(1.0)
        for s, a, sp in np.argwhere(m["P"] > 0):
            rows.append(sp); cols.append(s * nA + a); vals.append(-GAMMA * m["P"][s, a, sp])
        m["_flow"] = (sparse.csr_matrix((vals, (rows, cols)), shape=(nS, nS * nA)), (1.0 - GAMMA) * m["mu0"])
    return m["_flow"]


def lp(m, obj, rows, rhs):
    A_eq, b_eq = flow(m)
    res = linprog(-obj, A_ub=sparse.csr_matrix(np.array(rows)) if rows else None, b_ub=np.array(rhs) if rows else None,
                  A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return (None, None) if res.status != 0 else (float(obj @ res.x), res.x)


def exact(m, d):
    K = m["K"]; r = m["r"].reshape(-1); c = [m["C"][k].reshape(-1) for k in range(K)]
    cmax = np.max(m["C"], axis=0).reshape(-1)
    V_unc, x_unc = lp(m, r, [], []); V_U, x_U = lp(m, r, c, [d] * K); V_surr, _ = lp(m, r, [cmax], [d])
    out = dict(V_unc=V_unc, V_U=V_U, V_surr=V_surr, price=(V_U - V_surr) / V_U,
               rates_unconstrained=[float(c[k] @ x_unc) for k in range(K)], readings=[])
    for psi in range(K):
        V_psi, x_psi = lp(m, r, [c[psi]], [d])
        Ws = [lp(m, c[phi], [c[psi], -r], [d, -(V_psi - EPS)])[0] for phi in range(K) if phi != psi]
        W = max(w for w in Ws if w is not None)
        out["readings"].append(dict(name=m["names"][psi], V_psi=V_psi, margin=W - d, sufficient=bool(W <= d + 1e-9),
                                    rates_at_psi_optimum=[float(c[k] @ x_psi) for k in range(K)]))
    out["certified"] = [rd["name"] for rd in out["readings"] if rd["sufficient"]]
    return out


def main():
    res = dict(registration="V50 scope-ambiguous multi-service agent", gamma=GAMMA, r_local=R_LOCAL, v_item=V_ITEM,
               primary_demand=PRIMARY_DEMAND, aux_demand=AUX_DEMAND, k_grid=list(K_GRID), d_grid=list(D_GRID), eps=EPS, exact={})
    print(f"{'var':6s} {'K':>2s} {'d':>5s} | {'V_unc':>6s} {'V_U':>6s} {'V_surr':>6s} {'price':>6s} | unconstrained call rates | certified   | margin of psi* | rates at psi*-optimum")
    for variant in AUX_DEMAND:
        for K in K_GRID:
            m = build(K, variant)
            for d in D_GRID:
                e = exact(m, d); res["exact"][f"{variant}|K{K}|d{d}"] = e
                cert = ",".join(e["certified"]) or "-"
                rd = next((x for x in e["readings"] if x["sufficient"]), None)
                print(f"{variant:6s} {K:2d} {d:5.2f} | {e['V_unc']:6.3f} {e['V_U']:6.3f} {e['V_surr']:6.3f} {100*e['price']:5.1f}% | {np.round(e['rates_unconstrained'],3)} | {cert:11s} | {rd['margin'] if rd else float('nan'):+.3f} | {np.round(rd['rates_at_psi_optimum'],3) if rd else '-'}")
    json.dump(res, open(OUT, "w")); print("wrote", OUT)


if __name__ == "__main__":
    main()
