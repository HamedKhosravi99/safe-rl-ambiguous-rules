"""Source-grounded benchmark, plan 11.7: the genuinely non-dominated control MDP.

Prometheus-derived readings of one CPU alert whose text is ambiguous between:

  * reading A: alert when CPU > 90% for 5 minutes  (short extreme spikes);
  * reading B: alert when CPU > 80% for 15 minutes (sustained moderate load).

Neither dominates the other, so on a control task the retained set U = {A, B}
has nu_Pi(U) > 0 and no singleton reproduces the robust feasible-return frontier
(plan 11.7 completion criteria).  We build a small finite discounted MDP over
the recent-CPU state and solve the *occupancy-measure LP* exactly with
scipy.optimize.linprog(method='highs') -- the same solver saorl/exact_lp.py uses
-- so nu_Pi here is the exact tabular value (plan 11.5: "in finite tabular MDPs
nu_Pi can be computed with occupancy-measure LPs").

Cost of a reading = discounted rate of *continuing while that reading's alert
predicate holds*.  Reward favours continued service (1.0) over intervention
(0.3).
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linprog

GAMMA = 0.97
HR_CAP = 3           # high-run for reading A ( >90% "for 5m", short spike axis )
MR_CAP = 8           # moderate+-run for reading B ( >80% "for 15m", sustained axis )
R_CONT, R_INT = 1.0, 0.3

# load Markov chain on {0:low, 1:moderate(~85%), 2:high(~95%)}.
# Persistent moderate produces sustained load (B fires); persistent high
# produces short spikes that drop before the moderate-run hits MR_CAP (A fires,
# B does not) -- the two readings are genuinely incomparable.
_LOAD_P = np.array([
    [0.55, 0.30, 0.15],   # from low
    [0.22, 0.70, 0.08],   # from moderate  (sustained moderate load)
    [0.55, 0.05, 0.40],   # from high      (spikes fall back to low, not moderate)
])


def _states() -> List[tuple]:
    st = [("safe", 0, 0)]
    for load, hr, mr in product((0, 1, 2), range(HR_CAP + 1), range(MR_CAP + 1)):
        st.append((load, hr, mr))
    return st


def _next_counts(load: int, hr: int, mr: int):
    hr2 = min(HR_CAP, hr + 1) if load == 2 else 0
    mr2 = min(MR_CAP, mr + 1) if load in (1, 2) else 0
    return hr2, mr2


def build_mdp() -> dict:
    S = _states()
    idx = {s: i for i, s in enumerate(S)}
    nS, nA = len(S), 2       # actions: 0=continue, 1=intervene
    P = np.zeros((nS, nA, nS))
    r = np.zeros((nS, nA))
    cA = np.zeros((nS, nA))
    cB = np.zeros((nS, nA))
    for s in S:
        i = idx[s]
        if s[0] == "safe":
            P[i, 0, i] = 1.0
            P[i, 1, i] = 1.0
            r[i, :] = R_INT           # throttled service
            continue
        load, hr, mr = s
        # continue: load evolves, counters update; cost if predicate holds
        for load2 in (0, 1, 2):
            hr2, mr2 = _next_counts(load2, hr, mr)
            P[i, 0, idx[(load2, hr2, mr2)]] += _LOAD_P[load, load2]
        r[i, 0] = R_CONT
        cA[i, 0] = 1.0 if hr >= HR_CAP else 0.0
        cB[i, 0] = 1.0 if mr >= MR_CAP else 0.0
        # intervene: go safe, reduced reward, no cost
        P[i, 1, idx[("safe", 0, 0)]] = 1.0
        r[i, 1] = R_INT
    mu0 = np.zeros(nS)
    # start mixed across both regimes: a fresh high spike and a fresh moderate ramp
    mu0[idx[(2, 1, 1)]] = 0.5     # high-load spike beginning
    mu0[idx[(1, 0, 1)]] = 0.5     # moderate ramp beginning
    return dict(S=S, idx=idx, nS=nS, nA=nA, P=P, r=r, cA=cA, cB=cB, mu0=mu0)


def _flow_matrices(m: dict):
    """Equality constraints of the occupancy polytope:
    sum_a x[s',a] - gamma sum_{s,a} P[s,a,s'] x[s,a] = (1-gamma) mu0[s']."""
    nS, nA = m["nS"], m["nA"]
    A_eq = np.zeros((nS, nS * nA))
    for sp in range(nS):
        for a in range(nA):
            A_eq[sp, sp * nA + a] += 1.0
        for s in range(nS):
            for a in range(nA):
                A_eq[sp, s * nA + a] -= GAMMA * m["P"][s, a, sp]
    b_eq = (1.0 - GAMMA) * m["mu0"]
    return A_eq, b_eq


def _flat(mat: np.ndarray) -> np.ndarray:
    return mat.reshape(-1)


def solve_constrained(m: dict, bA=np.inf, bB=np.inf) -> dict:
    """max reward s.t. J_A<=bA, J_B<=bB over the occupancy polytope."""
    nS, nA = m["nS"], m["nA"]
    A_eq, b_eq = _flow_matrices(m)
    r = _flat(m["r"])
    A_ub, b_ub = [], []
    if np.isfinite(bA):
        A_ub.append(_flat(m["cA"]))
        b_ub.append(bA)
    if np.isfinite(bB):
        A_ub.append(_flat(m["cB"]))
        b_ub.append(bB)
    res = linprog(-r, A_ub=(np.array(A_ub) if A_ub else None),
                  b_ub=(np.array(b_ub) if b_ub else None),
                  A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if res.status != 0:
        return dict(feasible=False, status=res.message)
    x = res.x
    return dict(feasible=True, ret=float(r @ x),
                JA=float(_flat(m["cA"]) @ x), JB=float(_flat(m["cB"]) @ x))


def _sup_gap(m: dict, minus: np.ndarray) -> float:
    """max over occupancy polytope of x . minus   (linear objective)."""
    A_eq, b_eq = _flow_matrices(m)
    res = linprog(-minus, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return float(-res.fun) if res.status == 0 else float("nan")


def nu_pi_exact(m: dict) -> dict:
    """nu_Pi(U={A,B}) = min_psi sup_pi [max(J_A,J_B)-J_psi], exact via LPs.
    For psi=A this is max(0, sup_pi(J_B-J_A)); for psi=B, max(0, sup_pi(J_A-J_B))."""
    cA, cB = _flat(m["cA"]), _flat(m["cB"])
    gA = max(0.0, _sup_gap(m, cB - cA))    # A's worst policy-dependent shortfall
    gB = max(0.0, _sup_gap(m, cA - cB))    # B's worst policy-dependent shortfall
    return dict(nu_pi=float(min(gA, gB)), gap_singleton_A=gA, gap_singleton_B=gB)


def run(budgets: Tuple[float, ...] = (0.005, 0.010, 0.015, 0.020, 0.030)) -> dict:
    m = build_mdp()
    unc = solve_constrained(m)
    nu = nu_pi_exact(m)
    frontier = []
    for b in budgets:
        robust = solve_constrained(m, bA=b, bB=b)
        onlyA = solve_constrained(m, bA=b)          # bB free -> may violate B
        onlyB = solve_constrained(m, bB=b)          # bA free -> may violate A
        frontier.append(dict(
            budget=b,
            robust_ret=robust.get("ret"), robust_JA=robust.get("JA"), robust_JB=robust.get("JB"),
            onlyA_ret=onlyA.get("ret"), onlyA_JB=onlyA.get("JB"),   # B-cost incurred by A-policy
            onlyB_ret=onlyB.get("ret"), onlyB_JA=onlyB.get("JA"),   # A-cost incurred by B-policy
            onlyA_violates_B=(onlyA.get("JB", 0.0) > b + 1e-9),
            onlyB_violates_A=(onlyB.get("JA", 0.0) > b + 1e-9),
        ))
    # a singleton beats robust return only by violating the other constraint:
    robust_dominated_by_singleton = any(
        (f["onlyA_ret"] > f["robust_ret"] + 1e-9 and not f["onlyA_violates_B"]) or
        (f["onlyB_ret"] > f["robust_ret"] + 1e-9 and not f["onlyB_violates_A"])
        for f in frontier if f["robust_ret"] is not None)
    return dict(
        gamma=GAMMA, n_states=m["nS"],
        unconstrained=dict(ret=unc["ret"], JA=unc["JA"], JB=unc["JB"]),
        nu_pi_exact=nu["nu_pi"], gap_A=nu["gap_singleton_A"], gap_B=nu["gap_singleton_B"],
        genuinely_non_dominated=bool(nu["nu_pi"] > 1e-6),
        frontier=frontier,
        no_singleton_matches_robust_frontier=(not robust_dominated_by_singleton),
        readings=dict(A="CPU>90% for 5m (spike)", B="CPU>80% for 15m (sustained)"),
    )


def main() -> None:
    import json
    rep = run()
    print(f"nu_Pi(exact tabular occupancy LP) = {rep['nu_pi_exact']:.4f}  "
          f"(gap_A={rep['gap_A']:.4f}, gap_B={rep['gap_B']:.4f})")
    print(f"genuinely non-dominated: {rep['genuinely_non_dominated']}")
    print(f"no singleton matches robust frontier: {rep['no_singleton_matches_robust_frontier']}")
    for f in rep["frontier"]:
        print(f"  b={f['budget']:.2f}  robust_ret={f['robust_ret']:.3f}  "
              f"onlyA_ret={f['onlyA_ret']:.3f}(JB={f['onlyA_JB']:.3f},viol_B={f['onlyA_violates_B']})  "
              f"onlyB_ret={f['onlyB_ret']:.3f}(JA={f['onlyB_JA']:.3f},viol_A={f['onlyB_violates_A']})")


if __name__ == "__main__":
    main()
