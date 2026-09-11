"""Exact optimal values and certified prices via occupancy-measure LPs.

Blueprint items W12 / W14 / W22 (corset_iclr_8plus_revision_blueprint.md
sections 14, 16, 24, 30.5, 30.9): the paper's price-of-ambiguity table
(results/conformal/price/price_report.json) reports LEARNED-policy returns, so
its measured gaps mix the true (information-theoretic) price with singleton and
robust optimizer error.  This module certifies the true quantities by solving
each tabular domain's exact finite decision process:

  * unconstrained optimum          V*            (exact DP, gamma=1 finite horizon)
  * singleton optimum per reading  V_psi         (occupancy LP, budget d)
  * pairwise robust optimum        V_{psi_i,psi_j}
  * full-set robust optimum        V_U
  * certified true price           max_psi V_psi - V_U
  * pair-witness check (30.9): does some pair realize the full-set price?

Decision-process construction, per domain (all CPU, all exact):

  synthetic maintenance  state = (true RUL, firing pattern w).  Dynamics and
      reward depend only on (RUL, action); the retained readings fire on the
      NOISY observed features (rul_hat, anom), so the firing pattern w -- which
      subset of U fires this step -- is drawn i.i.d. given RUL with analytic
      Gaussian-CDF probabilities (verified by Monte Carlo through the env's own
      observe()).  A Markov policy on (t, RUL, w) dominates every
      observation-history policy the paper evaluates (standard occupancy
      sufficiency: (RUL_t, w_t) is Markov given actions, and reward, cost and
      transitions depend only on (RUL_t, w_t, a_t)), so each LP optimum is a
      certified upper bound for the paper's policy class and the exact optimum
      of the full-information class.

  gridworld  state = offset-since-warning belief MDP.  The offset o (steps
      since the most recent warning edge) is a complete summary of the
      observable history; danger probability and next-onset hazard are exact
      posteriors over the latent duration D ~ U{dur_min..dur_max} given o.
      Offsets >= dur_max + refractory - 1 are exactly lumped into one CLEAR
      state (identical hazard, zero danger, no reading fires there).

  budget  state = running bill / unit_cost.  Fully deterministic.

  C-MAPSS replay (FD001)  state = (engine, position) over the real recorded
      trajectories; features (hence firing) are deterministic per state.  The
      reset action's uniform-fresh-engine transition is routed through one hub
      variable per stage to keep the LP sparse.  Every FD001 engine is longer
      than the 120-step horizon, so run-to-failure is unreachable from pos 0
      (asserted), and the unconstrained optimum is exactly always-continue.

Constraint convention: the paper's learners enforce the ACTIVE-normalized
expected semantic cost J = E[sum_t c_psi] / E[sum_t 1{active}] <= eps (saorl/
offline.py, normalize="active"; denominator = steps where some HONORED reading
fires).  The exact programs use the on-policy analogue, which is linear in the
occupancy measure:  sum mu(s,a) * (c_psi(s,a) - d * active(s)) <= 0.  Both the
raw expected episodic cost and the normalized ratio are reported per solution
so numbers are comparable with the paper's normalization.  d = 0.05 (the
paper's eps) is primary; d = 0 (hard compliance) is also reported.

Normalization modes (plan section 9.2, "fixed per-reading normalization"):

  normalization="honored_active" (DEFAULT, the paper's numbers)
      denominator = E[sum_t 1{some HONORED reading fires}]; this denominator
      is SET-DEPENDENT (a singleton program divides by the active-steps over
      {psi}, the full-set program by the active-steps over U), so at d > 0 a
      pairwise program is not nested inside the full-set program and a pair
      price can nominally exceed the full-set price.  Reproduces lp_report.json
      byte-for-byte and remains the source of the paper's certified numbers.

  normalization="fixed_psi"  (set-independent re-run of the d = 0.05 rows)
      each reading psi is normalized by a FIXED per-reading constant Z_psi that
      is computed once, before any retained set is constructed, and used
      identically in the singleton, pair and full-set programs.  The constraint
      becomes  E_pi[sum_t c_psi(s_t,a_t)] <= d * Z_psi.  Z_psi is the expected
      number of steps the RETURN-GREEDY reference policy (the always-greedy
      action per domain, GREEDY_ACTION[domain]) spends in states where psi
      fires:  Z_psi = sum_s occ_greedy(s) * fire_psi(s), occ_greedy the
      horizon-summed state visitation of the greedy action from p0.  This
      reference is independent of every reading and of the honored set (it is
      the reading-agnostic run-to-failure behavior), so each reading's
      constraint is identical across all programs; the full-set feasible region
      is therefore nested inside every pair's and the witness anomaly cannot
      occur at any d.  At d = 0 both modes reduce to E[sum_t c_psi] <= 0 and
      coincide exactly.

Run (from the repo root):

  SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.exact_lp
  SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.exact_lp --normalization fixed_psi

The default writes results/conformal/lp/lp_report.json and
paper/generated/gen_lp.tex; --normalization fixed_psi writes
results/conformal/lp/lp_report_fixedZ.json and paper/generated/gen_lp_fixedZ.tex.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.stats import norm

from .dsl import Atom, Candidate, Within

EPS_BUDGET = 0.05          # the paper's eps (saorl.experiments.EPS)
BUDGETS = (0.05, 0.0)      # primary (paper eps) and hard-compliance
PATTERN_PRUNE = 1e-14      # drop firing patterns with probability below this
PAIR_TOL = 1e-6            # witness-pair tolerance (absolute, return units)
IPM_THRESHOLD = 1_000_000  # above this many LP variables use HiGHS IPM
                           # (interior point + crossover) instead of simplex

_HERE = Path(__file__).parent
_ROOT = _HERE.parent.parent
_PRICE_REPORT = _ROOT / "results/conformal" / "price" / "price_report.json"
_OUT_DIR = _ROOT / "results/conformal" / "lp"
_TEX_OUT = _ROOT / "paper" / "generated" / "gen_lp.tex"


# ---------------------------------------------------------------------------
# Generic finite CMDP (stage-invariant dynamics, finite horizon, gamma = 1)
# ---------------------------------------------------------------------------

@dataclass
class TabularCMDP:
    name: str
    horizon: int
    states: List                    # hashable state labels (index = position)
    action_names: Tuple[str, ...]
    T: List[sparse.csr_matrix]      # per action: S x S row-stochastic kernel
    r: np.ndarray                   # S x A expected one-step reward
    cost: Dict[str, np.ndarray]     # reading name -> S x A per-step cost
    fire: Dict[str, np.ndarray]     # reading name -> S {0,1} fires-at-state
    p0: np.ndarray                  # initial state distribution
    hub_actions: Tuple[int, ...] = ()   # actions whose T-row equals hub_dist
    hub_dist: Optional[np.ndarray] = None  # for every state (uniform reset)
    notes: str = ""

    @property
    def S(self) -> int:
        return len(self.states)

    @property
    def A(self) -> int:
        return len(self.action_names)

    def active(self, group: Sequence[str]) -> np.ndarray:
        """1{some reading in `group` fires at s} (exact: fire is 0/1 here)."""
        act = np.zeros(self.S)
        for k in group:
            act = np.maximum(act, self.fire[k])
        return act


def dp_optimal(m: TabularCMDP) -> float:
    """Exact unconstrained finite-horizon DP value from p0."""
    V = np.zeros(m.S)
    for _ in range(m.horizon):
        Q = np.stack([m.r[:, a] + m.T[a] @ V for a in range(m.A)], axis=1)
        V = Q.max(axis=1)
    return float(m.p0 @ V)


def dp_policy_value(m: TabularCMDP, act_of_state: np.ndarray) -> float:
    """Exact value of a stationary deterministic policy (S-vector of actions)."""
    V = np.zeros(m.S)
    for _ in range(m.horizon):
        Vn = np.zeros(m.S)
        for a in range(m.A):
            sel = act_of_state == a
            if sel.any():
                Vn[sel] = m.r[sel, a] + (m.T[a] @ V)[sel]
        V = Vn
    return float(m.p0 @ V)


def guard_policy_actions(m: TabularCMDP, U_names: Sequence[str],
                         a_greedy: int, a_safe: int) -> np.ndarray:
    """The training-free union-guard fallback: safe action wherever some
    retained reading fires, greedy action elsewhere (zero cost by construction)."""
    act = m.active(U_names)
    return np.where(act > 0, a_safe, a_greedy).astype(np.int64)


def greedy_occupancy(m: TabularCMDP, a_greedy: int) -> np.ndarray:
    """Horizon-summed state visitation E[sum_t 1{s_t = s}] under the
    RETURN-GREEDY reference policy that plays action `a_greedy` in every state,
    started from p0 and pushed through that action's exact transition kernel.

    This occupancy is a fixed reference: it depends on neither any reading nor
    any honored set (it is the reading-agnostic always-greedy / run-to-failure
    behavior), which is what makes the per-reading normalizer Z_psi built from
    it identical across the singleton, pair and full-set programs."""
    occ = np.zeros(m.S)
    mu = m.p0.copy()
    TaT = m.T[a_greedy].T.tocsr()
    for _ in range(m.horizon):
        occ += mu
        mu = TaT @ mu
    return occ


def fixed_psi_Z(m: TabularCMDP, readings: Sequence[str],
                a_greedy: int) -> Dict[str, float]:
    """Fixed per-reading normalizer Z_psi = E_greedy[sum_t 1{psi fires at s_t}],
    the expected number of steps the return-greedy reference policy spends in
    states where psi fires.  Computed once, set-independently (plan 9.2)."""
    occ = greedy_occupancy(m, a_greedy)
    return {k: float(occ @ m.fire[k]) for k in readings}


# ---------------------------------------------------------------------------
# Occupancy-measure LP
# ---------------------------------------------------------------------------

@dataclass
class LPSolution:
    value: float
    x: np.ndarray                 # occupancy, shape (H, S, A)
    diagnostics: dict
    on_policy: dict               # per-reading E[cost], E[active], ratio


def solve_occupancy_lp(
    m: TabularCMDP,
    honored: Sequence[str],
    d: float,
    report_readings: Optional[Sequence[str]] = None,
    normalization: str = "honored_active",
    Z: Optional[Dict[str, float]] = None,
) -> LPSolution:
    """max E[sum r] s.t. flow conservation and, for each psi in `honored`,
    a per-reading expected-cost budget.

    normalization="honored_active" (default): the set-dependent on-policy
        analogue of the learners' normalize="active" convention,
        E[sum c_psi] - d * E[sum 1{active_honored}] <= 0, where
        `active_honored` = some member of `honored` fires.

    normalization="fixed_psi": the set-INDEPENDENT budget
        E[sum c_psi] <= d * Z_psi with Z_psi supplied in `Z` (a fixed
        per-reading constant, identical across the singleton/pair/full-set
        programs).  At d = 0 both modes reduce to E[sum c_psi] <= 0 and give
        identical programs.
    """
    H, S, A = m.horizon, m.S, m.A
    nv = H * S * A
    hub = bool(m.hub_actions)
    n_hub = (H - 1) if hub else 0        # hub var h_t for stages t = 1..H-1
    ntot = nv + n_hub

    rows, cols, vals = [], [], []

    # own-node entries: +1 at row (t, s) for every x[t,s,a]
    t_idx = np.repeat(np.arange(H), S * A)
    s_idx = np.tile(np.repeat(np.arange(S), A), H)
    rows.append(t_idx * S + s_idx)
    cols.append(np.arange(nv))
    vals.append(np.ones(nv))

    # successor entries: -P(s'|s,a) at row (t+1, s') for x[t,s,a], t < H-1
    for a in range(m.A):
        if hub and a in m.hub_actions:
            continue
        Ta = m.T[a].tocoo()
        src, dst, p = Ta.row, Ta.col, Ta.data
        for t in range(H - 1):
            rows.append((t + 1) * S + dst)
            cols.append(t * S * A + src * A + a)
            vals.append(-p)

    hub_row0 = H * S
    if hub:
        # hub equality rows: h_{t+1} - sum_{s, a in hub_actions} x[t,s,a] = 0;
        # node rows at t+1 receive -hub_dist[s] * h_{t+1}
        nz = np.nonzero(m.hub_dist)[0]
        for t in range(H - 1):
            hv = nv + t                    # hub var for stage t+1
            hr = hub_row0 + t              # its equality row
            rows.append(np.array([hr]))
            cols.append(np.array([hv]))
            vals.append(np.array([1.0]))
            for a in m.hub_actions:
                rows.append(np.full(S, hr))
                cols.append(t * S * A + np.arange(S) * A + a)
                vals.append(-np.ones(S))
            rows.append((t + 1) * S + nz)
            cols.append(np.full(len(nz), hv))
            vals.append(-m.hub_dist[nz])

    n_eq = H * S + n_hub
    A_eq = sparse.coo_matrix(
        (np.concatenate(vals),
         (np.concatenate(rows).astype(np.int64),
          np.concatenate(cols).astype(np.int64))),
        shape=(n_eq, ntot),
    ).tocsr()
    b_eq = np.zeros(n_eq)
    b_eq[:S] = m.p0

    # cost constraints (sparse: only firing/active states carry coefficients)
    A_ub = None
    b_ub = None
    act = m.active(honored) if honored else np.zeros(S)
    fixed = (normalization == "fixed_psi")
    if honored:
        if fixed and Z is None:
            raise ValueError("normalization='fixed_psi' requires Z")
        c_rows, c_cols, c_vals = [], [], []
        for ci, k in enumerate(honored):
            # honored_active: coefficients c_psi - d*active carry the budget on
            #     the LHS (RHS 0).  fixed_psi: only c_psi on the LHS; the budget
            #     d*Z_psi moves to a constant RHS b_ub.
            w = m.cost[k] if fixed else (m.cost[k] - d * act[:, None])   # S x A
            snz, anz = np.nonzero(w)
            wv = w[snz, anz]
            for t in range(H):
                c_rows.append(np.full(len(snz), ci))
                c_cols.append(t * S * A + snz * A + anz)
                c_vals.append(wv)
        A_ub = sparse.coo_matrix(
            (np.concatenate(c_vals),
             (np.concatenate(c_rows), np.concatenate(c_cols))),
            shape=(len(honored), ntot)).tocsr()
        b_ub = (np.array([d * float(Z[k]) for k in honored]) if fixed
                else np.zeros(len(honored)))

    # objective: minimize -reward
    c = np.concatenate([np.tile(-m.r.reshape(-1), H), np.zeros(n_hub)])

    # provably-unreachable (t, s) pairs are fixed to zero so presolve strips
    # them (exact: reachability propagated through the transition kernels)
    reach = m.p0 > 0
    ub = np.full(ntot, np.inf)
    free = np.ones(ntot, dtype=bool)
    for t in range(H):
        dead = np.nonzero(~reach)[0]
        if len(dead):
            idx = (t * S * A
                   + (dead[:, None] * A + np.arange(A)[None, :]).reshape(-1))
            ub[idx] = 0.0
            free[idx] = False
        nxt = np.zeros(S, dtype=bool)
        for a in range(m.A):
            nxt |= (m.T[a].T @ reach.astype(float)) > 0
        reach = nxt
    bounds = np.stack([np.zeros(ntot), ub], axis=1)

    method = "highs-ipm" if ntot > IPM_THRESHOLD else "highs"
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method=method)
    if res.status != 0:
        raise RuntimeError(f"LP failed ({m.name}, honored={honored}, d={d}): "
                           f"status={res.status} {res.message}")

    x = np.maximum(res.x[:nv], 0.0).reshape(H, S, A)

    # --- optimality evidence -------------------------------------------------
    primal_res = float(np.abs(A_eq @ res.x - b_eq).max())
    y = res.eqlin.marginals
    red = c - A_eq.T @ y
    if honored:
        lam = res.ineqlin.marginals
        red = red - A_ub.T @ lam
        # dual objective of min c'x s.t. A_eq x = b_eq, A_ub x <= b_ub, x >= 0;
        # b_ub is the zero vector in honored_active, so this reduces to b_eq @ y
        dual_obj = float(b_eq @ y + b_ub @ lam)
    else:
        dual_obj = float(b_eq @ y)
    gap = abs(res.fun - dual_obj) / (1.0 + abs(res.fun))
    min_red = float(red[free].min())     # fixed-at-zero vars carry no dual info
    mass_err = float(np.abs(x.sum(axis=(1, 2)) - 1.0).max())
    # constraint VIOLATION (slack): A_ub x - b_ub; b_ub = 0 in honored_active
    cons_viol = float((A_ub @ res.x - b_ub).max()) if honored else 0.0

    diagnostics = dict(
        status=int(res.status), message=str(res.message), method=method,
        n_variables=int(ntot), n_eq_rows=int(n_eq), n_ub_rows=len(honored or ()),
        primal_objective=float(-res.fun), dual_objective=float(-dual_obj),
        relative_duality_gap=float(gap),
        max_primal_residual=primal_res,
        min_reduced_cost=min_red,
        max_stage_mass_error=mass_err,
        max_constraint_value=cons_viol,
    )
    if honored:
        # per-reading shadow prices of the max-form program: lambda_psi =
        # dV/d(b_psi) >= 0 (scipy marginals are <=0 for the min form)
        diagnostics["cost_duals"] = {
            k: float(max(-lam[i], 0.0)) for i, k in enumerate(honored)}

    # --- on-policy quantities (paper's normalization, for comparability) -----
    readings = list(report_readings) if report_readings else list(honored)
    occ_state = x.sum(axis=(0, 2))                # E[# visits to s]
    E_active = float(occ_state @ act)
    onp = dict(E_active_steps=E_active)
    for k in readings:
        Ec = float((x * m.cost[k][None, :, :]).sum())
        onp[k] = dict(
            E_episodic_cost=Ec,
            active_normalized_cost=(Ec / E_active if E_active > 0 else 0.0),
        )
    return LPSolution(float(-res.fun), x, diagnostics, onp)


# ---------------------------------------------------------------------------
# Dantzig-Wolfe column generation for the SAME occupancy LP (large domains)
#
# The occupancy polytope is the convex hull of the occupancies of
# deterministic nonstationary policies, so the LP
#     max r.mu  s.t. flow(mu), q_k.mu <= 0
# equals the master  max sum_p theta_p R_p  s.t.  sum_p theta_p Q_pk <= 0,
# sum_p theta_p = 1, theta >= 0  over policy columns p, priced by the exact
# DP  max_pi [R(pi) - sum_k lambda_k Q_k(pi)].  The master (tiny, HiGHS) gives
# a primal-feasible explicit policy mixture; the pricing value g(lambda) is a
# valid Lagrangian dual bound; termination at gap <= tol certifies optimality.
# ---------------------------------------------------------------------------

def _dp_shaped(m: TabularCMDP, shaped_r: np.ndarray):
    """Backward induction on a shaped reward; returns (value, plan[t][s])."""
    V = np.zeros(m.S)
    plans = np.zeros((m.horizon, m.S), dtype=np.int8)
    for t in range(m.horizon - 1, -1, -1):
        Q = np.stack([shaped_r[:, a] + m.T[a] @ V for a in range(m.A)], axis=1)
        plans[t] = Q.argmax(axis=1)
        V = Q.max(axis=1)
    return float(m.p0 @ V), plans


def _evaluate_plan(m: TabularCMDP, plans: np.ndarray,
                   payoffs: List[np.ndarray]) -> List[float]:
    """Exact expected episodic totals of several S x A payoff matrices under a
    nonstationary deterministic plan (plans[t][s] = action)."""
    J = len(payoffs)
    V = np.zeros((m.S, J))
    for t in range(m.horizon - 1, -1, -1):
        acts = plans[t]
        Vn = np.empty_like(V)
        for a in range(m.A):
            sel = acts == a
            if sel.any():
                cont = m.T[a] @ V
                Vn[sel] = cont[sel] + np.stack(
                    [pay[sel, a] for pay in payoffs], axis=1)
        V = Vn
    return [float(m.p0 @ V[:, j]) for j in range(J)]


def solve_occupancy_cg(
    m: TabularCMDP,
    honored: Sequence[str],
    d: float,
    report_readings: Optional[Sequence[str]] = None,
    tol: float = 1e-9,
    max_iter: int = 400,
    normalization: str = "honored_active",
    Z: Optional[Dict[str, float]] = None,
) -> LPSolution:
    """Column-generation solve of the identical occupancy LP; returns an
    LPSolution whose x is None and whose diagnostics carry the certified
    primal/dual bound pair.  Extra fields: .columns (plans) and .theta.

    `normalization` / `Z` have the same meaning as in solve_occupancy_lp: the
    per-reading constraint sum_p theta_p Q_pk <= b_k has RHS b_k = 0 in
    honored_active (the -d*active term is folded into Q) and b_k = d*Z_psi in
    fixed_psi (Q carries only the raw cost)."""
    readings = list(report_readings) if report_readings else list(honored)
    act = m.active(honored) if honored else np.zeros(m.S)
    K = len(honored)
    fixed = (normalization == "fixed_psi")
    if honored and fixed and Z is None:
        raise ValueError("normalization='fixed_psi' requires Z")
    q_mats = [(m.cost[k] if fixed else m.cost[k] - d * act[:, None])
              for k in honored]
    b_cg = (np.array([d * float(Z[k]) for k in honored]) if fixed
            else np.zeros(K))
    cost_mats = [m.cost[k] for k in readings]
    act_mat = np.tile(act[:, None], (1, m.A))

    columns: List[dict] = []

    def add_column(plans: np.ndarray):
        vals = _evaluate_plan(m, plans, [m.r] + q_mats + cost_mats + [act_mat])
        columns.append(dict(
            plans=plans, R=vals[0], Q=np.array(vals[1:1 + K]),
            raw_costs=np.array(vals[1 + K:1 + K + len(cost_mats)]),
            active=vals[-1]))

    # seed: unconstrained optimum and (if needed) the zero-cost guard policy
    _v, plans0 = _dp_shaped(m, m.r)
    add_column(plans0)
    if K and columns[0]["Q"].max() > 0:
        forb = np.zeros(m.S, dtype=bool)
        for k in honored:
            forb |= m.cost[k][:, 0] > 0  # any state where the greedy action
        # guard: replace the greedy action by the first zero-cost action
        gact = np.zeros(m.S, dtype=np.int8)
        for s in np.nonzero(forb)[0]:
            for a in range(m.A):
                if all(m.cost[k][s, a] == 0 for k in honored):
                    gact[s] = a
                    break
        add_column(np.tile(gact, (m.horizon, 1)))

    lam = np.zeros(K)
    it = 0
    master_diag: dict = {}
    theta = np.array([1.0])
    dual_bound = math.inf
    primal = -math.inf
    while it < max_iter:
        it += 1
        # master over current columns
        if K:
            A_ub = np.stack([c["Q"] for c in columns], axis=1)   # K x P
            res = linprog(
                np.array([-c["R"] for c in columns]),
                A_ub=A_ub, b_ub=b_cg,
                A_eq=np.ones((1, len(columns))), b_eq=np.array([1.0]),
                bounds=(0, None), method="highs")
            if res.status != 0:
                raise RuntimeError(f"CG master failed: {res.message}")
            theta = np.maximum(res.x, 0.0)
            primal = float(-res.fun)
            lam = np.maximum(-res.ineqlin.marginals, 0.0)
        else:
            theta = np.zeros(len(columns))
            theta[int(np.argmax([c["R"] for c in columns]))] = 1.0
            primal = float(max(c["R"] for c in columns))
            lam = np.zeros(0)
        # pricing: exact DP on the Lagrangian-shaped reward.  The Lagrangian
        # dual value for the full problem is g(lam) = max_pi [R - lam.Q] +
        # lam.b (b = 0 in honored_active); it upper-bounds the LP optimum, and
        # the master optimum equals sigma* = primal - lam.b, so the reduced-
        # cost optimality test is g + lam.b <= primal.
        shaped = m.r.copy()
        for k in range(K):
            shaped = shaped - lam[k] * q_mats[k]
        g, plans = _dp_shaped(m, shaped)
        g_dual = (g + float(lam @ b_cg)) if K else g
        dual_bound = min(dual_bound, g_dual)
        if g_dual <= primal + tol * (1.0 + abs(primal)) or K == 0:
            break
        add_column(plans)

    gap = (dual_bound - primal) / (1.0 + abs(primal))
    if K:
        qtheta = (np.stack([c["Q"] for c in columns], axis=1)
                  * theta).sum(axis=1)
        max_q = float(max((qtheta - b_cg).max(), 0.0))
    else:
        max_q = 0.0
    diagnostics = dict(
        status=0,
        message="column generation converged (Dantzig-Wolfe over "
                "deterministic DP policies; master: HiGHS)",
        method="dantzig-wolfe column generation",
        n_columns=len(columns), n_iterations=it,
        primal_objective=primal, dual_objective=float(dual_bound),
        relative_duality_gap=float(max(gap, 0.0)),
        max_primal_residual=float(abs(theta.sum() - 1.0)),
        min_reduced_cost=float(primal - dual_bound),
        max_stage_mass_error=float(abs(theta.sum() - 1.0)),
        max_constraint_value=max_q,
        theta_support=int((theta > 1e-12).sum()),
    )
    if K:
        # master duals at termination; pricing certifies lam is (near-)optimal
        # for the full dual up to the reported relative_duality_gap
        diagnostics["cost_duals"] = {
            k: float(lam[i]) for i, k in enumerate(honored)}
    E_active = float(sum(t * c["active"] for t, c in zip(theta, columns)))
    onp = dict(E_active_steps=E_active)
    for j, k in enumerate(readings):
        Ec = float(sum(t * c["raw_costs"][j] for t, c in zip(theta, columns)))
        onp[k] = dict(E_episodic_cost=Ec,
                      active_normalized_cost=(Ec / E_active
                                              if E_active > 0 else 0.0))
    sol = LPSolution(primal, None, diagnostics, onp)
    sol.columns = [c["plans"] for c in columns]      # type: ignore[attr-defined]
    sol.theta = theta                                # type: ignore[attr-defined]
    _ = master_diag
    return sol


def lp_policy(m: TabularCMDP, sol: LPSolution, default: np.ndarray) -> np.ndarray:
    """Extract pi_t(a|s) from the occupancy (H, S, A); default action where the
    state carries no mass (unreachable / off-support)."""
    H, S, A = sol.x.shape
    pol = np.zeros((H, S, A))
    mass = sol.x.sum(axis=2)
    for t in range(H):
        has = mass[t] > 1e-12
        pol[t, has] = sol.x[t, has] / mass[t, has, None]
        pol[t, ~has, :] = 0.0
        pol[t, ~has, default[~has]] = 1.0
    return pol


# ---------------------------------------------------------------------------
# Domain builders
# ---------------------------------------------------------------------------

def _atom(c: Candidate) -> Atom:
    if not isinstance(c.predicate, Atom):
        raise ValueError(f"non-atomic retained reading {c.name}: exact analytic "
                         f"firing probabilities are implemented for Atoms only")
    return c.predicate


def _pattern_table(U: Sequence[Candidate], env):
    """Joint firing-pattern distribution for the maintenance readings.

    Readings on the SAME feature share one noise draw (e.g. rul_hat<=22 and
    rul_hat<=40 are nested events of one Gaussian); different features are
    independent given RUL (independent noise draws in env.observe).  Returns
    (patterns, probs) where patterns is the (growing) list of fire-bit tuples
    in U order and probs(rul) -> vector aligned with the current pattern list.
    """
    groups: Dict[str, List[Tuple[int, Atom]]] = {}
    for i, c in enumerate(U):
        a = _atom(c)
        groups.setdefault(a.feature, []).append((i, a))
    feats = sorted(groups)

    def marginal(feature: str, atom: Atom, rul: float) -> float:
        if feature == "rul_hat":
            # rul_hat = max(0, rul + N(0, obs_noise)); for theta >= 0 the
            # events {rul_hat <= theta} and {rul + N <= theta} coincide
            assert atom.op == "le" and atom.theta >= 0
            return float(norm.cdf((atom.theta - rul) / env.obs_noise))
        if feature == "anom":
            # anom = clip(1 - rul/max_life + N(0, anom_noise), 0, 1); for
            # theta in (0,1) the clip does not change {anom >= theta}
            assert atom.op == "ge" and 0 < atom.theta < 1
            return float(norm.cdf(((1.0 - rul / env.max_life) - atom.theta)
                                  / env.anom_noise))
        raise ValueError(f"unhandled feature {feature}")

    def group_cells(feature: str, members, rul: float):
        # events on one feature are nested; sort widest first.  Cell j =
        # exactly the j widest events fire, prob p_j - p_{j+1} (p_0 = 1).
        ms = sorted(members, key=lambda ia: -marginal(feature, ia[1], rul))
        p = [1.0] + [marginal(feature, a, rul) for _, a in ms] + [0.0]
        cells = []
        for j in range(len(ms) + 1):
            bits = {idx: (1 if rank < j else 0)
                    for rank, (idx, _) in enumerate(ms)}
            cells.append((bits, max(0.0, p[j] - p[j + 1])))
        return cells

    patterns: List[Tuple[int, ...]] = []
    seen: Dict[Tuple[int, ...], int] = {}

    def probs(rul: float) -> np.ndarray:
        cells = [group_cells(f, groups[f], rul) for f in feats]
        out: Dict[Tuple[int, ...], float] = {}

        def rec(fi: int, bits: dict, p: float):
            if p <= 0:
                return
            if fi == len(cells):
                key = tuple(bits[i] for i in range(len(U)))
                out[key] = out.get(key, 0.0) + p
                return
            for cb, cp in cells[fi]:
                b2 = dict(bits)
                b2.update(cb)
                rec(fi + 1, b2, p * cp)

        rec(0, {}, 1.0)
        for key in out:
            if key not in seen:
                seen[key] = len(patterns)
                patterns.append(key)
        vec = np.zeros(len(patterns))
        for key, p in out.items():
            vec[seen[key]] = p
        return vec

    return patterns, probs


def build_synthetic(env, U: Sequence[Candidate]) -> TabularCMDP:
    """(true RUL, firing pattern) maintenance CMDP under the TENSION economics."""
    H = env.horizon
    ruls = np.arange(0, int(env.max_life) + 1)
    patterns, probs_fn = _pattern_table(U, env)
    raw = [probs_fn(float(r)) for r in ruls]
    P = np.zeros((len(ruls), len(patterns)))
    for i, p in enumerate(raw):
        P[i, : len(p)] = p

    states: List[Tuple[int, int]] = []
    sidx: Dict[Tuple[int, int], int] = {}
    trunc = 0.0
    for i, r in enumerate(ruls):
        kept = P[i] > PATTERN_PRUNE
        trunc = max(trunc, float(P[i][~kept].sum()))
        for j in np.nonzero(kept)[0]:
            sidx[(int(r), int(j))] = len(states)
            states.append((int(r), int(j)))
    S = len(states)

    def w_dist(r: int) -> List[Tuple[int, float]]:
        i = int(r)
        js = [j for j in range(len(patterns)) if (i, j) in sidx]
        tot = float(sum(P[i, j] for j in js))
        return [(sidx[(i, j)], float(P[i, j]) / tot) for j in js]

    A = 3
    action_names = ("continue", "minor_repair", "replace")
    r_mat = np.zeros((S, A))
    T = [sparse.lil_matrix((S, S)) for _ in range(A)]
    top = int(env.max_life)
    for (rul, j), s in sidx.items():
        if rul > 0:
            r_mat[s, 0] = env.r_op
            nxt = rul - 1
        else:
            r_mat[s, 0] = -env.c_fail
            nxt = top
        for s2, p in w_dist(nxt):
            T[0][s, s2] = p
        r_mat[s, 1] = -env.c_minor
        for s2, p in w_dist(int(min(top, rul + env.repair_gain))):
            T[1][s, s2] = p
        r_mat[s, 2] = -env.c_replace
        for s2, p in w_dist(top):
            T[2][s, s2] = p
    T = [t.tocsr() for t in T]

    cost, fire = {}, {}
    for i, c in enumerate(U):
        f = np.array([patterns[j][i] for (_, j) in states], dtype=float)
        fire[c.name] = f
        cm = np.zeros((S, A))
        cm[:, 0] = f                       # forbidden action = continue
        cost[c.name] = cm

    p0 = np.zeros(S)
    lo, hi = int(env.start_low), int(env.start_high)
    for r0 in range(lo, hi + 1):
        for s2, p in w_dist(r0):
            p0[s2] += p / (hi - lo + 1)

    return TabularCMDP(
        name="synthetic", horizon=H, states=states, action_names=action_names,
        T=T, r=r_mat, cost=cost, fire=fire, p0=p0,
        notes=(f"state=(RUL,firing pattern); {len(patterns)} patterns, "
               f"{S} states; pattern probs analytic (Gaussian CDF); patterns "
               f"below {PATTERN_PRUNE:g} pruned (max truncated mass "
               f"{trunc:.2e}, renormalized)"),
    )


def build_gridworld(env, U: Sequence[Candidate]) -> TabularCMDP:
    """Offset-since-warning belief MDP (exact posteriors over duration D)."""
    p_warn, R = env.p_warn, env.refractory
    durs = np.arange(env.dur_min, env.dur_max + 1)
    o_clear = env.dur_max + R - 1          # offsets >= o_clear behave identically
    for c in U:
        assert isinstance(c.predicate, Within), c.name
        assert c.predicate.W - 1 < o_clear
    n_o = o_clear                          # explicit offsets 0..o_clear-1
    CLEAR = n_o
    S = n_o + 1
    states: List = list(range(n_o)) + ["CLEAR"]

    def post(o: int) -> np.ndarray:
        # posterior over D given: onset o steps ago, no onset since.  Idle
        # offsets for duration d start at d+R; each survived idle step weighs
        # (1 - p_warn).
        w = np.array([(1.0 - p_warn) ** max(0, o - (d + R) + 1) for d in durs],
                     dtype=float)
        return w / w.sum()

    def p_unsafe(o: int) -> float:
        return float(post(o) @ (durs > o))

    def hazard(o: int) -> float:
        # P(new onset at offset o+1 | reached offset o)
        return float(p_warn * (post(o) @ (o + 1 >= durs + R)))

    A = 2
    action_names = ("advance", "wait")
    r_mat = np.zeros((S, A))
    for o in range(n_o):
        pu = p_unsafe(o)
        r_mat[o, 0] = env.r_advance * (1 - pu) - env.c_accident * pu
        r_mat[o, 1] = -env.c_wait
    r_mat[CLEAR, 0] = env.r_advance
    r_mat[CLEAR, 1] = -env.c_wait

    T = [sparse.lil_matrix((S, S)) for _ in range(A)]
    for o in range(n_o):
        h = hazard(o)
        nxt = o + 1 if o + 1 < n_o else CLEAR
        for a in range(A):
            T[a][o, 0] = h
            T[a][o, nxt] = 1 - h
    for a in range(A):
        T[a][CLEAR, 0] = p_warn
        T[a][CLEAR, CLEAR] = 1 - p_warn
    T = [t.tocsr() for t in T]

    cost, fire = {}, {}
    for c in U:
        W = c.predicate.W
        f = np.array([1.0 if (isinstance(s, int) and s <= W - 1) else 0.0
                      for s in states])
        fire[c.name] = f
        cm = np.zeros((S, A))
        cm[:, 0] = f                       # forbidden action = advance
        cost[c.name] = cm

    p0 = np.zeros(S)
    p0[0] = p_warn                          # an onset can occur at t = 0
    p0[CLEAR] = 1 - p_warn

    return TabularCMDP(
        name="gridworld", horizon=env.horizon, states=states,
        action_names=action_names, T=T, r=r_mat, cost=cost, fire=fire, p0=p0,
        notes=(f"belief MDP on offset-since-warning; offsets 0..{n_o - 1} + "
               f"CLEAR (exact lump: for offsets >= {o_clear} hazard = p_warn, "
               f"safe, no reading fires); posteriors over "
               f"D~U{{{env.dur_min}..{env.dur_max}}} exact"),
    )


def build_budget(env, U: Sequence[Candidate]) -> TabularCMDP:
    """Deterministic running-bill CMDP."""
    H = env.horizon
    S = H + 1                               # bill buckets 0..H (spend = 10*b)
    states = list(range(S))
    A = 2
    action_names = ("buy", "skip")
    r_mat = np.zeros((S, A))
    r_mat[:, 0] = env.r_buy
    r_mat[:, 1] = env.r_skip
    T = [sparse.lil_matrix((S, S)) for _ in range(A)]
    for b in range(S):
        T[0][b, min(b + 1, S - 1)] = 1.0
        T[1][b, b] = 1.0
    T = [t.tocsr() for t in T]
    cost, fire = {}, {}
    for c in U:
        a = _atom(c)
        assert a.feature == "spend" and a.op == "ge"
        f = np.array([1.0 if b * env.unit_cost >= a.theta else 0.0
                      for b in range(S)])
        fire[c.name] = f
        cm = np.zeros((S, A))
        cm[:, 0] = f                       # forbidden action = buy
        cost[c.name] = cm
    p0 = np.zeros(S)
    p0[0] = 1.0
    return TabularCMDP(
        name="budget", horizon=H, states=states, action_names=action_names,
        T=T, r=r_mat, cost=cost, fire=fire, p0=p0,
        notes="deterministic; bill bucket = spend/unit_cost",
    )


def build_real(env, U: Sequence[Candidate]) -> TabularCMDP:
    """(engine, position) replay CMDP over the real FD001 trajectories."""
    H = env.horizon
    lens = [len(tr) for tr in env.engines]
    assert min(lens) > H, (
        "an engine is shorter than the horizon: run-to-failure would be "
        "reachable and the failure branch must be modeled")
    n_eng = len(env.engines)
    pos_max = H                             # positions 0..H-1 (pos <= t < H)
    states = [(ei, p) for ei in range(n_eng) for p in range(pos_max)]
    sidx = {st: i for i, st in enumerate(states)}
    S = len(states)
    A = 3
    action_names = ("continue", "minor_repair", "replace")
    r_mat = np.zeros((S, A))
    r_mat[:, 0] = env.r_op
    r_mat[:, 1] = -env.c_minor
    r_mat[:, 2] = -env.c_replace

    p0 = np.zeros(S)
    for ei in range(n_eng):
        p0[sidx[(ei, 0)]] = 1.0 / n_eng

    rows = np.arange(S)
    # continue: pos+1 (the last position self-loops; it is unreachable before
    # the final stage because pos <= t, so the loop never carries mass)
    cont = np.array([sidx[(ei, min(p + 1, pos_max - 1))] for ei, p in states])
    # minor repair: `repair_gain` real cycles back up the curve
    rep = np.array([sidx[(ei, max(0, p - env.repair_gain))] for ei, p in states])
    # replace: uniform fresh engine at pos 0 (sparse outer product with p0)
    nz0 = np.nonzero(p0)[0]
    T = [
        sparse.csr_matrix((np.ones(S), (rows, cont)), shape=(S, S)),
        sparse.csr_matrix((np.ones(S), (rows, rep)), shape=(S, S)),
        sparse.csr_matrix(
            (np.tile(p0[nz0], S),
             (np.repeat(rows, len(nz0)), np.tile(nz0, S))), shape=(S, S)),
    ]

    cost, fire = {}, {}
    for c in U:
        f = np.array([1.0 if c.fires([env.engines[ei][p]], 0) else 0.0
                      for ei, p in states])
        fire[c.name] = f
        cm = np.zeros((S, A))
        cm[:, 0] = f                       # forbidden action = continue
        cost[c.name] = cm

    return TabularCMDP(
        name="real", horizon=H, states=states, action_names=action_names,
        T=T, r=r_mat, cost=cost, fire=fire, p0=p0,
        hub_actions=(2,), hub_dist=p0,
        notes=(f"replay over {n_eng} real FD001 engines (min length "
               f"{min(lens)} > horizon {H}: failure unreachable); features "
               f"and firing deterministic per state"),
    )


# ---------------------------------------------------------------------------
# Monte-Carlo verification (through the env's own noise process)
# ---------------------------------------------------------------------------

def mc_check_synthetic(env, U: Sequence[Candidate], n_draws: int,
                       seed: int = 7) -> dict:
    """>= n_draws observations per RUL state through env.observe(); compare the
    empirical firing-pattern frequencies with the analytic Gaussian CDFs."""
    rng = np.random.default_rng(seed)
    patterns, probs_fn = _pattern_table(U, env)
    ruls = np.arange(0, int(env.max_life) + 1)
    max_abs, max_se, worst = 0.0, 0.0, None
    for r in ruls:
        counts: Dict[Tuple[int, ...], int] = {}
        for _ in range(n_draws):
            obs = env.observe(float(r), rng)
            key = tuple(int(c.fires([obs], 0)) for c in U)
            counts[key] = counts.get(key, 0) + 1
        vec = probs_fn(float(r))
        ana = {patterns[j]: float(vec[j]) for j in range(len(vec))
               if vec[j] > 0}
        for key in set(list(counts) + list(ana)):
            emp = counts.get(key, 0) / n_draws
            th = ana.get(key, 0.0)
            se = math.sqrt(max(th * (1 - th), 1e-12) / n_draws)
            max_se = max(max_se, se)
            if abs(emp - th) > max_abs:
                max_abs = abs(emp - th)
                worst = dict(rul=int(r), pattern=list(key), analytic=th,
                             empirical=emp, mc_se=se)
    return dict(n_draws_per_state=int(n_draws), n_states=len(ruls),
                max_abs_diff=max_abs, max_mc_se=max_se, worst_case=worst,
                note="empirical firing-pattern frequencies via env.observe() "
                     "vs the analytic Gaussian-CDF pattern probabilities")


def mc_check_gridworld(env, m: TabularCMDP, n_episodes: int,
                       seed: int = 11) -> dict:
    """Simulate the env's own hazard automaton; verify P(unsafe|offset) and the
    onset hazard used by the belief MDP."""
    rng = np.random.default_rng(seed)
    CLEAR = m.S - 1
    unsafe_cnt = np.zeros(m.S)
    visit = np.zeros(m.S)
    onset_next = np.zeros(m.S)
    trans = np.zeros(m.S)
    for _ in range(n_episodes):
        state = env.initial(rng)
        o = CLEAR
        prev_o = None
        for _t in range(env.horizon):
            latent, obs = env.signals(state, rng)
            o = 0 if obs["warning"] else min(o + 1, CLEAR)
            if prev_o is not None:
                trans[prev_o] += 1
                if o == 0:
                    onset_next[prev_o] += 1
            visit[o] += 1
            unsafe_cnt[o] += 1 - obs["safe_true"]
            prev_o = o
            state = dict(pos=state["pos"], dleft=latent["dleft"],
                         cool=latent["cool"])
    emp_pu = np.divide(unsafe_cnt, visit, out=np.zeros(m.S), where=visit > 0)
    emp_h = np.divide(onset_next, trans, out=np.zeros(m.S), where=trans > 0)
    ana_pu = np.array([
        (env.r_advance - m.r[s, 0]) / (env.r_advance + env.c_accident)
        for s in range(m.S)])
    ana_h = np.array([m.T[0][s, 0] for s in range(m.S)])
    se_pu = np.sqrt(ana_pu * (1 - ana_pu) / np.maximum(visit, 1))
    se_h = np.sqrt(ana_h * (1 - ana_h) / np.maximum(trans, 1))
    return dict(
        n_episodes=int(n_episodes),
        min_state_visits=float(visit.min()),
        p_unsafe=dict(max_abs_diff=float(np.abs(emp_pu - ana_pu).max()),
                      max_mc_se=float(se_pu.max())),
        hazard=dict(max_abs_diff=float(np.abs(emp_h - ana_h).max()),
                    max_mc_se=float(se_h.max())),
        note="empirical P(unsafe|offset) and onset hazard from env.signals() "
             "vs the belief-MDP posteriors",
    )


# --- Monte-Carlo rollouts of extracted LP policies (end-to-end check) --------

def rollout_value(domain: str, env, U: Sequence[Candidate], m: TabularCMDP,
                  pol: Optional[np.ndarray], n_episodes: int, seed: int = 123,
                  mixture: Optional[Tuple[List[np.ndarray],
                                          np.ndarray]] = None) -> dict:
    rng = np.random.default_rng(seed)
    H = m.horizon
    totals = np.zeros(n_episodes)
    costs = {c.name: np.zeros(n_episodes) for c in U}
    active = np.zeros(n_episodes)
    act_state = m.active([c.name for c in U])
    sidx = {st: i for i, st in enumerate(m.states)}
    if domain == "synthetic":
        bitmap = {}
        for i, (rr, _j) in enumerate(m.states):
            bits = tuple(int(m.fire[c.name][i]) for c in U)
            bitmap[(rr, bits)] = i

    def score(ep, s, a):
        active[ep] += act_state[s]
        for c in U:
            costs[c.name][ep] += m.cost[c.name][s, a]

    plans, theta = mixture if mixture is not None else (None, None)
    cur_plan = None

    def choose(t, s):
        if cur_plan is not None:
            return int(cur_plan[t, s])
        return int(rng.choice(m.A, p=pol[t, s]))

    for ep in range(n_episodes):
        tot = 0.0
        if plans is not None:   # policy mixture: draw a column per episode
            cur_plan = plans[int(rng.choice(len(plans), p=theta / theta.sum()))]
        if domain == "synthetic":
            rul = env.initial_rul(rng)
            for t in range(H):
                obs = env.observe(rul, rng)
                bits = tuple(int(c.fires([obs], 0)) for c in U)
                s = bitmap.get((int(rul), bits))
                if s is None:      # pattern pruned as numerically impossible
                    a = 1 if any(bits) else 0
                else:
                    a = choose(t, s)
                    score(ep, s, a)
                r, rul, _ = env.step(rul, m.action_names[a])
                tot += r
        elif domain == "gridworld":
            state = env.initial(rng)
            CLEAR = m.S - 1
            o = CLEAR
            for t in range(H):
                latent, obs = env.signals(state, rng)
                o = 0 if obs["warning"] else min(o + 1, CLEAR)
                a = choose(t, o)
                score(ep, o, a)
                r, pos = env.step(state, m.action_names[a], obs)
                tot += r
                state = dict(pos=pos, dleft=latent["dleft"],
                             cool=latent["cool"])
        elif domain == "budget":
            state = env.initial(rng)
            for t in range(H):
                b = int(state["spend"] // env.unit_cost)
                a = choose(t, b)
                score(ep, b, a)
                r, state = env.step(state, m.action_names[a])
                tot += r
        elif domain == "real":
            st = env.reset(rng)
            for t in range(H):
                s = sidx[st]
                a = choose(t, s)
                score(ep, s, a)
                r, st, _ = env.step(st, m.action_names[a], rng)
                tot += r
        totals[ep] = tot
    out = dict(
        n_episodes=int(n_episodes),
        mean_return=float(totals.mean()),
        se_return=float(totals.std(ddof=1) / math.sqrt(n_episodes)),
        mean_active_steps=float(active.mean()),
    )
    for c in U:
        mc = float(costs[c.name].mean())
        out[f"cost[{c.name}]"] = dict(
            mean_episodic=mc,
            active_normalized=(mc / out["mean_active_steps"]
                               if out["mean_active_steps"] > 0 else 0.0))
    return out


# ---------------------------------------------------------------------------
# Cross-check investigations: exact re-evaluation of the paper's learned
# policies (the price study evaluates every policy with ONE fixed 40-episode
# seed, so its reported returns carry a shared, correlated MC offset; the
# certified dominance checks must therefore be run against the learned
# policies' EXACT values, not against those noisy evaluations)
# ---------------------------------------------------------------------------

def investigate_gridworld(env, U, m: TabularCMDP, data) -> dict:
    """Exact values of all offset-threshold policies; reproduce the paper's
    fixed-seed evaluation protocol; re-run the paper's learner and evaluate
    the learned single/robust policies EXACTLY in the belief MDP."""
    from .gridworld_rl import evaluate_gridworld_return, learn_gw_constrained

    n_o = m.S - 1
    out: dict = {}

    def thresh_actions(k: int) -> np.ndarray:
        # wait (action 1) iff offset <= k; CLEAR (index n_o) never waits
        return np.array([1 if (s < n_o and s <= k) else 0
                         for s in range(m.S)], dtype=np.int64)

    out["threshold_policy_exact_values"] = {
        f"wait<={k}": dp_policy_value(m, thresh_actions(k))
        for k in range(-1, n_o)}

    # the paper's evaluation protocol (n_episodes=40, seed=123) applied to the
    # canonical threshold policies -- fingerprints the price_report numbers
    from .gridworld_rl import offset_since_warning

    def thresh_pol(k):
        def pol(traj, t):
            return "wait" if offset_since_warning(traj, t) <= k else "advance"
        return pol

    out["paper_eval_protocol_reproduction"] = {
        f"wait<={k}": dict(
            eval_40ep_seed123=evaluate_gridworld_return(
                env, thresh_pol(k), n_episodes=40, seed=123),
            exact_value=out["threshold_policy_exact_values"][f"wait<={k}"])
        for k in (3, 4)}

    # re-run the paper's learner (seed-0 dataset) and evaluate exactly
    def learned_actions(policy) -> np.ndarray:
        acts = np.zeros(m.S, dtype=np.int64)
        for s in range(m.S):
            o = s if s < n_o else n_o          # CLEAR ~ any offset >= cap
            eff = min(o, 8)                    # learner's offset cap (CAP=8)
            traj = ([{"warning": 1}] + [{"warning": 0}] * eff
                    if eff < 8 else [{"warning": 0}] * 9)
            a = policy(traj, len(traj) - 1)
            acts[s] = m.action_names.index(a)
        return acts

    ret_stub = lambda pol: float("nan")
    learned = {}
    for tag, honor in (("single[psi_W3]", [U[0]]), ("robust", list(U))):
        res = learn_gw_constrained(data, honor=honor, U_eval=U,
                                   eps=EPS_BUDGET, return_fn=ret_stub,
                                   lam_grid=(0.0, 1.0, 2.0, 5.0, 10.0, 20.0,
                                             40.0, 80.0))
        acts = learned_actions(res.policy)
        learned[tag] = dict(
            actions_by_offset={str(m.states[s]): m.action_names[int(acts[s])]
                               for s in range(m.S)},
            exact_value=dp_policy_value(m, acts))
    out["learned_policies_reevaluated_exactly"] = learned
    return out


def investigate_real(env, U, m: TabularCMDP, data) -> dict:
    """Re-run the paper's learner on the seed-0 replay dataset and evaluate
    the learned policies EXACTLY (features are deterministic per state, and
    the learned policy reads only the current features, so it is a function
    of (engine, position) and its value is exact under dp_policy_value)."""
    from .experiments import LAM_GRID
    from .offline_rl import learn_fqi_constrained

    def learned_actions(policy) -> np.ndarray:
        acts = np.zeros(m.S, dtype=np.int64)
        for i, (ei, p) in enumerate(m.states):
            a = policy([env.engines[ei][p]], 0)
            acts[i] = m.action_names.index(a)
        return acts

    ret_stub = lambda pol: float("nan")
    out = {}
    jobs = [("robust", list(U))] + [(f"single[{c.name}]", [c]) for c in U]
    for tag, honor in jobs:
        res = learn_fqi_constrained(data, env, honor=honor, U_eval=U,
                                    eps=EPS_BUDGET, return_fn=ret_stub,
                                    bcq_tau=0.05, lam_grid=LAM_GRID)
        acts = learned_actions(res.policy)
        out[tag] = dict(exact_value=dp_policy_value(m, acts),
                        lambda_selected=res.lam)
    return out


# ---------------------------------------------------------------------------
# Per-domain study
# ---------------------------------------------------------------------------

GREEDY_ACTION = dict(synthetic="continue", real="continue",
                     gridworld="advance", budget="buy")
SAFE_ACTION = dict(synthetic="minor_repair", real="minor_repair",
                   gridworld="wait", budget="skip")


def study_domain(domain: str, env, U: Sequence[Candidate], m: TabularCMDP,
                 price_rows: List[dict], budgets: Sequence[float],
                 mc_draws: int, mc_episodes: int, skip_mc: bool,
                 data=None, normalization: str = "honored_active") -> dict:
    t0 = time.time()
    names = [c.name for c in U]
    a_greedy = m.action_names.index(GREEDY_ACTION[domain])
    a_safe = m.action_names.index(SAFE_ACTION[domain])
    default_act = guard_policy_actions(m, names, a_greedy, a_safe)

    # fixed per-reading normalizers Z_psi (plan 9.2), computed ONCE from the
    # return-greedy reference occupancy; set-independent, so identical across
    # the singleton/pair/full-set programs.  None in honored_active mode.
    Z = (fixed_psi_Z(m, names, a_greedy) if normalization == "fixed_psi"
         else None)

    def solve_prog(honored, d):
        return solve(m, honored, d, report_readings=names,
                     normalization=normalization, Z=Z)

    v_star = dp_optimal(m)
    v_greedy = dp_policy_value(m, np.full(m.S, a_greedy, dtype=np.int64))
    v_guard = dp_policy_value(m, default_act)

    out: dict = dict(
        U=names,
        n_states=m.S, n_actions=m.A, horizon=m.horizon,
        model_notes=m.notes,
        unconstrained=dict(
            V=v_star, method="exact finite-horizon DP (gamma=1)",
            always_greedy_value=v_greedy,
            greedy_action=GREEDY_ACTION[domain],
        ),
        guard_fallback=dict(
            V=v_guard,
            note="union-guard fallback (safe action wherever some retained "
                 "reading fires): zero semantic cost by construction, so it "
                 "certifies feasibility of every analyzed program (the "
                 "program-level fallback rate is 0)",
        ),
    )

    # solver selection: monolithic occupancy LP for small instances, the
    # equivalent Dantzig-Wolfe column-generation solve for large ones
    n_lp_vars = m.horizon * m.S * m.A
    use_cg = n_lp_vars > IPM_THRESHOLD
    solve = (solve_occupancy_cg if use_cg else solve_occupancy_lp)
    out["solver"] = ("dantzig-wolfe column generation (HiGHS master, exact "
                     "DP pricing; certified primal/dual bounds)" if use_cg
                     else "monolithic occupancy LP (HiGHS)")

    per_budget: dict = {}
    all_gaps: List[float] = []
    for d in budgets:
        key = f"d={d:g}"
        entry: dict = dict(singleton={}, pairwise={}, robust=None)
        sols: Dict[str, LPSolution] = {}

        # unconstrained LP as DP validation -- cheap domains only, once
        # (no honored readings, so normalization is irrelevant here)
        if not use_cg and n_lp_vars <= 500_000 and d == budgets[0]:
            sol_u = solve_occupancy_lp(m, [], d, report_readings=names,
                                       normalization=normalization, Z=Z)
            out["lp_vs_dp_unconstrained"] = dict(
                lp=sol_u.value, dp=v_star, abs_diff=abs(sol_u.value - v_star))
            all_gaps.append(sol_u.diagnostics["relative_duality_gap"])

        for c in U:
            sol = solve_prog([c.name], d)
            sols[c.name] = sol
            entry["singleton"][c.name] = dict(
                V=sol.value, on_policy=sol.on_policy, lp=sol.diagnostics)
            all_gaps.append(sol.diagnostics["relative_duality_gap"])

        sol_rob = solve_prog(names, d)
        entry["robust"] = dict(V=sol_rob.value, on_policy=sol_rob.on_policy,
                               lp=sol_rob.diagnostics)
        all_gaps.append(sol_rob.diagnostics["relative_duality_gap"])

        # cross-validation of the two solvers on the robust program
        if not use_cg and d == budgets[0]:
            sol_cg = solve_occupancy_cg(m, names, d, report_readings=names,
                                        normalization=normalization, Z=Z)
            out["cg_vs_lp_robust"] = dict(
                lp=sol_rob.value, cg=sol_cg.value,
                abs_diff=abs(sol_rob.value - sol_cg.value))

        pair_prices: Dict[str, float] = {}
        for i in range(len(U)):
            for j in range(i + 1, len(U)):
                pair = [U[i].name, U[j].name]
                pkey = " | ".join(pair)
                if len(U) == 2:
                    sol_p = sol_rob
                    diag = {"same_as": "robust (|U|=2)"}
                else:
                    sol_p = solve_prog(pair, d)
                    diag = sol_p.diagnostics
                    all_gaps.append(diag["relative_duality_gap"])
                best_single = max(entry["singleton"][n]["V"] for n in pair)
                pair_prices[pkey] = best_single - sol_p.value
                entry["pairwise"][pkey] = dict(
                    V=sol_p.value, pair_price=pair_prices[pkey], lp=diag)

        v_psi = {n: entry["singleton"][n]["V"] for n in names}
        v_u = sol_rob.value
        true_price = max(v_psi.values()) - v_u
        best_pair = max(pair_prices, key=pair_prices.get)
        entry["V_psi"] = v_psi
        entry["V_U"] = v_u
        entry["certified_true_price"] = true_price
        if normalization == "fixed_psi":
            wit_note = (
                "fixed per-reading normalization: every program uses the same "
                "set-independent budget d*Z_psi (Z_psi = return-greedy "
                "active-steps of psi), so each reading's constraint is "
                "identical across programs and the full-set feasible region is "
                "nested inside every pair's; V_pair >= V_U and no pair price "
                "exceeds the full-set price" if d > 0 else
                "at d = 0 the RHS is 0 in every mode; the programs nest exactly "
                "and coincide with honored_active (V_pair >= V_U, every pair "
                "price <= the full-set price)")
        else:
            wit_note = (
                "pairwise programs normalize the budget over the PAIR's own "
                "active set (the learners' convention), so for d > 0 they "
                "are not nested inside the full-set program and a pair "
                "price may slightly exceed the full-set price; at d = 0 "
                "nesting is exact" if d > 0 else
                "at d = 0 the programs nest exactly: V_pair >= V_U and "
                "every pair price <= the full-set price")
        entry["pair_witness"] = dict(
            best_pair=best_pair,
            best_pair_price=pair_prices[best_pair],
            all_pair_prices=pair_prices,
            witnesses_full_price=bool(
                pair_prices[best_pair] >= true_price - PAIR_TOL),
            tolerance=PAIR_TOL,
            note=wit_note,
        )

        if not skip_mc and d == budgets[0]:
            entry["mc_policy_rollouts"] = {}
            top_psi = max(v_psi, key=v_psi.get)
            for tag, sol in (("robust", sol_rob),
                             (f"singleton[{top_psi}]", sols[top_psi])):
                if sol.x is not None:
                    pol, mix = lp_policy(m, sol, default_act), None
                else:
                    pol, mix = None, (sol.columns, sol.theta)
                entry["mc_policy_rollouts"][tag] = dict(
                    lp_value=sol.value,
                    **rollout_value(domain, env, U, m, pol, mc_episodes,
                                    mixture=mix))
        per_budget[key] = entry

    out["programs"] = per_budget
    out["max_relative_duality_gap"] = float(max(all_gaps))

    # ------------------------------------------------------------------
    # learned-policy quantities from price_report.json + decomposition
    # ------------------------------------------------------------------
    prim = per_budget[f"d={budgets[0]:g}"]
    v_psi, v_u = prim["V_psi"], prim["V_U"]
    true_price = prim["certified_true_price"]
    rows_dec = []
    for row in price_rows:
        lv = {p["psi"]: p["v_psi"] for p in row["per_psi"]}
        lp_price = max(lv.values()) - row["v_robust"]
        rows_dec.append(dict(
            seed=row["seed"],
            learned_price=lp_price,
            robust_optimizer_error=v_u - row["v_robust"],
            singleton_shortfall=max(v_psi.values()) - max(lv.values()),
            identity_residual=lp_price - (
                true_price + (v_u - row["v_robust"])
                - (max(v_psi.values()) - max(lv.values()))),
        ))
    out["learned_from_price_report"] = dict(
        n_seeds=len(price_rows),
        v_psi_mean={n: float(np.mean([
            p["v_psi"] for row in price_rows for p in row["per_psi"]
            if p["psi"] == n])) for n in names},
        v_psi_max={n: float(np.max([
            p["v_psi"] for row in price_rows for p in row["per_psi"]
            if p["psi"] == n])) for n in names},
        v_robust_mean=float(np.mean([r["v_robust"] for r in price_rows])),
        v_robust_max=float(np.max([r["v_robust"] for r in price_rows])),
        v_surgery_max=float(np.max([p["v_surgery"] for row in price_rows
                                    for p in row["per_psi"]])),
        learned_price_mean=float(np.mean([r["learned_price"]
                                          for r in rows_dec])),
        learned_price_sd=float(np.std([r["learned_price"]
                                       for r in rows_dec])),
        surgery_feasible_rate=float(np.mean([
            p["surgery_worst"] <= EPS_BUDGET
            for row in price_rows for p in row["per_psi"]])),
        decomposition_per_seed=rows_dec,
        decomposition_summary=dict(
            certified_true_price=true_price,
            robust_optimizer_error_mean=float(np.mean(
                [r["robust_optimizer_error"] for r in rows_dec])),
            singleton_shortfall_mean=float(np.mean(
                [r["singleton_shortfall"] for r in rows_dec])),
            max_identity_residual=float(np.max(np.abs(
                [r["identity_residual"] for r in rows_dec]))),
            identity="learned_price = true_price + robust_optimizer_error "
                     "- singleton_shortfall (exact per seed)",
        ),
    )

    # ------------------------------------------------------------------
    # cross-checks
    # ------------------------------------------------------------------
    lf = out["learned_from_price_report"]
    checks: dict = {}
    checks["V_U_geq_learned_robust"] = dict(
        V_U=v_u, learned_robust_max=lf["v_robust_max"],
        margin=v_u - lf["v_robust_max"],
        passed=bool(v_u >= lf["v_robust_max"] - 1e-6))
    checks["V_U_geq_surgery"] = dict(
        V_U=v_u, v_surgery_max=lf["v_surgery_max"],
        margin=v_u - lf["v_surgery_max"],
        passed=bool(v_u >= lf["v_surgery_max"] - 1e-6))
    checks["V_psi_geq_learned_singleton"] = {
        n: dict(V_psi=v_psi[n], learned_max=lf["v_psi_max"][n],
                margin=v_psi[n] - lf["v_psi_max"][n],
                passed=bool(v_psi[n] >= lf["v_psi_max"][n] - 1e-6))
        for n in names}
    checks["all_passed"] = bool(
        checks["V_U_geq_learned_robust"]["passed"]
        and checks["V_U_geq_surgery"]["passed"]
        and all(v["passed"]
                for v in checks["V_psi_geq_learned_singleton"].values()))

    # any raw failure is investigated by re-running the paper's learner and
    # comparing EXACT policy values (the price study's returns are one fixed
    # 40-episode-seed MC evaluations and carry a shared, correlated offset)
    if not checks["all_passed"] and data is not None and domain in (
            "gridworld", "real"):
        inv = (investigate_gridworld(env, U, m, data) if domain == "gridworld"
               else investigate_real(env, U, m, data))
        tol = 1e-9
        if domain == "gridworld":
            L = inv["learned_policies_reevaluated_exactly"]
            resolved = (
                L["robust"]["exact_value"] <= v_u + tol
                and L["single[psi_W3]"]["exact_value"]
                <= v_psi[names[0]] + tol)
        else:
            resolved = (
                inv["robust"]["exact_value"] <= v_u + tol
                and all(inv[f"single[{n}]"]["exact_value"] <= v_psi[n] + tol
                        for n in names))
        checks["resolved_by_exact_reevaluation"] = bool(resolved)
        # exact optimizer error (immune to the shared eval-seed noise): V_U
        # minus the learned robust policy's EXACT value
        rob_exact = (L["robust"]["exact_value"] if domain == "gridworld"
                     else inv["robust"]["exact_value"])
        out["learned_from_price_report"]["robust_optimizer_error_exact"] = (
            v_u - rob_exact)
        out["cross_check_investigation"] = dict(
            cause=("price_report returns are single-fixed-seed 40-episode MC "
                   "evaluations (seed=123) sharing a correlated offset of "
                   "order SD/sqrt(40); certified dominance therefore has to "
                   "be checked against the learned policies' EXACT values, "
                   "recomputed here by re-running the paper's learner on the "
                   "seed-0 dataset and evaluating it in the exact chain"),
            resolved=bool(resolved),
            **inv)
    out["cross_checks"] = checks

    if not skip_mc:
        if domain == "synthetic":
            out["mc_integration"] = mc_check_synthetic(env, U, mc_draws)
        elif domain == "gridworld":
            out["mc_integration"] = mc_check_gridworld(
                env, m, max(mc_episodes, 60_000))
        else:
            out["mc_integration"] = dict(
                note="deterministic domain: firing and rewards are exact "
                     "functions of the state; no feature-noise integration "
                     "needed (the MC policy rollouts verify end-to-end)")

    out["elapsed_s"] = time.time() - t0
    return out


# ---------------------------------------------------------------------------
# LaTeX fragment
# ---------------------------------------------------------------------------

TEX_LABEL = dict(synthetic="Maint.\\ (synthetic)", real="Maint.\\ (C-MAPSS)",
                 gridworld="Warning-window", budget="Budget agent")


def write_tex(report: dict, path: Path, primary_budget: float) -> None:
    lines = ["% AUTO-GENERATED by saorl/exact_lp.py -- do not edit"]
    key = f"d={primary_budget:g}"
    for dom in ("synthetic", "real", "gridworld", "budget"):
        if dom not in report["domains"]:
            continue
        d = report["domains"][dom]
        prog = d["programs"][key]
        lf = d["learned_from_price_report"]
        cert = prog["certified_true_price"]
        # prefer the exact re-evaluated optimizer error where the noisy
        # fixed-seed evaluations made the price_report-based mean misleading
        opt_err = lf.get("robust_optimizer_error_exact",
                         lf["decomposition_summary"]
                         ["robust_optimizer_error_mean"])
        gap = d["max_relative_duality_gap"]
        gap_exp = int(math.floor(math.log10(gap))) + 1 if gap > 0 else -16
        pw = prog["pair_witness"]
        wit = "yes" if pw["witnesses_full_price"] else "no"
        lines.append(
            f"{TEX_LABEL[dom]} & {len(d['U'])} & ${cert:.2f}$ & "
            f"${lf['learned_price_mean']:.2f}\\pm{lf['learned_price_sd']:.2f}$"
            f" & ${opt_err:.2f}$ & gap $<10^{{{gap_exp}}}$ & "
            f"{wit} (${pw['best_pair_price']:.2f}$) \\\\")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(domains: Sequence[str], budgets: Sequence[float], mc_draws: int,
        mc_episodes: int, skip_mc: bool,
        normalization: str = "honored_active") -> dict:
    assert os.environ.get("SAORL_CONFORMAL") == "1", \
        "run with SAORL_CONFORMAL=1: prices are certified over the conformal sets"
    from .experiments import _domain_specs

    price = json.load(open(_PRICE_REPORT))
    specs = _domain_specs(list(domains), ["fqi"])
    builders: Dict[str, Callable] = dict(
        synthetic=build_synthetic, real=build_real,
        gridworld=build_gridworld, budget=build_budget)

    cfg = dict(
        budgets=list(budgets), eps_paper=EPS_BUDGET,
        constraint=("on-policy active-normalized expected semantic cost: "
                    "E[sum_t c_psi] <= d * E[sum_t 1{some honored reading "
                    "fires}] (the exact analogue of saorl/offline.py "
                    "normalize='active'; the learners enforce the dataset-"
                    "state version of the same quantity)"),
        policy_class=("Markov policies on the domain's exact sufficient state "
                      "augmented with the current firing pattern where "
                      "readings fire on noisy features; this class contains "
                      "every observation-history policy the paper evaluates, "
                      "so V-values are certified upper bounds for the paper's "
                      "policies and exact optima of the stated class"),
        solver="scipy.optimize.linprog(method='highs')",
        price_report=str(_PRICE_REPORT.relative_to(_ROOT)),
        gamma=1.0,
    )
    # fixed_psi is an ADDITIVE variant: the default (honored_active) config is
    # left byte-for-byte identical so lp_report.json reproduces exactly.
    if normalization == "fixed_psi":
        cfg["normalization"] = "fixed_psi"
        cfg["constraint"] = (
            "fixed per-reading normalization (plan section 9.2): E[sum_t "
            "c_psi(s_t,a_t)] <= d * Z_psi, with Z_psi a set-INDEPENDENT "
            "constant used identically in the singleton/pair/full-set programs")
        cfg["Z_definition"] = (
            "Z_psi = E_greedy[sum_t 1{psi fires at s_t}] = sum_s occ_greedy(s) "
            "* fire_psi(s); occ_greedy is the horizon-summed state visitation "
            "of the RETURN-GREEDY reference policy (the always-greedy action "
            "GREEDY_ACTION[domain], run from p0 through its own transition "
            "kernel).  Fixed before any retained set is built, independent of "
            "every reading and honored set; at d = 0 this coincides with "
            "honored_active (both reduce to E[sum_t c_psi] <= 0)")
    report: dict = dict(config=cfg, domains={})

    for dname in domains:
        if dname not in specs:
            print(f"  ({dname} skipped: spec unavailable)")
            continue
        print(f"[{dname}] building exact model...")
        _data, env, U, _ret_fn, _pool = specs[dname].build(0)
        pr_names = price["domains"][dname]["rows"][0]["u_names"]
        assert [c.name for c in U] == pr_names, (
            dname, [c.name for c in U], pr_names)
        m = builders[dname](env, U)
        print(f"[{dname}] S={m.S} A={m.A} H={m.horizon}; solving programs...")
        dom = study_domain(dname, env, U, m,
                           price["domains"][dname]["rows"], budgets,
                           mc_draws, mc_episodes, skip_mc, data=_data,
                           normalization=normalization)
        report["domains"][dname] = dom
        prog = dom["programs"][f"d={budgets[0]:g}"]
        ck = dom["cross_checks"]
        verdict = ("OK" if ck["all_passed"] else
                   "RESOLVED" if ck.get("resolved_by_exact_reevaluation")
                   else "FAIL")
        print(f"[{dname}] V*={dom['unconstrained']['V']:.3f} "
              f"V_U={prog['V_U']:.3f} "
              f"price={prog['certified_true_price']:.3f} "
              f"witness={prog['pair_witness']['witnesses_full_price']} "
              f"checks={verdict} ({dom['elapsed_s']:.0f}s)")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="exact occupancy-LP certification of the price of ambiguity")
    ap.add_argument("--domains", default="synthetic,real,gridworld,budget")
    ap.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS))
    ap.add_argument("--mc-draws", type=int, default=120_000,
                    help="observation draws per state (synthetic pattern check)")
    ap.add_argument("--mc-episodes", type=int, default=20_000,
                    help="episodes per MC policy rollout / automaton check")
    ap.add_argument("--skip-mc", action="store_true")
    ap.add_argument("--no-tex", action="store_true")
    ap.add_argument("--normalization", default="honored_active",
                    choices=["honored_active", "fixed_psi"],
                    help="honored_active (default; reproduces lp_report.json) "
                         "or fixed_psi (set-independent per-reading Z_psi)")
    ap.add_argument("--out", default=None,
                    help="output JSON path (default: lp_report.json, or "
                         "lp_report_fixedZ.json under --normalization fixed_psi)")
    args = ap.parse_args()

    norm = args.normalization
    fixed = (norm == "fixed_psi")
    out = Path(args.out) if args.out else (
        _OUT_DIR / ("lp_report_fixedZ.json" if fixed else "lp_report.json"))
    tex_out = (_ROOT / "paper" / "generated" /
               ("gen_lp_fixedZ.tex" if fixed else "gen_lp.tex"))

    t0 = time.time()
    budgets = [float(b) for b in args.budgets.split(",")]
    report = run([d for d in args.domains.split(",") if d], budgets,
                 args.mc_draws, args.mc_episodes, args.skip_mc,
                 normalization=norm)
    report["elapsed_s"] = time.time() - t0

    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    print(f"wrote {out}  ({report['elapsed_s']:.0f}s)")

    if not args.no_tex and set(report["domains"]) >= {"synthetic", "real",
                                                      "gridworld", "budget"}:
        write_tex(report, tex_out, budgets[0])
        print(f"wrote {tex_out}")


if __name__ == "__main__":
    main()
