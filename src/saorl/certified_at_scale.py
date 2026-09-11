"""A7: certified-at-scale — the actual Theorem-2 estimator on the P1 suite.

For every source-grounded rule instance (the 17 compiled by the P1
control suite), at 3 logged-data sizes x R independent draws:
  * frozen generic behavior policy (episodes from mu0, uniform actions,
    length 40) — independent of psi_src;
  * counts -> support class (n(s,a) >= m_supp) -> corrected radius
    b(s,a) = min{2, sqrt(2(S ln2 + ln(SA N / delta_tr)) / n(s,a))}
    (the T0.1-synchronized form) -> penalized robust occupancy LP in the
    estimated model over the support class;
  * infeasible program -> fallback pi0 = always-intervene (pathwise-zero,
    mode (i), certified cost 0).

Per-draw events (the five split columns of A7):
  conf_event      ||Phat - P||_1 <= b on the support class
  returned_opt    the LP returned an optimizer (vs fallback)
  true_feasible   max_k J^P_{c_k}(pi_hat) <= d in the TRUE model
  margin_ok       the exact-LP oracle pi* is margin-feasible
                  (max_k J_ck(pi*) <= d - 2 beta_c(pi*))
  bound_covered   J_r(pi_hat) >= J_r(pi*) - 2 beta_r(pi*)  (when margin_ok)

Also per instance: exact-LP PoA benchmark and the switch-gap endpoint
(exact E[gamma^tau (V_r^pi(s_tau) - V_r^{pi0}(s_tau))] in the product
chain) vs the closed form and the realized two-rollout gap.

Registered: Addendum A7. Run:
  SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.certified_at_scale
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from saorl.benchmark_sg.control_suite import compile_instance
from saorl.benchmark_sg.control_mdp import GAMMA

_ROOT = Path(__file__).resolve().parents[2]
_SUITE = _ROOT / "results/conformal" / "benchmark_sg" / "control_suite.json"
_OUT = _ROOT / "results/conformal" / "lp" / (
    "certified_at_scale.json" if __import__("os").environ.get("A7_D", "0.01") == "0.01"
    else "certified_at_scale_d%s.json" % __import__("os").environ["A7_D"].replace("0.", "").replace(".", ""))

D_BUDGET = float(__import__("os").environ.get("A7_D", "0.01"))
DELTA_TR = 0.10
M_SUPP = 5
SIZES = (2000, 8000, 32000)
DRAWS = 50
EP_LEN = 40
SEED0 = 20260801


def flow_matrices(P, mu0):
    nS, nA, _ = P.shape
    A_eq = np.zeros((nS, nS * nA))
    for s in range(nS):
        for a in range(nA):
            A_eq[s, s * nA + a] += 1.0
    A_eq -= GAMMA * np.transpose(P, (2, 0, 1)).reshape(nS, nS * nA)
    return A_eq, (1.0 - GAMMA) * mu0


def solve_lp(P, r, Cs, mu0, d, mask=None):
    """Occupancy LP; mask: boolean (nS*nA) allowed pairs (support class)."""
    nS, nA, _ = P.shape
    A_eq, b_eq = flow_matrices(P, mu0)
    bounds = [(0, None)] * (nS * nA)
    if mask is not None:
        bounds = [(0, None) if m else (0, 0) for m in mask]
    A_ub = np.array([c.reshape(-1) for c in Cs]) if len(Cs) else None
    b_ub = np.array([d] * len(Cs)) if len(Cs) else None
    res = linprog(-r.reshape(-1), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                  b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        return None
    return res.x


def policy_of(x, nS, nA):
    x = x.reshape(nS, nA)
    pi = np.zeros((nS, nA))
    tot = x.sum(1)
    for s in range(nS):
        if tot[s] > 1e-12:
            pi[s] = x[s] / tot[s]
        else:
            pi[s, 1] = 1.0          # default to intervene off-support
    return pi


def eval_policy(P, f, mu0, pi):
    """Exact value of stationary pi in the SAME normalized units as the
    occupancy LP (flow constraint uses (1-gamma) mu0, so LP values are
    (1-gamma) x discounted): J = (1-gamma) mu0' v."""
    nS, nA, _ = P.shape
    Ppi = np.einsum("sap,sa->sp", P, pi)
    fpi = np.einsum("sa,sa->s", f, pi)
    v = np.linalg.solve(np.eye(nS) - GAMMA * Ppi, fpi)
    return float((1.0 - GAMMA) * (mu0 @ v)), v


def occupancy_of(P, mu0, pi):
    nS, nA, _ = P.shape
    Ppi = np.einsum("sap,sa->sp", P, pi)
    d_s = np.linalg.solve(np.eye(nS) - GAMMA * Ppi.T, (1 - GAMMA) * mu0)
    return d_s[:, None] * pi        # (nS, nA) discounted occupancy


def gen_data(P, mu0, N, rng):
    nS, nA, _ = P.shape
    counts = np.zeros((nS, nA), dtype=int)
    trans = np.zeros((nS, nA, nS), dtype=int)
    n = 0
    while n < N:
        s = rng.choice(nS, p=mu0)
        for _ in range(EP_LEN):
            a = rng.integers(nA)
            s2 = rng.choice(nS, p=P[s, a])
            counts[s, a] += 1
            trans[s, a, s2] += 1
            s = s2
            n += 1
            if n >= N:
                break
    return counts, trans


def certified_run(m, d, N, rng):
    P, r, C, mu0 = m["P"], m["r"], m["C"], m["mu0"]
    nS, nA, _ = P.shape
    S_ln2 = nS * np.log(2.0)
    counts, trans = gen_data(P, mu0, N, rng)
    Phat = np.zeros_like(P)
    b = np.full((nS, nA), 2.0)
    for s in range(nS):
        for a in range(nA):
            n = counts[s, a]
            if n > 0:
                Phat[s, a] = trans[s, a] / n
                b[s, a] = min(2.0, np.sqrt(
                    2 * (S_ln2 + np.log(nS * nA * N / DELTA_TR)) / n))
            else:
                Phat[s, a, s] = 1.0   # unvisited: self-loop placeholder
    mask = (counts >= M_SUPP).reshape(-1)
    # ensure the fallback (always-intervene) stays representable: allow
    # (s,1) pairs regardless of support (their estimated transition is the
    # certified true one: intervene -> safe deterministically by contract)
    safe_idx = 0
    for s in range(nS):
        Phat[s, 1] = 0.0
        Phat[s, 1, safe_idx] = 1.0   # mechanically certified contract
        b[s, 1] = 0.0
    mask = mask.reshape(nS, nA)
    mask[:, 1] = True
    mask = mask.reshape(-1)

    Gam = 1.0 / (1.0 - GAMMA)
    Vr = float(r.max() - r.min()) * Gam
    Vc = Gam
    r_pen = r - GAMMA * Vr * b
    C_pen = [C[k] + GAMMA * Vc * b for k in range(C.shape[0])]
    # certified fallback exemption: intervene column has b=0 already
    x = solve_lp(Phat, r_pen, C_pen, mu0, d, mask=mask)
    if x is None:
        pi = np.zeros((nS, nA))
        pi[:, 1] = 1.0
        returned_opt = False
    else:
        pi = policy_of(x, nS, nA)
        returned_opt = True
    # events in the true model
    conf = True
    for s in range(nS):
        for a in range(nA):
            if counts[s, a] >= M_SUPP:
                if np.abs(Phat[s, a] - P[s, a]).sum() > b[s, a] + 1e-12:
                    conf = False
                    break
        if not conf:
            break
    Jc = [eval_policy(P, C[k], mu0, pi)[0] for k in range(C.shape[0])]
    Jr = eval_policy(P, r, mu0, pi)[0]
    return dict(conf=conf, returned_opt=returned_opt,
                true_feasible=max(Jc) <= d + 1e-9, Jr=Jr,
                maxJc=max(Jc), b_mean=float(b[counts >= M_SUPP].mean())
                if (counts >= M_SUPP).any() else 2.0)


def margin_and_bound(m, d, oracle_x, pi_hat_Jr, Phat_occ_terms):
    return None  # margin computed in caller with estimated occupancies


def switch_gap(m, d):
    """Exact switch-gap vs closed form for each singleton oracle."""
    P, r, C, mu0 = m["P"], m["r"], m["C"], m["mu0"]
    nS, nA, _ = P.shape
    K = C.shape[0]
    x_full = solve_lp(P, r, list(C), mu0, d)
    rows = []
    for k in range(K):
        xk = solve_lp(P, r, [C[k]], mu0, d)
        if xk is None:
            continue
        pik = policy_of(xk, nS, nA)
        Jr_k, Vr_k = eval_policy(P, r, mu0, pik)   # Jr_k normalized
        # disputed region for reading j!=k: pairs where c_j fires under pik
        Dsp = np.zeros((nS, nA), dtype=bool)
        for j in range(K):
            if j != k:
                Dsp |= (C[j] > 0)
        # product chain: state x {pre,post}; pre follows pik but switching
        # (to always-intervene) at disputed pairs; exact eval by expanding
        pi0 = np.zeros((nS, nA))
        pi0[:, 1] = 1.0
        _, Vr_pi0 = eval_policy(P, r, mu0, pi0)
        # pre-switch dynamics: from s take a~pik; if (s,a) disputed ->
        # switch occurs AT that step (take intervene instead)
        # value of surgery policy:
        Ppre = np.zeros((nS, nS))
        fpre = np.zeros(nS)
        gap_flow = np.zeros(nS)     # E[gamma^tau (Vr_pik - Vr_pi0)(s_tau)]
        Psw = np.zeros((nS, nS))
        for s in range(nS):
            for a in range(nA):
                p = pik[s, a]
                if p <= 0:
                    continue
                if Dsp[s, a]:
                    # switched: act intervene now
                    fpre[s] += p * r[s, 1]
                    Psw[s] += p * P[s, 1]
                else:
                    fpre[s] += p * r[s, a]
                    Ppre[s] += p * P[s, a]
        # J(pi') = mu0 [ (I - g Ppre)^{-1} (fpre + g Psw Vr_pi0) ]
        v_surg = np.linalg.solve(np.eye(nS) - GAMMA * Ppre,
                                 fpre + GAMMA * Psw @ Vr_pi0)
        Jr_surg = float((1.0 - GAMMA) * (mu0 @ v_surg))
        # switch-gap bound: E[sum over switch events gamma^tau
        #   (Vr_pik(s_tau) - Vr_pi0(s_tau))], s_tau = state AT switch
        sw_ind = np.einsum("sa,sa->s", pik, Dsp.astype(float))
        g_vec = sw_ind * (Vr_k - Vr_pi0)
        occ_pre = np.linalg.solve(
            np.eye(nS) - GAMMA * Ppre.T, (1.0 - GAMMA) * mu0)
        sg_bound = float(occ_pre @ g_vec)   # g_vec uses discounted V's:
        # normalized occupancy x discounted value-gap = normalized units
        # closed form: Dr * E[kappa(tau)] with Dr = span(r)*Gam
        Gam = 1.0 / (1.0 - GAMMA)
        Dr = float(r.max() - r.min()) * Gam   # discounted-span
        p_switch = float(occ_pre @ sw_ind)    # normalized switch mass
        closed = Dr * p_switch                # normalized closed form
        rows.append(dict(k=k, Jr_oracle=Jr_k, Jr_surgery=Jr_surg,
                         realized_gap=Jr_k - Jr_surg,
                         switch_gap_bound=sg_bound, closed_form=closed))
    Jr_full = None
    if x_full is not None:
        Jr_full = float(m["r"].reshape(-1) @ x_full)
    return rows, Jr_full


def main():
    suite = json.loads(_SUITE.read_text())
    rng0 = np.random.default_rng(SEED0)
    out = dict(delta_tr=DELTA_TR, m_supp=M_SUPP, sizes=list(SIZES),
               draws=DRAWS, d=D_BUDGET, instances=[])
    for inst in suite["instances"]:
        readings = [type("R", (), dict(threshold=th, for_s=fs))()
                    for th, fs in
                    [(x["theta"], x["for_s"]) for x in inst["readings"]]]
        m = compile_instance(readings)
        P, r, C, mu0 = m["P"], m["r"], m["C"], m["mu0"]
        nS, nA, _ = P.shape
        x_star = solve_lp(P, r, list(C), mu0, D_BUDGET)
        Jr_star = float(r.reshape(-1) @ x_star) if x_star is not None \
            else None
        pi_star = policy_of(x_star, nS, nA) if x_star is not None else None
        sg_rows, Jr_full = switch_gap(m, D_BUDGET)
        irec = dict(instance=inst["uid"],
                    geometry=inst["geometry"], nS=nS,
                    Jr_oracle_fullset=Jr_star, switch_gap=sg_rows,
                    sizes={})
        for N in SIZES:
            cols = dict(conf=0, opt=0, feas_opt=0, feas_sub=0,
                        margin=0, bound=0, n=DRAWS)
            for rdx in range(DRAWS):
                import hashlib as _h
                hseed = int.from_bytes(_h.sha256(
                    f"{inst['uid']}|{N}|{rdx}".encode()
                ).digest()[:4], "big")
                rng = np.random.default_rng(SEED0 + hseed)
                res = certified_run(m, D_BUDGET, N, rng)
                cols["conf"] += res["conf"]
                cols["opt"] += res["returned_opt"]
                if res["returned_opt"]:
                    cols["feas_opt"] += res["true_feasible"]
                cols["feas_sub"] += (res["true_feasible"]
                                     if res["returned_opt"] else 1)
                if pi_star is not None:
                    # margin + bound with beta from estimated occupancy of
                    # pi*: approximate with true occupancy x b_mean (exact
                    # per-draw estimated-occupancy beta is recorded via
                    # b_mean; conservative diagnostic)
                    Gam = 1.0 / (1.0 - GAMMA)
                    beta_c = GAMMA * Gam * res["b_mean"]
                    Jc_star = max(float(C[k].reshape(-1) @ x_star)
                                  for k in range(C.shape[0]))
                    mok = Jc_star <= D_BUDGET - 2 * beta_c
                    cols["margin"] += mok
                    if mok:
                        Vr = float(r.max() - r.min()) * Gam
                        beta_r = GAMMA * Vr * res["b_mean"] / Gam
                        cols["bound"] += (res["Jr"] >=
                                          Jr_star - 2 * beta_r - 1e-9)
            irec["sizes"][N] = {k: (v / DRAWS if k != "n" else v)
                                for k, v in cols.items()}
        out["instances"].append(irec)
        print(inst["uid"], inst["geometry"],
              {N: round(irec["sizes"][N]["feas_sub"], 3) for N in SIZES})
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(out, indent=1))
    print("wrote", _OUT)


if __name__ == "__main__":
    main()
