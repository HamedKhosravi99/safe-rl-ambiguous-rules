"""T3.1: exact switch-state value-gap form of Theorem 4 (P1 instances).

For a STATIONARY singleton oracle pi_k and its surgery policy pi', the
exact identity (one-step decomposition at the switch event) is

  J_r(pi_k) - J_r(pi') =
    sum_s occ_pre(s) sum_a pi_k(a|s) 1{(s,a) disputed}
        [ Q^{pi_k}(s,a) - (r(s,1) + gamma P(s,1)' V^{pi0}) ] ,

where occ_pre is the (1-gamma)-normalized occupancy of the pre-switch
chain. The evaluation-free closed form replaces the bracket by its
worst-case span Delta_R. Self-check: the identity must match the
independently computed realized gap to numerical precision on every
instance; the Lemma-2 occupancy-equivalence step is NOT needed because
the oracles here are stationary (V, Q well-defined at the switch state).

Run: SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.t31_switchgap
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from saorl.benchmark_sg.control_mdp import GAMMA
from saorl.benchmark_sg.control_suite import compile_instance
from saorl.certified_at_scale import eval_policy, policy_of, solve_lp

_ROOT = Path(__file__).resolve().parents[2]
_SUITE = _ROOT / "results/conformal" / "benchmark_sg" / "control_suite.json"
_OUT = _ROOT / "results/conformal" / "lp" / "switch_gap_exact.json"
D = 0.01


def main():
    suite = json.loads(_SUITE.read_text())
    rows = []
    for inst in suite["instances"]:
        readings = [type("R", (), dict(threshold=x["theta"],
                                       for_s=x["for_s"]))()
                    for x in inst["readings"]]
        m = compile_instance(readings)
        P, r, C, mu0 = m["P"], m["r"], m["C"], m["mu0"]
        nS, nA, _ = P.shape
        K = C.shape[0]
        pi0 = np.zeros((nS, nA))
        pi0[:, 1] = 1.0
        _, Vr_pi0 = eval_policy(P, r, mu0, pi0)   # discounted v-vector
        for k in range(K):
            xk = solve_lp(P, r, [C[k]], mu0, D)
            if xk is None:
                continue
            pik = policy_of(xk, nS, nA)
            Jk_norm, Vk = eval_policy(P, r, mu0, pik)
            Dsp = np.zeros((nS, nA), dtype=bool)
            for j in range(K):
                if j != k:
                    Dsp |= (C[j] > 0)
            # pre-switch chain + surgery value (independent computation)
            Ppre = np.zeros((nS, nS))
            fpre = np.zeros(nS)
            Psw = np.zeros((nS, nS))
            for s_ in range(nS):
                for a in range(nA):
                    p = pik[s_, a]
                    if p <= 0:
                        continue
                    if Dsp[s_, a]:
                        fpre[s_] += p * r[s_, 1]
                        Psw[s_] += p * P[s_, 1]
                    else:
                        fpre[s_] += p * r[s_, a]
                        Ppre[s_] += p * P[s_, a]
            v_surg = np.linalg.solve(np.eye(nS) - GAMMA * Ppre,
                                     fpre + GAMMA * Psw @ Vr_pi0)
            J_surg_norm = float((1 - GAMMA) * (mu0 @ v_surg))
            realized = Jk_norm - J_surg_norm
            # exact switch-gap: occ_pre (normalized) x disputed-action
            # advantage vs fallback continuation
            occ_pre = np.linalg.solve(np.eye(nS) - GAMMA * Ppre.T,
                                      (1 - GAMMA) * mu0)
            Q = r + GAMMA * np.einsum("sap,p->sa", P, Vk)
            fb_cont = r[:, 1] + GAMMA * P[:, 1, :] @ Vr_pi0
            g = 0.0
            for s_ in range(nS):
                for a in range(nA):
                    if Dsp[s_, a] and pik[s_, a] > 0:
                        g += occ_pre[s_] * pik[s_, a] * \
                            (Q[s_, a] - fb_cont[s_])
            # closed form: worst-case span x switch mass
            Gam = 1.0 / (1.0 - GAMMA)
            Dr = float(r.max() - r.min()) * Gam
            sw_mass = float(occ_pre @ np.einsum(
                "sa,sa->s", pik, Dsp.astype(float)))
            closed = Dr * sw_mass * (1 - GAMMA) * Gam  # normalized units
            closed = float(r.max() - r.min()) * Gam * sw_mass
            rows.append(dict(
                uid=inst["uid"], k=k, realized=realized, exact_form=g,
                identity_err=abs(realized - g),
                closed_form=closed,
                ratio_closed=closed / realized if realized > 1e-9
                else None,
                ratio_exact=g / realized if realized > 1e-9 else None))
    errs = [x["identity_err"] for x in rows]
    ratios = [x["ratio_closed"] for x in rows if x["ratio_closed"]]
    out = dict(n=len(rows), max_identity_err=max(errs) if errs else None,
               closed_over_realized=dict(
                   median=float(np.median(ratios)) if ratios else None,
                   max=float(np.max(ratios)) if ratios else None),
               rows=rows)
    _OUT.write_text(json.dumps(out, indent=1))
    print(f"n={out['n']} max identity err={out['max_identity_err']:.2e}")
    print("closed/realized median",
          round(out['closed_over_realized']['median'], 1),
          "max", round(out['closed_over_realized']['max'], 1))


if __name__ == "__main__":
    main()
