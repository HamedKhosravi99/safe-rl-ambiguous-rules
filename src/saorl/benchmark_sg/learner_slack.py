"""V45: approximate-learner slack, read from the archives (no new runs).

eta   = max(0, C_max - d)/d        constraint-feasibility error (true model)
eps_r = (V_U - J_r)/V_U            reward gap to the full-set optimum
from safe_face_offline.json (V29, Lagrangian FQI, d = 0.005) and
three_way.json (V38, occupancy LP on P_hat, rho in {0, 0.5, 0.7}); and the
tightening check: observed loss of the LP at rho = 0.5 against the exact
price V_U(d) - V_U(d/2) from ambiguity_profile.json (V39).  Registration V45.

Run: PYTHONPATH=. python3 -m saorl.benchmark_sg.learner_slack
Writes results/e2e/learner_slack.json
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
R = os.path.join(ROOT, "results/e2e")
OUT = os.path.join(R, "learner_slack.json")
D = 0.005


def stats(eta, eps):
    eta = np.asarray(eta); eps = np.asarray(eps)
    return dict(n=int(eta.size), eta_pos_frac=float(np.mean(eta > 1e-9)), eta_median_pos=(float(np.median(eta[eta > 1e-9])) if (eta > 1e-9).any() else 0.0),
                eta_p90=float(np.percentile(eta, 90)), eps_r_median=float(np.median(eps)), eps_r_mean=float(np.mean(eps)), eps_r_p90=float(np.percentile(eps, 90)))


def main():
    out = dict(registration="V45 (REGISTRATION_V28.md)", d=D, learners={})
    v29 = json.load(open(os.path.join(R, "safe_face_offline.json")))
    for n in ("2000", "20000"):
        for learner in ("fullset", "single"):
            eta, eps = [], []
            for r in v29["rows"]:
                if str(r["n"]) != n:
                    continue
                L = r["learners"][learner]
                eta.append(max(0.0, L["C_max"] - D) / D); eps.append((r["V_U"] - L["J_r"]) / r["V_U"])
            out["learners"][f"fqi_lagrangian_{learner}_n{n}"] = stats(eta, eps)
    v38 = json.load(open(os.path.join(R, "three_way.json")))
    for arm in ("fullset_rho0.0", "fullset_rho0.5", "fullset_rho0.7", "single_rho0.0", "single_rho0.5"):
        eta, eps = [], []
        for r in v38["rows"]:
            a = r["arms"].get(arm)
            if not a or not a.get("feasible", True) or a.get("ret") is None:
                continue
            eta.append(max(0.0, a["C_over_d"] - 1.0)); eps.append(1.0 - a["ret"])   # ret is a fraction of V_U
        out["learners"][f"occupancy_lp_{arm}"] = stats(eta, eps)
    # tightening check: observed LP loss at rho = 0.5 vs exact price V_U(d) - V_U(d/2)
    v39 = json.load(open(os.path.join(R, "ambiguity_profile.json"))); kf = v39["kappa_frac"]; i_half = int(np.argmin(np.abs(np.asarray(kf) - 0.5)))
    price = {}
    for rec in v39["rows"]:
        pr = rec["profiles"].get(str(D))
        if not pr:
            continue
        VU = pr["V_U"]; anchors = pr["anchors"]
        # V_U(d - kappa) = V_psi - A_psi(kappa) for any anchor; take the first anchor with a finite value
        for k, a in anchors.items():
            if a["A"][i_half] is not None:
                VU_half = a["V_psi"] - a["A"][i_half]; price[rec["uid"]] = (VU - VU_half) / VU; break
    obs = {}
    for r in v38["rows"]:
        a = r["arms"].get("fullset_rho0.5")
        if a and a.get("ret") is not None:
            obs.setdefault(r["rule_id"], []).append(1.0 - a["ret"])
    pairs = [(np.mean(obs[u]), price[u]) for u in obs if u in price]
    diff = np.array([o - p for o, p in pairs])
    out["tightening_check"] = dict(n_instances=len(pairs), observed_loss_mean=float(np.mean([o for o, _ in pairs])), exact_price_mean=float(np.mean([p for _, p in pairs])),
                                   paired_diff_mean=float(diff.mean()), paired_diff_max_abs=float(np.abs(diff).max()),
                                   note="observed = 1 - ret of the buffered LP (rho = 0.5 d) on the true model, mean over seeds; exact price = (V_U(d) - V_U(d/2))/V_U(d) from V39")
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
