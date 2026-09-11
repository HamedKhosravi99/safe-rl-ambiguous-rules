"""REGISTRATION_V24: the exposure ceiling of the certificate normalizer.

For every domain and retained reading: Z under the always-greedy reference
(asserted equal to the archive), under the uniform-random reference, and
the ceiling Z_max = max over all policies of the expected steps spent in
psi-firing states (finite-horizon DP on the tabular model), with the range
B, the zero-violation sample n_0 and the Z needed at n = 300 implied by each.

Run: SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.exposure_ceiling [--with-run results/conformal/lp/evaluator_scale_50k.json]
Writes results/conformal/lp/exposure_ceiling.json
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from .alt_defense import GREEDY_ACTION
from .exact_lp import _ROOT, build_budget, build_gridworld, build_real, build_synthetic, fixed_psi_Z
from .expected_cost_certificate import D_BUDGET, LEVEL, cp_upper

OUT = _ROOT / "results/conformal" / "lp" / "exposure_ceiling.json"
BUILDER = dict(synthetic=build_synthetic, real=build_real, gridworld=build_gridworld, budget=build_budget)
N_AUDIT = 300


def uniform_Z(m, names):
    A = len(m.action_names)
    TaT = [m.T[a].T.tocsr() for a in range(A)]
    occ = np.zeros(m.S)
    mu = m.p0.copy()
    for _ in range(m.horizon):
        occ += mu
        mu = sum(T @ mu for T in TaT) / A
    return {k: float(occ @ m.fire[k]) for k in names}


def max_Z(m, names):
    """max over all (history-dependent, hence Markov) policies of E[sum_{t<H} fire_psi(s_t)]."""
    A = len(m.action_names)
    out = {}
    for k in names:
        f = np.asarray(m.fire[k], dtype=float)
        V = np.zeros(m.S)
        for _ in range(m.horizon):
            V = f + np.max(np.column_stack([m.T[a] @ V for a in range(A)]), axis=1)
        out[k] = float(m.p0 @ V)
    return out


def n_zero(B, d=D_BUDGET, nmax=100_000_000):
    lo, hi = 1, nmax
    if B * cp_upper(0, hi, LEVEL) > d:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if B * cp_upper(0, mid, LEVEL) <= d:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-run", default=None)
    a = ap.parse_args()
    from .experiments import _domain_specs
    arch_all = json.load(open(_ROOT / "results/conformal" / "lp" / "expected_cost_certificate.json"))["domains"]
    dep_all = json.load(open(_ROOT / "results/conformal" / "lp" / "deploy_certificate.json"))["domains"]
    cp0 = cp_upper(0, N_AUDIT, LEVEL)
    res = dict(registration="REGISTRATION_V24.md", d=D_BUDGET, level=LEVEL, n_audit=N_AUDIT, cp_zero_at_audit=cp0, domains={})
    for domain in ("synthetic", "real", "gridworld", "budget"):
        spec = _domain_specs([domain], ["fqi"])[domain]
        data, env, U, ret_fn, _raw = spec.build(0)
        names = [c.name for c in U]
        m = BUILDER[domain](env, list(U))
        a_g = m.action_names.index(GREEDY_ACTION[domain])
        Zg = fixed_psi_Z(m, names, a_g)
        Zu = uniform_Z(m, names)
        Zm = max_Z(m, names)
        arch = arch_all[domain]
        maximal = set(dep_all[domain]["maximal"]) if isinstance(dep_all[domain].get("maximal"), list) else None
        per = {}
        for k in names:
            ar = arch["per_reading"][k]
            assert abs(Zg[k] - ar["Z"]) < 1e-6, (domain, k, Zg[k], ar["Z"])
            raw_max = float(ar["raw_max"])
            row = dict(raw_max=raw_max, Z_greedy=Zg[k], Z_uniform=Zu[k], Z_max=Zm[k],
                       maximal=(k in maximal) if maximal is not None else None)
            for tag, z in (("greedy", Zg[k]), ("uniform", Zu[k]), ("max", Zm[k])):
                B = raw_max / z if z > 0 else float("inf")
                row[f"B_{tag}"] = B
                row[f"n_zero_{tag}"] = n_zero(B)
            row["Z_need_k0"] = raw_max * cp0 / D_BUDGET
            row["ceiling_short_k0"] = bool(Zm[k] < row["Z_need_k0"])
            per[k] = row
        dom = dict(horizon=int(m.horizon), S=int(m.S), greedy_action=GREEDY_ACTION[domain], B_archived=arch["B"],
                   w=dep_all[domain]["w"], per_reading=per,
                   B_max_over_readings=dict(greedy=max(p["B_greedy"] for p in per.values()),
                                            uniform=max(p["B_uniform"] for p in per.values()),
                                            ceiling=max(p["B_max"] for p in per.values())))
        dom["ceiling_short_any_maximal_k0"] = any(p["ceiling_short_k0"] for p in per.values() if p["maximal"] in (True, None))
        res["domains"][domain] = dom
        print(domain, json.dumps({k: {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in p.items()} for k, p in per.items()}, indent=None))
    # the audited C-MAPSS seed: Z needed with its archived nine violations
    real = res["domains"]["real"]
    k9 = 9
    cp9 = cp_upper(k9, N_AUDIT, LEVEL)
    for k, p in real["per_reading"].items():
        p["Z_need_k9"] = p["raw_max"] * cp9 / D_BUDGET
        p["ceiling_short_k9"] = bool(p["Z_max"] < p["Z_need_k9"])
    real["cp_nine_at_audit"] = cp9
    if a.with_run:
        run = json.load(open(a.with_run))
        exp = {}
        for t in run["tasks"]:
            if "count_multiset" not in t:
                continue
            key = f"{t['domain']}/{t['arm']}"
            names = t["reading_names"]
            tot = 0; F = np.zeros(len(names)); C = np.zeros(len(names))
            for tup, cnt in t["count_multiset"]:
                c_part, f_part = tup[: len(names)], tup[len(names):]
                tot += cnt; C += cnt * np.asarray(c_part, float); F += cnt * np.asarray(f_part, float)
            exp.setdefault(key, []).append(dict(seed=t["seed"], E_fire={n: float(F[i] / tot) for i, n in enumerate(names)},
                                                E_charged={n: float(C[i] / tot) for i, n in enumerate(names)}))
        res["own_exposure"] = {key: dict(n_seeds=len(v), median_E_fire={n: float(np.median([s["E_fire"][n] for s in v])) for n in v[0]["E_fire"]},
                                         max_E_fire={n: float(max(s["E_fire"][n] for s in v)) for n in v[0]["E_fire"]})
                               for key, v in exp.items()}
        print(json.dumps(res["own_exposure"], indent=1))
    OUT.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
