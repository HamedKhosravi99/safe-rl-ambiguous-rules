"""Is the barrier the CERTIFICATE or the POLICY CLASS? (decisive variant)

The coverage diagnosis showed pi_hat puts ZERO true occupancy on the pairs that
wreck the bound: it assigns small positive action probability in states it never
visits, those pairs get the full simplex, and the robust adversary routes
probability into them. The pairs contribute nothing to the true cost and
everything to the bound.

The standard offline-RL response is a SUPPORT-CONSTRAINED policy: never put
probability on an action the log cannot vouch for. `certified_at_scale.py`
already uses this idea as a fallback (pi0 = always-intervene, certified cost 0).

INDEPENDENCE.  The projection uses D_TRAIN's support only, never D_check's, so
the projected policy remains independent of the held-out data and the
certificate stays valid for it. Projecting onto D_check's support would make
the policy a function of the data certifying it -- that is the selection bias
this whole experiment exists to avoid, and it is NOT done here.

Two projections, both computed from D_train:
  proj_renorm   renormalise pi_hat over D_train-supported actions in each state
  proj_fallback move all unsupported mass to the lowest-cost supported action
"""
from __future__ import annotations

import json
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import numpy as np

from .control_suite import _eligible, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool
from .safe_face_offline import (BUDGET, SEL, Log, behaviour_policy, sample_log,
                                score, check_ship, N_EV, DELTA_EV)
from .offline_check_run import learn_all, N_CHECK_GRID
from .offline_check import HeldOut

_OUT = Path(__file__).resolve().parents[3] / "results" / "offline_check_experiment"
SEEDS = tuple(range(10))
N_TRAIN = 20000


def project(pi: np.ndarray, support: np.ndarray, C: np.ndarray,
            mode: str) -> np.ndarray:
    """Move probability off D_train-unsupported actions. Never sees D_check."""
    out = np.array(pi, dtype=float)
    worst = C.max(axis=0)                      # worst-case per-(s,a) cost
    for s in range(pi.shape[0]):
        sup = support[s]
        if not sup.any():
            continue
        lost = out[s, ~sup].sum()
        if lost <= 0:
            continue
        out[s, ~sup] = 0.0
        if mode == "renorm" and out[s, sup].sum() > 1e-12:
            out[s] = out[s] / out[s].sum()
        else:                                   # fallback: cheapest supported
            a = int(np.argmin(np.where(sup, worst[s], np.inf)))
            out[s, a] += lost
        out[s] = out[s] / out[s].sum()
    return out


def main():
    sel = json.load(open(SEL))
    mid = [r for r in sel["rows"] if r["regime"] == "optimizer_resolvable"
           and abs(r["budget"] - BUDGET) < 1e-12]
    pt, _ = parse_prometheus(); bank = prom_threshold_bank(pt)
    comp = {}
    for ti, t in enumerate(pt):
        ok, _w, rd = _eligible(build_prom_pool(t, bank))
        if ok:
            comp[f'{getattr(t, "name", "?")}#{ti}'] = compile_instance(rd)

    rows = []
    for row in mid:
        m = comp[row["rule_id"]]
        P, mu0, r, C = m["P"], m["mu0"], m["r"], m["C"]
        pi_b = behaviour_policy(m, row["anchor"], BUDGET)
        for seed in SEEDS:
            rng_tr = np.random.default_rng(
                abs(hash((row["rule_id"], "train", seed))) % (2 ** 32))
            rng_ck = np.random.default_rng(
                abs(hash((row["rule_id"], "check", seed, 20260909))) % (2 ** 32))
            lg = Log(sample_log(m, pi_b, N_TRAIN, rng_tr))
            hos = {n: HeldOut(sample_log(m, pi_b, n, rng_ck)) for n in N_CHECK_GRID}
            pols = learn_all(lg, row["anchor"], BUDGET)
            crng = np.random.default_rng(
                abs(hash((row["rule_id"], "ev", seed))) % (2 ** 32))
            for name, pol in pols.items():
                for mode in ("raw", "renorm", "fallback"):
                    p = pol if mode == "raw" else project(pol, lg.support, C, mode)
                    tr = score(P, mu0, r, C, p, BUDGET)
                    rec = dict(rule_id=row["rule_id"], seed=seed, arm=name,
                               mode=mode, true_safe=bool(tr["safe"]),
                               J_r=float(tr["J_r"]), V_U=float(row["V_U"]),
                               cur_ship=bool(check_ship(P, mu0, C, p, BUDGET,
                                                        N_EV, crng, DELTA_EV)["ship"]))
                    for n, ho in hos.items():
                        oc = ho.check(p, C, BUDGET, DELTA_EV)
                        rec[f"off_ship_{n}"] = oc["ship"]
                        rec[f"off_bound_{n}"] = oc["bound"]
                        rec[f"frac_unsup_{n}"] = oc["frac_unsupported"]
                    rows.append(rec)
        print(f"  {row['rule_id']:38s} done", flush=True)

    out = {"n_rows": len(rows), "budget": BUDGET, "seeds": len(SEEDS)}
    for mode in ("raw", "renorm", "fallback"):
        rs = [x for x in rows if x["mode"] == mode]
        e = dict(n=len(rs),
                 true_safe_rate=float(np.mean([x["true_safe"] for x in rs])),
                 return_frac_med=float(np.nanmedian(
                     [x["J_r"] / x["V_U"] if x["V_U"] > 0 else np.nan for x in rs])),
                 current_check_ship=float(np.mean([x["cur_ship"] for x in rs])))
        for n in N_CHECK_GRID:
            sh = np.array([x[f"off_ship_{n}"] for x in rs], dtype=bool)
            tr = np.array([x["true_safe"] for x in rs], dtype=bool)
            e[f"offline_ship_{n}"] = float(sh.mean())
            e[f"offline_unsafe_ships_{n}"] = int((sh & ~tr).sum())
            e[f"median_bound_{n}"] = float(np.median([x[f"off_bound_{n}"] for x in rs]))
            e[f"median_frac_unsup_{n}"] = float(np.median([x[f"frac_unsup_{n}"] for x in rs]))
        out[mode] = e
    (_OUT / "support_constrained.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
