"""Held-out Offline-CHECK experiment (spec sections 6-8, 12).

ADDITIVE: imports safe_face_offline, never modifies it, never overwrites
results/e2e/safe_face_offline.json.  Writes only under
results/offline_check_experiment/.

PROTOCOL (this is the point of the experiment).  For each instance and each
seed the two logs are drawn as two INDEPENDENT calls to sample_log from the
same frozen behaviour policy, BOTH BEFORE ANY TRAINING:

    D_train (n_train transitions)  ->  learners  ->  frozen pi_hat
    D_check (n_check transitions)  ->  never seen by any learner

so D_check is genuinely offline: it exists before pi_hat and is not resampled
after seeing it.  n_train is held fixed while n_check is swept, so the sweep
does not confound data size with policy quality.

Three verdicts are recorded per policy:
    truth        oracle safety in the TRUE model (experimental scoring only;
                 NEVER visible to the offline certificate)
    offline      held-out robust bound <= d           (the proposal)
    current      check_ship on FRESH true-model draws (the incumbent; these
                 samples are generated after the policy exists and are NOT
                 offline -- they are the simulator baseline being replaced)
    plugin       held-out empirical model, no uncertainty term (unsafe control)
"""
from __future__ import annotations

import json
import os

# scipy's linprog calls here are tiny; default BLAS threading costs a measured
# 75x (90s vs 1.2s per seed) in thread-sync overhead. Pin before numpy loads.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np

from .control_suite import _eligible, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool
from .safe_face_offline import (
    BUDGET, EPS_GRID, KAPPA_FRAC, DELTA_EV, N_EV, SEL,
    Log, behaviour_policy, sample_log, score, check_ship,
    learn_single, learn_fullset, learn_safeface,
)
from .offline_check import HeldOut

_OUT = Path(__file__).resolve().parents[3] / "results" / "offline_check_experiment"

N_TRAIN = 20000                       # fixed while n_check is swept
N_CHECK_GRID = (2000, 5000, 20000, 50000, 200000)
SEEDS = tuple(range(20))
SEED_BASE = 20260909


def learn_all(lg: Log, anchor: int, d: float) -> dict:
    """Every registered learner arm, trained on D_train only."""
    pol = {"single": learn_single(lg, anchor, d),
           "fullset": learn_fullset(lg, anchor, d)}
    for e in EPS_GRID:
        pol[f"safeface_eps{e}"] = learn_safeface(lg, anchor, d, e)[0]
    for kf in KAPPA_FRAC:
        pol[f"margin_k{kf}"] = learn_safeface(lg, anchor, d, EPS_GRID[-1],
                                              kappa=kf * d)[0]
    return pol


def main() -> None:
    with open(SEL, encoding="utf8") as fh:
        sel = json.load(fh)
    mid = [r for r in sel["rows"]
           if r["regime"] == "optimizer_resolvable"
           and abs(r["budget"] - BUDGET) < 1e-12]
    print(f"{len(mid)} optimizer-resolvable instances at d={BUDGET}", flush=True)

    pt, _ = parse_prometheus()
    bank = prom_threshold_bank(pt)
    compiled = {}
    for ti, t in enumerate(pt):
        ok, _w, readings = _eligible(build_prom_pool(t, bank))
        if ok:
            compiled[f'{getattr(t, "name", "?")}#{ti}'] = compile_instance(readings)

    rows = []
    for row in mid:
        m = compiled[row["rule_id"]]
        P, mu0, r, C = m["P"], m["mu0"], m["r"], m["C"]
        anchor = row["anchor"]
        pi_b = behaviour_policy(m, anchor, BUDGET)

        for seed in SEEDS:
            # --- both logs drawn BEFORE training, independent streams ------
            rng_tr = np.random.default_rng(
                abs(hash((row["rule_id"], "train", seed))) % (2 ** 32))
            rng_ck = np.random.default_rng(
                abs(hash((row["rule_id"], "check", seed, SEED_BASE))) % (2 ** 32))
            lg_train = sample_log(m, pi_b, N_TRAIN, rng_tr)
            checks = {n: HeldOut(sample_log(m, pi_b, n, rng_ck))
                      for n in N_CHECK_GRID}

            # --- train on D_train only ------------------------------------
            policies = learn_all(Log(lg_train), anchor, BUDGET)

            # --- verdicts -------------------------------------------------
            crng = np.random.default_rng(
                abs(hash((row["rule_id"], "ev", seed))) % (2 ** 32))
            for name, pol in policies.items():
                truth = score(P, mu0, r, C, pol, BUDGET)
                cur = check_ship(P, mu0, C, pol, BUDGET, N_EV, crng, DELTA_EV)
                rec = dict(rule_id=row["rule_id"], seed=seed, arm=name,
                           n_train=N_TRAIN,
                           true_safe=bool(truth["safe"]),
                           true_cmax=float(truth["C_max"]),
                           J_r=float(truth["J_r"]), V_U=float(row["V_U"]),
                           cur_bound=float(cur["bound"]),
                           cur_ship=bool(cur["ship"]))
                for n, ho in checks.items():
                    oc = ho.check(pol, C, BUDGET, DELTA_EV)
                    rec[f"off_bound_{n}"] = oc["bound"]
                    rec[f"off_ship_{n}"] = oc["ship"]
                    rec[f"plug_bound_{n}"] = oc["plugin"]
                    rec[f"plug_ship_{n}"] = oc["plugin_ship"]
                    rec[f"frac_unsup_{n}"] = oc["frac_unsupported"]
                    rec[f"medrad_{n}"] = oc["median_radius"]
                rows.append(rec)
        _OUT.mkdir(parents=True, exist_ok=True)
        (_OUT / "raw_runs_partial.json").write_text(json.dumps(
            dict(done=len(set(x["rule_id"] for x in rows)), rows=rows)))
        print(f"  {row['rule_id']:38s} done  ({len(rows)} rows so far)", flush=True)

    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "raw_runs.json").write_text(json.dumps(
        dict(protocol="independent D_train/D_check, both drawn before training",
             n_train=N_TRAIN, n_check_grid=list(N_CHECK_GRID),
             seeds=list(SEEDS), budget=BUDGET, delta_ev=DELTA_EV,
             rows=rows), indent=1))
    print(f"\n{len(rows)} rows -> {_OUT/'raw_runs.json'}")


if __name__ == "__main__":
    main()
