"""Decisive test: can a data-only occupancy bound be valid AND non-vacuous?

Three arms per frozen policy, all on the SAME held-out log:

  ORACLE-W  : uses the TRUE occupancy ratio + empirical-Bernstein UCB.
              NOT IMPLEMENTABLE. It is the ceiling on what any ratio-based
              method can achieve, because its ratio error is exactly zero.
              If this is vacuous, every data-only version is too.

  DICE-W    : ratio estimated from D_ratio, cost bounded on the independent
              D_eval (sample splitting, section 7). Implementable, but the
              Bernstein term treats the frozen w_hat as fixed, so it covers
              only the SAMPLING error, NOT the ratio-estimation bias. It is
              therefore NOT YET A VALID CERTIFICATE -- reported as such.

  BIAS      : the realised bias E_dmu[(w* - w_hat) c_phi], measured with the
              oracle. A valid certificate must add a bound on this term; we
              measure how large it actually is to see whether any such bound
              could leave the certificate non-vacuous.

Sample splitting: D_check -> first half D_ratio (estimate w, then FREEZE),
second half D_eval (weighted cost + UCB). w_hat never sees D_eval.
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
from .offline_check_run import learn_all
from .occupancy_check import sample_log_traj, true_occupancy, dice_ratio

_OUT = Path(__file__).resolve().parents[3] / "results" / "occupancy_check_experiment"
N_TRAIN = 20000
GRID = (2000, 5000, 20000, 50000, 200000)
SEEDS = tuple(range(10))


def eb_ucb(x: np.ndarray, hi: float, delta: float) -> float:
    """One-sided empirical-Bernstein upper confidence bound on E[X], X in [0,hi]."""
    n = len(x)
    if n == 0:
        return float("inf")
    L = np.log(3.0 / delta)
    return float(x.mean() + np.sqrt(2.0 * x.var(ddof=0) * L / n) + 3.0 * hi * L / n)


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
        K = C.shape[0]
        pi_b = behaviour_policy(m, row["anchor"], BUDGET)
        dmu_true = true_occupancy(P, mu0, pi_b)
        for seed in SEEDS:
            rng_tr = np.random.default_rng(abs(hash((row["rule_id"], "train", seed))) % (2**32))
            rng_ck = np.random.default_rng(abs(hash((row["rule_id"], "occ", seed))) % (2**32))
            pols = learn_all(Log(sample_log(m, pi_b, N_TRAIN, rng_tr)), row["anchor"], BUDGET)
            pols["behaviour"] = pi_b                      # positive control
            logs = {n: sample_log_traj(m, pi_b, n, rng_ck) for n in GRID}
            crng = np.random.default_rng(abs(hash((row["rule_id"], "ev", seed))) % (2**32))

            for name, pol in pols.items():
                tr = score(P, mu0, r, C, pol, BUDGET)
                dpi_true = true_occupancy(P, mu0, pol)
                w_true = np.where(dmu_true > 1e-12,
                                  dpi_true / np.maximum(dmu_true, 1e-300), 0.0)
                rec = dict(rule_id=row["rule_id"], seed=seed, arm=name,
                           true_safe=bool(tr["safe"]), true_cmax=float(tr["C_max"]),
                           cur_ship=bool(check_ship(P, mu0, C, pol, BUDGET, N_EV,
                                                    crng, DELTA_EV)["ship"]))
                for n, lg in logs.items():
                    h = n // 2
                    Lr = {k: (v[:h] if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n else v)
                          for k, v in lg.items()}
                    Se, Ae = lg["S"][h:], lg["A"][h:]
                    w_hat = dice_ratio(Lr, pol)["w"]
                    dp = DELTA_EV / K
                    bo, bd, bias = [], [], []
                    for k in range(K):
                        ck = C[k]
                        xo = w_true[Se, Ae] * ck[Se, Ae]
                        xd = w_hat[Se, Ae] * ck[Se, Ae]
                        bo.append(eb_ucb(xo, float(max(w_true.max(), 1e-12)), dp))
                        bd.append(eb_ucb(xd, float(max(w_hat.max(), 1e-12)), dp))
                        bias.append(float((dmu_true * (w_true - w_hat) * ck).sum()))
                    rec[f"oracle_bound_{n}"] = float(max(bo))
                    rec[f"oracle_ship_{n}"] = bool(max(bo) <= BUDGET)
                    rec[f"dice_bound_{n}"] = float(max(bd))
                    rec[f"dice_ship_{n}"] = bool(max(bd) <= BUDGET)
                    rec[f"bias_{n}"] = float(max(bias))
                    rec[f"wmax_hat_{n}"] = float(w_hat.max())
                rows.append(rec)
        print(f"  {row['rule_id']:38s} done", flush=True)

    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "raw.json").write_text(json.dumps(
        dict(grid=list(GRID), seeds=list(SEEDS), budget=BUDGET,
             delta_ev=DELTA_EV, n_train=N_TRAIN, rows=rows), indent=1))
    print(f"\n{len(rows)} rows -> {_OUT/'raw.json'}")


if __name__ == "__main__":
    main()
