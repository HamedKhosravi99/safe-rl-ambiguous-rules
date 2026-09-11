"""Coverage-failure anatomy (spec section 10) + positive control.

Two questions the headline table cannot answer:

 Q1 STRUCTURAL OR STATISTICAL?  Is the held-out log missing the learned
    policy's (s,a) pairs because it is too small, or because the behaviour
    policy assigns them probability ~0?  Computes the TRUE behaviour occupancy
    of the pairs pi_hat uses (true model, experimental scoring only) and the
    occupancy mass pi_hat puts on pairs with zero behaviour support.

 Q2 IS THE EVALUATOR ITSELF USABLE?  Runs Offline-CHECK on the BEHAVIOUR
    policy, which the log covers by construction, and on the true-safe subset.
    If the certificate ships a covered safe policy, the machinery works and the
    negative headline is a coverage result, not a broken bound.
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
                                score, occupancy, N_MIN)
from .offline_check_run import learn_all, N_CHECK_GRID
from .offline_check import HeldOut, M_SUPP

_OUT = Path(__file__).resolve().parents[3] / "results" / "offline_check_experiment"
SEEDS = tuple(range(5))
N_TRAIN = 20000


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

    rows, ctrl = [], []
    for row in mid:
        m = comp[row["rule_id"]]
        P, mu0, r, C = m["P"], m["mu0"], m["r"], m["C"]
        pi_b = behaviour_policy(m, row["anchor"], BUDGET)
        xb = occupancy(P, mu0, pi_b)                 # TRUE behaviour occupancy
        for seed in SEEDS:
            rng_tr = np.random.default_rng(
                abs(hash((row["rule_id"], "train", seed))) % (2 ** 32))
            rng_ck = np.random.default_rng(
                abs(hash((row["rule_id"], "check", seed, 20260909))) % (2 ** 32))
            lg = sample_log(m, pi_b, N_TRAIN, rng_tr)
            hos = {n: HeldOut(sample_log(m, pi_b, n, rng_ck)) for n in N_CHECK_GRID}
            pols = learn_all(Log(lg), row["anchor"], BUDGET)

            # ---- Q2 positive control: the behaviour policy itself ----------
            tb = score(P, mu0, r, C, pi_b, BUDGET)
            for n, ho in hos.items():
                oc = ho.check(pi_b, C, BUDGET, 0.05)
                ctrl.append(dict(rule_id=row["rule_id"], seed=seed, n=n,
                                 policy="behaviour", true_safe=bool(tb["safe"]),
                                 bound=oc["bound"], ship=oc["ship"],
                                 frac_unsup=oc["frac_unsupported"]))

            # ---- Q1 anatomy for each learned policy -----------------------
            for name, pol in pols.items():
                used = pol > 1e-12
                xpi = occupancy(P, mu0, pol)
                zero_b = xb <= 1e-12                 # zero TRUE behaviour mass
                rows.append(dict(
                    rule_id=row["rule_id"], seed=seed, arm=name,
                    true_safe=bool(score(P, mu0, r, C, pol, BUDGET)["safe"]),
                    # behaviour occupancy of the least-visited pair pi_hat uses
                    min_behav_occ_used=float(xb[used].min()),
                    med_behav_occ_used=float(np.median(xb[used])),
                    # how much of pi_hat's own occupancy sits on zero-behaviour pairs
                    pihat_mass_on_zero_behav=float(xpi[used & zero_b].sum()),
                    frac_used_zero_behav=float((used & zero_b).sum() / used.sum()),
                    n_used=int(used.sum()),
                    frac_unsup_200k=hos[200000].check(pol, C, BUDGET, 0.05)["frac_unsupported"],
                ))
        print(f"  {row['rule_id']:38s} done", flush=True)

    out = {}
    # Q1
    mb = np.array([r["min_behav_occ_used"] for r in rows])
    fz = np.array([r["frac_used_zero_behav"] for r in rows])
    mass = np.array([r["pihat_mass_on_zero_behav"] for r in rows])
    fu = np.array([r["frac_unsup_200k"] for r in rows])
    out["Q1_structural_vs_statistical"] = dict(
        n=len(rows),
        median_frac_used_pairs_with_ZERO_behaviour_mass=float(np.median(fz)),
        mean_frac_used_pairs_with_ZERO_behaviour_mass=float(np.mean(fz)),
        median_pihat_occupancy_mass_on_zero_behaviour_pairs=float(np.median(mass)),
        median_min_behaviour_occupancy_of_a_used_pair=float(np.median(mb)),
        median_frac_unsupported_at_200k=float(np.median(fu)),
        # a pair needs behaviour occupancy > M_SUPP/n to clear the threshold
        threshold_at_200k=M_SUPP / 200000,
        note=("If frac_used_pairs_with_ZERO_behaviour_mass ~ frac_unsupported_at_200k "
              "the failure is STRUCTURAL: those pairs have no behaviour mass at all, "
              "so no amount of held-out data supports them."))
    # Q2
    cs = {}
    for n in N_CHECK_GRID:
        sub = [c for c in ctrl if c["n"] == n]
        safe = [c for c in sub if c["true_safe"]]
        cs[str(n)] = dict(
            n_runs=len(sub),
            behaviour_true_safe_rate=float(np.mean([c["true_safe"] for c in sub])),
            behaviour_ship_rate=float(np.mean([c["ship"] for c in sub])),
            behaviour_ship_rate_among_truly_safe=(
                float(np.mean([c["ship"] for c in safe])) if safe else None),
            median_bound=float(np.median([c["bound"] for c in sub])),
            median_frac_unsup=float(np.median([c["frac_unsup"] for c in sub])),
            unsafe_ships=int(sum(1 for c in sub if c["ship"] and not c["true_safe"])))
    out["Q2_positive_control_behaviour_policy"] = cs
    out["budget"] = BUDGET
    (_OUT / "coverage_diagnosis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
