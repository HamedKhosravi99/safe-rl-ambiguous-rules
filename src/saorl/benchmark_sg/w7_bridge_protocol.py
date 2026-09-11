"""W7 bridge experiment (REGISTRATION_W7): neural set defense on the compiled
source-grounded instances -- protocol registration + the offline Stage-0b gate.

A9 (the registered non-nested neural campaign) failed its pre-declared
criterion, with two diagnosed mechanisms: (1) wherever a singleton cannot
satisfy its own reading, the set arm inherits that failure (competence, not
set defense); (2) channel sparsity -- the logged data carries too little
occupancy in the disputed region for any learner to see the distinction.
The bridge experiment re-tests the neural claim where a certified ground
truth exists: the compiled source-grounded instances (non-nested by
construction; exact LP price known on every instance).

REGISTERED PROTOCOL (frozen here before any training):

  Data. Offline dataset per (instance, seed): N_TRAJ trajectories of length
  H_TRAJ from the uniform-random behavior policy on the compiled MDP,
  seeds 0..4, generated once and hashed.

  Stage 0 -- competence gate (target-blind). Train the single-reading arm
  on psi_1 alone; admit the cell only if it certifies at d on its OWN
  reading in >= 4/5 seeds (per-seed exact Clopper-Pearson at n=300
  episodes, upper bound <= d).

  Stage 0b -- channel-density gate (offline, computed BEFORE any training;
  this module). For each instance, the exact discounted occupancy mass of
  the uniform behavior policy in the disputed region
  D = {(s,a): some pair of maximal readings disagrees on c(s,a)}.
  Admit only instances with disputed mass >= RHO_MIN. This is exactly
  computable from the compiled model -- no sampling.

  Stage 1 -- on admitted cells only. (a) hidden gap under the single arm
  (worst cost over U minus honored cost); (b) worst-over-U under the set
  arm; (c) per-seed Clopper-Pearson certificates, never pooled; (d) the
  learned-vs-certified decomposition against the instance's exact LP price.

  Endpoints and shortfall rule. PRIMARY: on admitted cells, the set arm's
  worst-over-U certified rate at d in >= 60% of cells, AND median hidden
  gap reduction >= 50%. If the primary fails, the paper reports the
  failure and ships the removal framing unchanged (already written); no
  post-hoc cell exclusion beyond the two registered gates.

Run:  PYTHONPATH=. python3 -m saorl.benchmark_sg.w7_bridge_protocol
Writes results/e2e/w7_registration.json (refuses to alter an existing one)
and results/e2e/w7_stage0b.json
"""
from __future__ import annotations

import json
import os
from typing import List

import numpy as np

from .control_suite import compile_instance, GAMMA

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SUITE = os.path.join(ROOT, "results/conformal", "benchmark_sg", "control_suite.json")
REG_OUT = os.path.join(ROOT, "results/e2e", "w7_registration.json")
OUT = os.path.join(ROOT, "results/e2e", "w7_stage0b.json")

REGISTRATION = {
    "id": "REGISTRATION_W7",
    "instances": "the compiled source-grounded suite (control_suite.json)",
    "behavior_policy": "uniform random over the 2 actions",
    "n_traj": 400, "h_traj": 60, "seeds": [0, 1, 2, 3, 4],
    "budget_d": 0.05,
    "stage0_rule": "single arm certifies its own reading at d in >= 4/5 seeds "
                   "(per-seed exact Clopper-Pearson, n=300 episodes)",
    "stage0b_rule": "exact uniform-policy discounted occupancy in the disputed "
                    "region >= rho_min",
    "rho_min": 0.01,
    "stage1_endpoints": [
        "set arm certifies worst-over-U at d in >= 60% of admitted cells",
        "median hidden-gap reduction >= 50% on admitted cells",
    ],
    "shortfall_rule": "on failure, report and ship the removal framing; no "
                      "post-hoc exclusions beyond the two registered gates",
}


def occupancy_fixed_policy(m: dict, pi: np.ndarray) -> np.ndarray:
    """Exact discounted state-action occupancy of a fixed policy:
    x(s,a) = pi(a|s) * nu(s), nu = (1-gamma) mu0 + gamma P_pi^T nu."""
    nS, nA = m["nS"], m["nA"]
    Ppi = np.einsum("san,sa->sn", m["P"], pi)          # (s -> s') under pi
    nu = np.linalg.solve(np.eye(nS) - GAMMA * Ppi.T, (1.0 - GAMMA) * m["mu0"])
    return pi * nu[:, None]                            # (nS, nA)


def disputed_mask(m: dict) -> np.ndarray:
    """(nS, nA) True where some pair of readings disagrees on the cost."""
    C = m["C"]
    return (C.max(axis=0) - C.min(axis=0)) > 1e-12


def main() -> None:
    os.makedirs(os.path.dirname(REG_OUT), exist_ok=True)
    if os.path.exists(REG_OUT):
        if json.load(open(REG_OUT)) != REGISTRATION:
            raise SystemExit("w7_registration.json exists with different content")
    else:
        json.dump(REGISTRATION, open(REG_OUT, "w"), indent=1)
    print("[w7] registration frozen at", REG_OUT)

    suite = json.load(open(SUITE))
    rows: List[dict] = []
    for inst in suite["instances"]:
        class R:                                        # reading shim
            pass
        readings = []
        for rd in inst["readings"]:
            r = R()
            r.threshold, r.for_s = rd["theta"], rd["for_s"]
            readings.append(r)
        m = compile_instance(readings)
        pi = np.full((m["nS"], m["nA"]), 0.5)
        occ = occupancy_fixed_policy(m, pi)
        mask = disputed_mask(m)
        rho = float(occ[mask].sum() / occ.sum())
        rows.append(dict(uid=inst["uid"], geometry=inst["geometry"],
                         n_states=m["nS"], disputed_mass=rho,
                         admitted=bool(rho >= REGISTRATION["rho_min"])))
        print(f"  {inst['uid']:34s} {inst['geometry']:12s} rho={rho:.4f} "
              f"{'ADMIT' if rho >= REGISTRATION['rho_min'] else 'drop'}")
    n_admit = sum(r["admitted"] for r in rows)
    out = dict(registration="REGISTRATION_W7", rho_min=REGISTRATION["rho_min"],
               n_instances=len(rows), n_admitted=n_admit, rows=rows)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"[w7] stage 0b: {n_admit}/{len(rows)} instances admitted; wrote {OUT}")


if __name__ == "__main__":
    main()
