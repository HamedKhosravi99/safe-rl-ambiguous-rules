"""E3 stage 1: the budget sweep (REGISTRATION_V10 addendum).

A thin wrapper around the FROZEN experiments/dsrl_learners/dsrl_vector.py. It rebinds the cost
limit -- the only thing the sweep varies -- and trains just the two arms the
endpoints need (`vector` and the `single_a` reference), leaving the driver's
data pipeline, witness construction, certification and evaluation untouched.

The frozen driver is not edited; nothing here can change the budget-10 leg.

Run:  python experiments/dsrl_learners/dsrl_e3.py --cell 0 --seed 0 --cost_limit 20 --out DIR
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cost_limit", type=float, required=True)
    ap.add_argument("--update_steps", type=int, default=100000)
    ap.add_argument("--eval_episodes", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import dsrl_nondominated as nd
    import dsrl_vector as dv

    # the sweep's only degree of freedom; rebound in BOTH modules because the
    # certificate reads dsrl_vector.COST_LIMIT and the config builder reads nd's
    nd.COST_LIMIT = args.cost_limit
    dv.COST_LIMIT = args.cost_limit
    # endpoints need the defended arm and its single-reading reference only
    dv.ARMS = ("single_a", "vector")

    sys.argv = [
        "dsrl_vector.py",
        "--cell", str(args.cell),
        "--seed", str(args.seed),
        "--update_steps", str(args.update_steps),
        "--eval_episodes", str(args.eval_episodes),
        "--device", args.device,
        "--num_workers", str(args.num_workers),
        "--out", args.out,
    ]
    print(f"[E3] cell={args.cell} seed={args.seed} "
          f"cost_limit={args.cost_limit} arms={dv.ARMS}", flush=True)
    dv.main()


if __name__ == "__main__":
    main()
