"""Per-interpretation violation table (reviewer addition #9).

The headline kill-test number collapses the ambiguous set with a max: it reports
the single WORST interpretation's cost. A reviewer rightly wants to see the cost
broken out PER interpretation -- which specific readings does the single-
interpretation policy leave violated, and does honoring all of U_alpha drive every
one of them below budget? This module answers that directly.

For each domain x seed it learns the same two policies as the main benchmark
(single = honor the most-plausible reading; SA-ORL = honor all of U_alpha), then,
for every retained interpretation psi_k in U_alpha, evaluates that interpretation's
expected semantic cost J_{c_k} over the offline dataset's active states under BOTH
policies. The table makes the mechanism legible:

  * under SINGLE, some psi_k carry J_{c_k} > eps -- the readings the committed
    single interpretation silently violates;
  * under SA-ORL, every psi_k is driven to J_{c_k} <= eps (up to the learner's
    feasibility tolerance) -- the per-interpretation guarantee Theorem~\\ref{thm:feas}
    targets.

No new modelling: it reuses experiments._domain_specs / most_plausible and
offline.policy_costs, so the per-k costs are exactly the quantities the worst-case
max in the main table is taken over. Numbers therefore agree by construction with
the aggregate `true_worst` column.

Run:  PYTHONPATH=. python3 -m saorl.per_interp --seeds 10 --domains synthetic,real,gridworld
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .experiments import EPS, _domain_specs, _learner_kwargs, most_plausible
from .offline import active_steps, policy_costs


@dataclass
class InterpRow:
    domain: str
    seed: int
    k: int                 # index of the interpretation within U_alpha
    name: str              # candidate (rule) name
    plaus: float           # ensemble-mean plausibility of this reading
    active_frac: float     # fraction of dataset steps where this reading fires
    jck_single: float      # J_{c_k} of the single-interpretation policy (active-norm)
    jck_saorl: float       # J_{c_k} of the SA-ORL policy (active-norm)


def collect(domains: Sequence[str], seeds: Sequence[int],
            eps: float = EPS, device: str = "cpu") -> List[InterpRow]:
    specs = _domain_specs(domains, ["fqi"])
    rows: List[InterpRow] = []
    for dname, spec in specs.items():
        fn = spec.learners.get("fqi")
        if fn is None:
            continue
        for seed in seeds:
            data, env, U, ret_fn, _ = spec.build(seed)
            if not U:
                continue
            kw = _learner_kwargs("fqi", seed, device)
            single = fn(data, env, honor=most_plausible(U), U_eval=U, eps=eps,
                        return_fn=ret_fn, **kw)
            saorl = fn(data, env, honor=U, U_eval=U, eps=eps, return_fn=ret_fn, **kw)
            cs = policy_costs(single.policy, data, U, normalize="active")
            cr = policy_costs(saorl.policy, data, U, normalize="active")
            denom = max(1, active_steps(data, U))
            for k, c in enumerate(U):
                fires = sum(c.fires(traj, t)
                            for traj in data.trajectories for t in range(len(traj)))
                rows.append(InterpRow(
                    dname, seed, k, c.name,
                    round(float(c.plaus_mean()), 4),
                    round(fires / denom, 4),
                    round(float(cs[c.name]), 4),
                    round(float(cr[c.name]), 4)))
            print(f"  {dname} seed={seed}: |U|={len(U)}  "
                  f"single violates {sum(cs[c.name] > eps for c in U)}/{len(U)}, "
                  f"SA-ORL violates {sum(cr[c.name] > eps for c in U)}/{len(U)}")
    return rows


def summarize(rows: List[InterpRow], eps: float = EPS) -> dict:
    out: dict = {}
    for dom in sorted({r.domain for r in rows}):
        sub = [r for r in rows if r.domain == dom]
        n_cells = len(sub)
        viol_single = sum(r.jck_single > eps for r in sub)
        viol_saorl = sum(r.jck_saorl > eps for r in sub)
        # per-seed: how many interpretations does each policy leave violated?
        seeds = sorted({r.seed for r in sub})
        per_seed_single = [sum(r.jck_single > eps for r in sub if r.seed == s) for s in seeds]
        per_seed_saorl = [sum(r.jck_saorl > eps for r in sub if r.seed == s) for s in seeds]
        out[dom] = dict(
            n_interp_cells=n_cells, n_seeds=len(seeds),
            violated_single=viol_single, violated_saorl=viol_saorl,
            frac_violated_single=round(viol_single / max(1, n_cells), 4),
            frac_violated_saorl=round(viol_saorl / max(1, n_cells), 4),
            mean_violated_per_seed_single=round(float(np.mean(per_seed_single)), 3),
            mean_violated_per_seed_saorl=round(float(np.mean(per_seed_saorl)), 3),
            max_jck_single=round(max((r.jck_single for r in sub), default=0.0), 4),
            max_jck_saorl=round(max((r.jck_saorl for r in sub), default=0.0), 4))
        print(f"\n  === {dom} === ({len(seeds)} seeds, {n_cells} interpretation-cells)")
        print(f"    interpretations with J_ck > eps:  single = {viol_single}/{n_cells} "
              f"({out[dom]['frac_violated_single']:.1%}),  "
              f"SA-ORL = {viol_saorl}/{n_cells} ({out[dom]['frac_violated_saorl']:.1%})")
        print(f"    worst single J_ck = {out[dom]['max_jck_single']:.3f}; "
              f"worst SA-ORL J_ck = {out[dom]['max_jck_saorl']:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser(description="per-interpretation violation table (#9)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-list", type=str, default=None)
    ap.add_argument("--domains", type=str, default="synthetic,real,gridworld")
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    seeds = ([int(s) for s in args.seed_list.split(",") if s.strip()]
             if args.seed_list else list(range(args.seeds)))
    domains = [d for d in args.domains.split(",") if d]
    print(f"per-interpretation table: seeds={seeds} domains={domains} eps={args.eps}")
    t0 = time.time()
    rows = collect(domains, seeds, eps=args.eps, device=args.device)
    summary = summarize(rows, eps=args.eps)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"per_interp_{stamp}.json"
    out_path.write_text(json.dumps(dict(
        config=dict(seeds=seeds, domains=domains, eps=args.eps),
        rows=[asdict(r) for r in rows], summary=summary,
    ), indent=2))
    print(f"\n  wrote {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
