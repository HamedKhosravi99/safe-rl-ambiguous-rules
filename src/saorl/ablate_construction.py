"""Set-construction sensitivity ablation (reviewer addition #6).

Reviewers want to know whether U_alpha -- and therefore the kill-test gap -- is an
artifact of the frozen protocol constants (the soft band alpha, the plausibility
gate tau) or of the LM-ensemble size. This sweeps each knob ONE AT A TIME around its
frozen value and reports two cheap, learning-free summaries per setting:

  * |U_alpha|        -- the retained set size (does the construction stay stable?);
  * semantic_gap     -- worst-case cost honoring only the single most-plausible
                        reading minus honoring all of U_alpha, over active states
                        (the policy-level signature of ambiguity the kill test
                        amplifies; >0 exactly when retained readings disagree).

Neither needs the offline learner, so the whole grid runs in seconds; the gap here
is the data-level driver of the learned kill-test gap in saorl.experiments. The
ensemble-size axis truncates the cached per-persona plausibility tuple to its first
n entries (the judges are frozen, so this is a faithful "smaller ensemble" probe and
needs no re-querying). alpha/tau default to the frozen ALPHA/TAU_PLAUS.

Run:  PYTHONPATH=. python3 -m saorl.ablate_construction --seeds 20
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .construct import ALPHA, TAU_PLAUS, construct_U_alpha
from .experiments import _cons_cands, _synthetic_seed
from .offline import semantic_gap

ALPHA_GRID = (0.05, 0.10, 0.15, 0.20, 0.30)
TAU_GRID = (0.40, 0.50, 0.60, 0.70)
ENS_GRID = (1, 3, 5, 7)


def _truncate_ensemble(pool, n: int):
    """Faithful 'smaller frozen ensemble' probe: keep the first n persona scores."""
    return [dataclasses.replace(c, plausibility=tuple(c.plausibility[:n]))
            if getattr(c, "plausibility", ()) else c
            for c in pool]


@dataclass
class AblRow:
    axis: str           # "alpha" | "tau" | "ensemble"
    value: float
    seed: int
    u_size: int
    sem_gap: float


def _one(pool, data, *, alpha=ALPHA, tau=TAU_PLAUS):
    dsem = data.to_semantic_dataset()
    U = construct_U_alpha(pool, dsem, alpha=alpha, tau_plaus=tau).U_alpha
    return len(U), float(semantic_gap(data, U, normalize="active"))


def collect(seeds: Sequence[int]) -> List[AblRow]:
    rows: List[AblRow] = []
    for seed in seeds:
        data, _, _, _, pool = _synthetic_seed(seed)
        for a in ALPHA_GRID:                       # vary alpha, freeze tau + full ensemble
            u, g = _one(pool, data, alpha=a)
            rows.append(AblRow("alpha", a, seed, u, g))
        for t in TAU_GRID:                         # vary tau, freeze alpha + full ensemble
            u, g = _one(pool, data, tau=t)
            rows.append(AblRow("tau", t, seed, u, g))
        for n in ENS_GRID:                         # vary ensemble size, freeze alpha + tau
            u, g = _one(_truncate_ensemble(pool, n), data)
            rows.append(AblRow("ensemble", float(n), seed, u, g))
        print(f"  seed={seed} swept "
              f"{len(ALPHA_GRID)+len(TAU_GRID)+len(ENS_GRID)} settings")
    return rows


def _ci95(xs):
    a = np.asarray(xs, float)
    n = len(a)
    if n == 0:
        return 0.0, 0.0
    m = float(a.mean())
    if n < 2:
        return m, 0.0
    from scipy import stats
    return m, float(stats.t.ppf(0.975, n - 1) * a.std(ddof=1) / np.sqrt(n))


def summarize(rows: List[AblRow]) -> dict:
    out: dict = {}
    for axis, grid in (("alpha", ALPHA_GRID), ("tau", TAU_GRID), ("ensemble", ENS_GRID)):
        print(f"\n  === sweep {axis} (frozen = "
              f"{ALPHA if axis=='alpha' else TAU_PLAUS if axis=='tau' else 7}) ===")
        print(f"  {axis:>8s} {'|U_alpha|':>14s} {'semantic_gap':>18s}")
        out[axis] = {}
        for v in grid:
            sub = [r for r in rows if r.axis == axis and r.value == float(v)]
            um, uh = _ci95([r.u_size for r in sub])
            gm, gh = _ci95([r.sem_gap for r in sub])
            star = "  <- frozen" if (
                (axis == "alpha" and v == ALPHA) or
                (axis == "tau" and v == TAU_PLAUS) or
                (axis == "ensemble" and v == 7)) else ""
            print(f"  {v:8.2f} {um:6.2f}+/-{uh:4.2f}   {gm:8.3f}+/-{gh:6.3f}{star}")
            out[axis][str(v)] = dict(u_size=(um, uh), sem_gap=(gm, gh), n=len(sub))
    return out


def main():
    ap = argparse.ArgumentParser(description="set-construction sensitivity ablation (#6)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-list", type=str, default=None)
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()
    seeds = ([int(s) for s in args.seed_list.split(",") if s.strip()]
             if args.seed_list else list(range(args.seeds)))
    print(f"construction ablation: seeds={seeds} (synthetic maintenance)")
    t0 = time.time()
    rows = collect(seeds)
    summary = summarize(rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"ablate_construction_{stamp}.json"
    out_path.write_text(json.dumps(dict(
        config=dict(seeds=seeds, alpha_grid=ALPHA_GRID, tau_grid=TAU_GRID,
                    ens_grid=ENS_GRID, frozen=dict(alpha=ALPHA, tau=TAU_PLAUS, ens=7)),
        rows=[dataclasses.asdict(r) for r in rows], summary=summary,
    ), indent=2))
    print(f"\n  wrote {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
