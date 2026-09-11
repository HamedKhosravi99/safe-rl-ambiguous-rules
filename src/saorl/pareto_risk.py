"""Risk/budget Pareto for U_alpha (reviewer addition #8).

The kill test reports one operating point: budget d-bar = eps = 0.05 and CVaR tail
level beta = 0.1. A reviewer rightly asks whether SA-ORL's dominance is an artifact
of that single choice. This sweeps BOTH axes, reusing the exact experiments.run()
pipeline (single = honor the most-plausible reading; SA-ORL = honor all of
U_alpha) so nothing but the swept knob changes:

  * budget sweep (d-bar = eps): re-learn single and SA-ORL at each constraint
    budget eps. Tighter eps forces a more conservative policy; looser eps lets both
    earn more return. This traces the realized return / tail-risk Pareto frontier.
    SA-ORL should dominate -- lower realized violation chance and CVaR at matched or
    better return -- at EVERY budget, not just at the headline 0.05.

  * CVaR tail-level sweep (beta): re-score the policies (learned at the headline
    eps) at several tail levels beta. By construction (offline.rollout_risk fixes
    the rollout seed and FQI is deterministic per seed) the realized chance and
    mean-worst are beta-invariant and ONLY CVaR_beta moves, so this isolates the
    tail-level axis exactly. SA-ORL keeps CVaR_beta below the single reading across
    the whole range.

Both sweeps are learning-based but fully local and reproducible (no LM, no cluster).
Emits a JSON, two console tables, and a Pareto figure (paper/figs/pareto_risk.pdf).

Run:  PYTHONPATH=. python3 -m saorl.pareto_risk --seeds 15 --domains real,synthetic
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .experiments import EPS, Record, _ci95, run

EPS_GRID = (0.02, 0.05, 0.10, 0.20, 0.50)
BETA_GRID = (0.05, 0.10, 0.25, 0.50)
HEAD_EPS = 0.05
HEAD_BETA = 0.10
LABEL = {"real": "C-MAPSS (real)", "synthetic": "Maint.\\ (synthetic)",
         "gridworld": "Gridworld"}


def _run_quiet(seeds, domains, eps, beta) -> List[Record]:
    """run() with its verbose per-seed prints suppressed."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        recs = run(seeds, domains, ["fqi"], eps=eps, beta=beta)
    return recs


def _agg(records: List[Record], domain: str, model: str) -> dict:
    rs = [r for r in records if r.domain == domain and r.model == model]
    if not rs:
        return {}
    out = {}
    for k in ("ret", "true_worst", "chance", "cvar"):
        vals = [getattr(r, k) for r in rs if getattr(r, k) is not None]
        m, h = _ci95(vals) if vals else (float("nan"), 0.0)
        out[k] = (m, h)
    out["n"] = len(rs)
    return out


def budget_sweep(seeds, domains) -> dict:
    """Re-learn single/SA-ORL at each budget eps; collect return + realized risk."""
    res: Dict[str, list] = {d: [] for d in domains}
    for eps in EPS_GRID:
        recs = _run_quiet(seeds, domains, eps=eps, beta=HEAD_BETA)
        for d in domains:
            row = dict(eps=eps,
                       single=_agg(recs, d, "fqi:single"),
                       saorl=_agg(recs, d, "fqi:saorl"))
            res[d].append(row)
            s, r = row["single"], row["saorl"]
            print(f"  [budget] {d:10s} eps={eps:.2f}: "
                  f"ret {s['ret'][0]:6.1f}->{r['ret'][0]:6.1f}  "
                  f"chance {s['chance'][0]:.3f}->{r['chance'][0]:.3f}  "
                  f"CVaR {s['cvar'][0]:.3f}->{r['cvar'][0]:.3f}  "
                  f"tw {s['true_worst'][0]:.3f}->{r['true_worst'][0]:.3f}")
    return res


def beta_sweep(seeds, domains) -> dict:
    """Re-score policies (learned at HEAD_EPS) at several CVaR tail levels beta."""
    res: Dict[str, list] = {d: [] for d in domains}
    for beta in BETA_GRID:
        recs = _run_quiet(seeds, domains, eps=HEAD_EPS, beta=beta)
        for d in domains:
            row = dict(beta=beta,
                       single=_agg(recs, d, "fqi:single"),
                       saorl=_agg(recs, d, "fqi:saorl"))
            res[d].append(row)
            s, r = row["single"], row["saorl"]
            print(f"  [tail]   {d:10s} beta={beta:.2f}: "
                  f"chance {s['chance'][0]:.3f}->{r['chance'][0]:.3f} (inv)  "
                  f"CVaR {s['cvar'][0]:.3f}->{r['cvar'][0]:.3f}")
    return res


def _figure(budget: dict, out_pdf: Path) -> None:
    """Return (x, higher better) vs realized tail risk CVaR_0.1 (y, lower better);
    one connected frontier per policy per domain, points annotated by budget eps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    domains = list(budget.keys())
    fig, axes = plt.subplots(1, len(domains), figsize=(5.2 * len(domains), 4.0),
                             squeeze=False)
    for ax, d in zip(axes[0], domains):
        rows = budget[d]
        for model, color, mk in (("single", "#c0392b", "o"),
                                  ("saorl", "#27ae60", "s")):
            xs = [row[model]["ret"][0] for row in rows]
            ys = [row[model]["cvar"][0] for row in rows]
            ax.plot(xs, ys, "-", color=color, marker=mk, ms=6,
                    label=("single reading" if model == "single" else "SA-ORL"))
            for row, x, y in zip(rows, xs, ys):
                ax.annotate(f"{row['eps']:.2f}", (x, y), fontsize=7,
                            textcoords="offset points", xytext=(4, 4), color=color)
        ax.set_title(LABEL.get(d, d).replace("\\", ""))
        ax.set_xlabel("realized return (higher better)")
        ax.set_ylabel(r"realized tail risk CVaR$_{0.1}$ (lower better)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("Return / tail-risk Pareto across constraint budgets "
                 r"$\bar d=\epsilon\in\{0.02,0.05,0.1,0.2,0.5\}$", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")


def main():
    ap = argparse.ArgumentParser(description="risk/budget Pareto for U_alpha (#8)")
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--domains", type=str, default="real,synthetic")
    ap.add_argument("--out", type=str, default="results/paper_extra/pareto")
    ap.add_argument("--fig", type=str, default="paper/figs/pareto_risk.pdf")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    domains = [d for d in args.domains.split(",") if d.strip()]
    print(f"#8 risk/budget Pareto: seeds={seeds} domains={domains}")
    print(f"  budget grid eps={EPS_GRID}; tail grid beta={BETA_GRID}")
    t0 = time.time()
    print("\n== budget sweep (d-bar = eps, beta=0.1) ==")
    budget = budget_sweep(seeds, domains)
    print("\n== CVaR tail-level sweep (beta, eps=0.05) ==")
    beta = beta_sweep(seeds, domains)
    _figure(budget, Path(args.fig))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"pareto_risk_{stamp}.json"
    out_path.write_text(json.dumps(dict(
        config=dict(seeds=seeds, domains=domains, eps_grid=list(EPS_GRID),
                    beta_grid=list(BETA_GRID), head_eps=HEAD_EPS, head_beta=HEAD_BETA),
        budget=budget, beta=beta), indent=2))
    print(f"\n  wrote {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
