"""Direct language->constraint baseline 1B: free-form LM cost inference (reviewer #1).

The second "just ask the model" baseline, complementary to 1C (code generation).
Here the LM never writes code and never touches the DSL: it directly JUDGES, for a
given asset condition, whether the rule forbids continuing to operate this cycle.
We elicit that judgment as a 0-1 violation score on a fixed grid of representative
conditions along the degradation axis, build the implied cost FIELD (nearest-grid
lookup), learn an offline policy that honors only it, and then -- exactly as every
other table -- score the policy's REALIZED safety against the audited ambiguity set
U_alpha.

This is the #1-family point in its most model-trusting form: the LM's free-form,
per-condition cost commits to ONE implicit stop-threshold. Wherever it places that
threshold, it is a single reading; if it lands more permissive than a retained
reading, the policy violates that reading on deployment, while SA-ORL -- honoring
the whole audited set -- does not. Free-form elicitation also has no built-in
consistency (no monotonicity guarantee across conditions), a second failure mode a
single structured reading does not have.

The grid is SEED-INDEPENDENT (it depends only on the rule + feature glossary), so
the LM is queried once for the whole study and the field is reused across all data
seeds: the entire live cost is `len(GRID)` calls.

SAFETY / BUDGET GATING. Identical to baseline_llm_codegen: defaults to --dry-run,
which prints the EXACT per-condition prompt, prints the precise `claude` call count
--live would spend, and exercises the FULL downstream pipeline on a HAND-WRITTEN
stand-in cost field (clearly NOT an LM output) so the offline-RL integration is
verified while nothing is spent. Pass --live to actually query the `claude` CLI.

Dry run (default; spends nothing):
    PYTHONPATH=. python3 -m saorl.baseline_llm_freeform
Live (spends API budget; needs the `claude` CLI):
    PYTHONPATH=. python3 -m saorl.baseline_llm_freeform --live --seeds 30
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from .baseline_llm_codegen import CodeRow, _summ, as_candidate
from .baselines import EPS, MAINT, N_SEEDS, _maint_rollout_once
from .build_plaus_cache import RULE_SETS
from .construct import construct_U_alpha
from .judge import FEATURE_GLOSSARY, LLMEnsembleJudge, attach_plausibility, load_llm_cache
from .offline import (evaluate_return, greedy_continue_policy,
                      make_offline_rl_dataset, rollout_risk, worst_case_cost)
from .offline_rl import learn_fqi_constrained

RULE_SET = "conservative"
GRID = tuple(range(2, 99, 6))   # rul levels probed along the degradation axis (17)
THRESH = 0.5                    # elicited score >= THRESH => the field forbids continue
ORDER = ("run-to-failure", "llm-freeform", "SA-ORL")   # summary row order


def grid_state(rul: float) -> dict:
    """Noise-free representative state at a given RUL, using the MaintenanceMDP
    generative means (so each probed condition is a coherent real asset state)."""
    rh = float(rul)
    return {"rul_hat": rh, "q05": max(0.0, rh - 6.0),
            "anom": float(np.clip(1.0 - rul / MAINT.max_life, 0.0, 1.0))}


def cost_prompt(rule_text: str, st: dict) -> str:
    """The exact per-condition elicitation prompt handed to the LM."""
    return (
        "You are a controls engineer enforcing a maintenance rule.\n\n"
        f"{FEATURE_GLOSSARY}\n\n"
        f"Natural-language maintenance rule:\n  \"{rule_text}\"\n\n"
        "Here is one asset's condition this cycle:\n"
        f"  rul_hat = {st['rul_hat']:.0f}, q05 = {st['q05']:.0f}, "
        f"anom = {st['anom']:.2f}\n\n"
        "According to the rule, must the asset STOP (i.e. NOT continue operating) "
        "this cycle? Reply with ONLY a single number between 0 and 1: 1.0 = the "
        "rule clearly forbids continuing now; 0.0 = continuing is clearly allowed; "
        "0.5 = borderline."
    )


def make_cost_field(grid_ruls: Sequence[float], scores: Sequence[Optional[float]],
                    thresh: float = THRESH) -> Callable[[dict], bool]:
    """Nearest-grid cost field: a state violates iff the elicited score at the
    nearest probed RUL meets `thresh`. Missing scores (None) count as 0 (allowed)."""
    arr_r = np.asarray(grid_ruls, dtype=float)
    arr_s = np.asarray([0.0 if s is None else float(s) for s in scores], dtype=float)

    def fn(s: dict) -> bool:
        i = int(np.argmin(np.abs(arr_r - s["rul_hat"])))
        return bool(arr_s[i] >= thresh)
    return fn


def standin_scores() -> List[float]:
    """Hand-written elicited field for --dry-run (NOT an LM output): a sane reading
    that forbids continuing once the asset is degraded (rul<=20 or anom>=0.6)."""
    out = []
    for r in GRID:
        st = grid_state(r)
        out.append(1.0 if (st["rul_hat"] <= 20 or st["anom"] >= 0.6) else 0.0)
    return out


def evaluate_field(fn: Callable[[dict], bool], seeds: Sequence[int],
                   rule_set: str = RULE_SET) -> List[CodeRow]:
    """Learn a policy honoring ONLY the elicited cost field, plus SA-ORL honoring
    the audited U_alpha, and score both on realized risk vs U_alpha across seeds."""
    raw = [c for c, _ in RULE_SETS[rule_set]["items"]]
    judge = load_llm_cache(rule_set)
    rollout_once = _maint_rollout_once(MAINT)
    field_c = as_candidate(fn, "llm-freeform")
    rows: List[CodeRow] = []
    for seed in seeds:
        data = make_offline_rl_dataset(MAINT, seed=seed)
        U = construct_U_alpha(attach_plausibility(raw, judge),
                              data.to_semantic_dataset()).U_alpha
        if not U:
            continue
        rf = lambda pol: evaluate_return(MAINT, pol, n_episodes=40)
        risk = lambda pol: rollout_risk(rollout_once, pol, U)
        field_res = learn_fqi_constrained(data, MAINT, honor=[field_c], U_eval=U,
                                          eps=EPS, return_fn=rf, bcq_tau=0.05)
        saorl = learn_fqi_constrained(data, MAINT, honor=U, U_eval=U, eps=EPS,
                                      return_fn=rf, bcq_tau=0.05)
        for name, pol in (("run-to-failure", greedy_continue_policy()),
                          ("llm-freeform", field_res.policy),  # reuse CodeRow schema
                          ("SA-ORL", saorl.policy)):
            r = risk(pol)
            rows.append(CodeRow(0, seed, len(U), name, rf(pol),
                                worst_case_cost(pol, data, U, normalize="active"),
                                r["chance"], r["cvar"]))
    return rows


def run(seeds: Sequence[int], live: bool) -> dict:
    rule_text = RULE_SETS[RULE_SET]["rule_text"]
    print(f"\n=== 1B free-form LM cost inference: rule_set='{RULE_SET}', "
          f"|grid|={len(GRID)} conditions = {len(GRID)} calls (seed-independent) ===")
    print(f"  rule: \"{rule_text}\"")
    if not live:
        st = grid_state(GRID[len(GRID) // 2])
        print("\n  [dry-run] sample per-condition prompt (mid-grid):\n")
        print("  " + cost_prompt(rule_text, st).replace("\n", "\n  "))
        print("\n  [dry-run] exercising the FULL pipeline on a HAND-WRITTEN stand-in "
              "cost field")
        print("  (clearly NOT an LM output -- proves integration, spends nothing):")
        scores = standin_scores()
        print("   grid rul -> stop?:  " + "  ".join(
            f"{r}:{int(sc)}" for r, sc in zip(GRID, scores)))
        fn = make_cost_field(GRID, scores)
        rows = evaluate_field(fn, list(seeds)[:3])
        _summ(rows, "STAND-IN (dry-run, 3 seeds)", order=ORDER)
        return dict(dry_run=True, n_calls=len(GRID), rule_text=rule_text,
                    grid=list(GRID), standin_scores=scores,
                    standin=[asdict(r) for r in rows])
    # ---- live: elicit the field once, reuse across seeds ----
    gen = LLMEnsembleJudge(rule_text=rule_text)
    scores: List[Optional[float]] = []
    for r in GRID:
        v = gen._ask(cost_prompt(rule_text, grid_state(r)))
        scores.append(v)
        print(f"  rul={r:3d}: score={v}")
    fn = make_cost_field(GRID, scores)
    rows = evaluate_field(fn, seeds)
    _summ(rows, "free-form field (ALL seeds)", order=ORDER)
    return dict(dry_run=False, n_calls=len(GRID), rule_text=rule_text,
                grid=list(GRID), scores=scores, rows=[asdict(r) for r in rows])


def main():
    ap = argparse.ArgumentParser(description="free-form LM cost inference baseline (#1B)")
    ap.add_argument("--live", action="store_true",
                    help="actually call the claude CLI to elicit the field (spends budget)")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--out", type=str, default="results/paper_extra/llm_freeform")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    mode = "LIVE (spending API budget)" if args.live else "DRY-RUN (no calls, no spend)"
    print(f"1B free-form LM cost inference baseline -- {mode}")
    res = run(seeds, args.live)
    print(f"\n  TOTAL claude calls if --live: {res['n_calls']}")
    if not args.live:
        print("  (dry run -- nothing was sent. Re-run with --live to execute.)")
        return
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    p = out_dir / f"llm_freeform_{stamp}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
