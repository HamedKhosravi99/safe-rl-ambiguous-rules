"""End-to-end DRY RUN for the Workstream-G harness (paper section 22.6 step 3).

Runs 5 episodes with the PASSTHROUGH guard and REAL paid calls -- this is the
<= $5 build-phase validation. It verifies:

  (a) the billing_log totals match the sum of the response usage fields EXACTLY
      (re-derive each line's cost from its logged token counts x the pinned
      price sheet and compare to the stored/accumulated totals);
  (b) reconciliation against the OpenAI Costs/Usage admin API (or documents that
      the endpoint is unavailable for this key type);
  (c) the episodes serialize to the saorl OfflineDataset schema and the existing
      FQI learner (train_three_guards) consumes them.

Hard stop-loss: the run STOPS as soon as measured spend reaches $3.00, whatever
the episode count. (The build phase's outer cap is $5; the dry run's is $3.)

Usage:
    python3 demo/run_dryrun.py [--model gpt-5-nano] [--episodes 5] [--stop 3.00]

Prints total spend to the cent.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from billing import (BillingLedger, cost_from_usage, load_price_sheet,
                     reconcile_with_usage_api)
from agent_loop import (load_corpus, load_tasks, run_episode, GuardLike,
                        Passthrough, greedy_paid_planner)
import guard as guard_mod

HERE = Path(__file__).resolve().parent
DRYRUN_BILLING_LOG = HERE / "logs" / "dryrun_billing.jsonl"
DRYRUN_EPISODES = HERE / "logs" / "dryrun_episodes.jsonl"


class StopLossGuard(GuardLike):
    """Passthrough until measured spend reaches `limit`, then deny PAID so the
    agent finishes on the free fallback -- prevents overshooting the stop-loss
    mid-episode."""
    def __init__(self, ledger: BillingLedger, limit: float):
        self.ledger = ledger
        self.limit = limit

    def allow_paid(self, features: Dict[str, Any]) -> bool:
        return self.ledger.total_cost < self.limit


def verify_billing_exact(log_path: Path, price_sheet: Dict[str, Any],
                         ledger_total: float) -> Dict[str, Any]:
    """(a) Re-derive every logged call's cost from its token counts and confirm
    it equals the stored per-line cost and the accumulated ledger total."""
    resummed = 0.0
    n = 0
    max_line_err = 0.0
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if not d.get("ok", True):
                continue
            usage = {
                "prompt_tokens": d["prompt_tokens"],
                "completion_tokens": d["completion_tokens"],
                "prompt_tokens_details": {"cached_tokens": d["cached_tokens"]},
            }
            recomputed = cost_from_usage(usage, d["model"], price_sheet)["total_cost"]
            max_line_err = max(max_line_err, abs(recomputed - d["total_cost"]))
            resummed += recomputed
            n += 1
    return {
        "n_calls": n,
        "resummed_total": resummed,
        "ledger_total": ledger_total,
        "abs_diff_total": abs(resummed - ledger_total),
        "max_per_line_abs_diff": max_line_err,
        "exact": abs(resummed - ledger_total) < 1e-12 and max_line_err < 1e-12,
    }


def verify_billing_math_synthetic(price_sheet: Dict[str, Any]) -> Dict[str, Any]:
    """Prove the cost-accounting logic is exact on KNOWN usage blocks, so
    requirement (a) is verifiable even when live billing is blocked (no credit).
    Hand-computes the dollar cost from the pinned rates and compares."""
    model = "gpt-5-nano"
    r = price_sheet["models"][model]
    cases = [
        {"prompt_tokens": 1000, "completion_tokens": 500,
         "prompt_tokens_details": {"cached_tokens": 0}},
        {"prompt_tokens": 2000, "completion_tokens": 300,
         "prompt_tokens_details": {"cached_tokens": 800},
         "completion_tokens_details": {"reasoning_tokens": 128}},
    ]
    checks = []
    for u in cases:
        cached = u["prompt_tokens_details"]["cached_tokens"]
        expect = ((u["prompt_tokens"] - cached) * r["input"]
                  + cached * r["cached_input"]
                  + u["completion_tokens"] * r["output"]) / 1e6
        got = cost_from_usage(u, model, price_sheet)["total_cost"]
        checks.append({"expect": expect, "got": got, "match": abs(expect - got) < 1e-15})
    return {"all_match": all(c["match"] for c in checks), "cases": checks}


def verify_schema_and_fqi(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """(c) Confirm episodes match the saorl OfflineDataset schema and the FQI
    learner consumes them (train the three guards on the dry-run logs)."""
    issues = []
    for ep in episodes:
        if not (len(ep["trajectory"]) == len(ep["actions"]) == len(ep["rewards"])):
            issues.append(f"{ep['task_id']}: length mismatch")
        for st in ep["trajectory"]:
            if "spend" not in st or "calls" not in st:
                issues.append(f"{ep['task_id']}: state missing spend/calls")
                break
        for a in ep["actions"]:
            if a not in ("buy", "skip"):
                issues.append(f"{ep['task_id']}: bad action {a!r}")
                break
    ds = guard_mod.episodes_to_offline(episodes, axis="calls")
    guards = guard_mod.train_three_guards(episodes, axis="calls")
    return {
        "schema_ok": not issues,
        "issues": issues,
        "offline_dataset_steps": ds.n_steps(),
        "guards_trained": {name: {"lam": g.lam, "honored_cost": g.honored_cost,
                                  "true_worst": g.true_worst}
                           for name, g in guards.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    price_sheet = load_price_sheet()
    ap.add_argument("--model", default=price_sheet["default_model"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--stop", type=float, default=3.00, help="stop-loss in $")
    args = ap.parse_args()

    DRYRUN_BILLING_LOG.parent.mkdir(parents=True, exist_ok=True)
    # fresh logs for a clean audit
    DRYRUN_BILLING_LOG.unlink(missing_ok=True)
    DRYRUN_EPISODES.unlink(missing_ok=True)

    corpus = load_corpus()
    tasks = load_tasks()[:args.episodes]
    ledger = BillingLedger(log_path=DRYRUN_BILLING_LOG, price_sheet=price_sheet)
    guard = StopLossGuard(ledger, limit=args.stop)
    planner = greedy_paid_planner()  # attempt PAID every step -> real billed calls

    run_start = int(time.time())
    print(f"== DRY RUN: {args.episodes} episodes, model={args.model}, "
          f"stop-loss=${args.stop:.2f} ==")

    episodes: List[Dict[str, Any]] = []
    total_attempts = 0
    api_errors: List[str] = []
    for i, task in enumerate(tasks):
        r = run_episode(task, guard, model=args.model, ledger=ledger,
                        planner=planner, corpus=corpus, episode_id=i)
        episodes.append(r.to_jsonable())
        total_attempts += r.n_paid_attempts
        if r.last_error:
            api_errors.append(r.last_error)
        with open(DRYRUN_EPISODES, "a") as f:
            f.write(json.dumps(r.to_jsonable()) + "\n")
        print(f"  ep {i} [{task['id']}] progress={r.final_progress:.2f} "
              f"paid_ok={r.n_paid}/{r.n_paid_attempts} ep_cost=${r.total_cost:.6f} "
              f"cum=${ledger.total_cost:.6f}"
              + (f"  API_ERR={r.last_error[:60]}" if r.last_error else ""))
        if ledger.total_cost >= args.stop:
            print(f"  ** STOP-LOSS hit at ${ledger.total_cost:.6f} "
                  f">= ${args.stop:.2f}; halting after episode {i} **")
            break
    run_end = int(time.time())

    # (a) exact internal reconciliation on live-billed calls (if any)
    bill = (verify_billing_exact(DRYRUN_BILLING_LOG, price_sheet, ledger.total_cost)
            if DRYRUN_BILLING_LOG.exists()
            else {"n_calls": 0, "resummed_total": 0.0, "ledger_total": 0.0,
                  "abs_diff_total": 0.0, "max_per_line_abs_diff": 0.0, "exact": True})
    # (a') synthetic proof the accounting logic is exact even with no live calls
    math_check = verify_billing_math_synthetic(price_sheet)
    # (b) provider usage/costs API cross-check
    recon = reconcile_with_usage_api(ledger.total_cost, run_start, run_end)
    # (c) schema + FQI consumption
    schema = verify_schema_and_fqi(episodes)

    live_ok = ledger.n_calls > 0
    print("\n== RESULTS ==")
    print(f"TOTAL SPEND: ${ledger.total_cost:.2f}  (exact: ${ledger.total_cost:.6f})")
    print(f"paid calls billed: {ledger.n_calls} / attempted: {total_attempts}")
    if not live_ok and total_attempts > 0:
        print("  !! NO PAID CALL SUCCEEDED -- live billing is BLOCKED.")
        if api_errors:
            print(f"     API error: {api_errors[0]}")
        print("     The <=$5 live-billing validation cannot run until the OpenAI")
        print("     account/project has spendable credit. Harness logic is proven")
        print("     below via the synthetic accounting check (a').")
    print("\n(a) internal billing reconciliation (log vs re-derived usage x price sheet):")
    print(f"    calls audited        : {bill['n_calls']}")
    print(f"    ledger total         : ${bill['ledger_total']:.8f}")
    print(f"    re-summed from log   : ${bill['resummed_total']:.8f}")
    print(f"    abs diff             : {bill['abs_diff_total']:.2e}")
    print(f"    max per-line diff    : {bill['max_per_line_abs_diff']:.2e}")
    print(f"    EXACT MATCH          : {bill['exact']}"
          + ("" if bill["n_calls"] else "  (no live calls; trivially exact)"))
    print("\n(a') synthetic accounting proof (known usage -> hand-computed $):")
    print(f"    all cases match      : {math_check['all_match']}")
    for c in math_check["cases"]:
        print(f"      expect=${c['expect']:.10f} got=${c['got']:.10f} match={c['match']}")
    print("\n(b) provider usage/costs API cross-check:")
    if recon.get("available"):
        print(f"    api reported cost    : ${recon['api_reported_cost']:.6f}")
        print(f"    difference vs ledger : ${recon['difference']:.6f}")
        print(f"    note                 : {recon['note']}")
    else:
        print(f"    UNAVAILABLE          : {recon['reason']}")
    print("\n(c) schema + FQI consumption:")
    print(f"    schema_ok            : {schema['schema_ok']}  issues={schema['issues']}")
    print(f"    offline dataset steps: {schema['offline_dataset_steps']}")
    for name, g in schema["guards_trained"].items():
        print(f"    guard[{name:12s}]   lam={g['lam']:.1f} "
              f"honored_cost={g['honored_cost']:.3f} true_worst={g['true_worst']:.3f}")

    summary = {
        "model": args.model, "episodes_run": len(episodes),
        "total_spend": ledger.total_cost, "paid_calls_billed": ledger.n_calls,
        "paid_calls_attempted": total_attempts, "live_billing_ok": live_ok,
        "api_error": api_errors[0] if api_errors else None,
        "billing_exact": bill["exact"],
        "billing_math_synthetic_ok": math_check["all_match"],
        "recon": recon.get("available"),
        "recon_reason": recon.get("reason"), "schema_ok": schema["schema_ok"],
        "run_start": run_start, "run_end": run_end,
    }
    (HERE / "logs" / "dryrun_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote logs/dryrun_summary.json")
    return summary


if __name__ == "__main__":
    main()
