"""Phase 2 LOGS: generate the offline logs for the Slack posting-rate demo
(paper section 22, phase 2).

~300 channel-session episodes under a MIXTURE behavior policy (per-episode
greedy-post or randomized-fallback), REAL billed gpt-4o-mini posts, a StopLoss
guard, per-step features (running calls + spend), deterministic task rewards, and
billed cost per post. Serializes to demo/logs/offline_episodes.jsonl in the
saorl OfflineDataset schema (trajectory state dicts carry 'calls' + 'spend';
actions in {buy, skip}; rewards = realized per-step value). Billing lines append
to demo/billing_log.jsonl (the demo-wide ledger).

Episodes run in parallel threads (each episode is sequential internally); the
LockedLedger serializes billing writes and the StopLoss check.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
from pathlib import Path
from typing import List

import numpy as np

from agent_loop import load_corpus, load_tasks
from billing import load_price_sheet
from channel_agent import (LockedLedger, StopLossGuard, make_batches,
                           mixture_behavior_planner, run_channel_episode)

HERE = Path(__file__).resolve().parent
BILLING_LOG = HERE / "billing_log.jsonl"
OFFLINE_EPISODES = HERE / "logs" / "offline_episodes.jsonl"
LOG_SUMMARY = HERE / "logs" / "log_summary.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--stop", type=float, default=10.0, help="stop-loss $")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    OFFLINE_EPISODES.parent.mkdir(parents=True, exist_ok=True)
    OFFLINE_EPISODES.unlink(missing_ok=True)

    corpus = load_corpus()
    tasks = load_tasks()
    ps = load_price_sheet()
    ledger = LockedLedger(log_path=BILLING_LOG, price_sheet=ps)
    guard = StopLossGuard(ledger, limit=args.stop)
    planner = mixture_behavior_planner(p_greedy=0.5, p_buy=0.55)
    batches = make_batches(tasks, args.episodes)

    run_start = int(time.time())
    print(f"== Phase 2 LOGS: {args.episodes} channel episodes, model={args.model}, "
          f"stop=${args.stop:.2f}, workers={args.workers} ==")

    episodes = [None] * args.episodes

    def _run(e: int):
        try:
            return e, run_channel_episode(
                batches[e], guard, model=args.model, ledger=ledger,
                planner=planner, corpus=corpus, episode_id=e,
                rng=np.random.default_rng(1000 + e)), None
        except Exception as ex:  # never let one episode crash the whole run
            return e, None, str(ex)[:200]

    done = 0
    errors = 0
    failed = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_run, e) for e in range(args.episodes)]
        for f in cf.as_completed(futs):
            e, ep, err = f.result()
            if ep is None:
                failed += 1
                continue
            episodes[e] = ep
            done += 1
            if ep.last_error:
                errors += 1
            if done % 50 == 0 or done == args.episodes:
                print(f"  {done}/{args.episodes} episodes | cum spend "
                      f"${ledger.total_cost:.6f} | calls {ledger.n_calls} | "
                      f"errors {errors} | failed {failed}", flush=True)

    episodes = [ep for ep in episodes if ep is not None]
    with open(OFFLINE_EPISODES, "w") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep.to_jsonable()) + "\n")

    total_paid = sum(ep.n_paid for ep in episodes)
    mean_value = float(np.mean([ep.total_value for ep in episodes]))
    calls_hist = np.bincount([ep.n_paid for ep in episodes], minlength=13).tolist()
    summary = {
        "phase": "logs",
        "episodes": len(episodes),
        "episodes_requested": args.episodes,
        "episodes_failed": failed,
        "model": args.model,
        "total_spend": ledger.total_cost,
        "paid_calls_billed": ledger.n_calls,
        "paid_posts_total": total_paid,
        "mean_episode_value": round(mean_value, 4),
        "n_paid_per_episode_hist": calls_hist,
        "episodes_with_api_error": errors,
        "run_start": run_start, "run_end": int(time.time()),
    }
    LOG_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"\n  TOTAL LOG SPEND: ${ledger.total_cost:.6f}  "
          f"({ledger.n_calls} billed posts)")
    print(f"  mean episode value: {mean_value:.3f} | n_paid/episode hist: {calls_hist}")
    print(f"  wrote {OFFLINE_EPISODES.relative_to(HERE)} and "
          f"{LOG_SUMMARY.relative_to(HERE)}")
    return summary


if __name__ == "__main__":
    main()
