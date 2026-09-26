# Live service-agent check

Harness for the live service-agent check of Appendix D.1 of the paper (ARROW, called CORSET in this code, governing a real tool-using agent). A bounded
tool-using agent does research-and-summarize over a **local** document corpus;
per step it chooses the **PAID tool** (a metered OpenAI mini/nano call that bills
real dollars) or a **FREE fallback** (local keyword retrieval + a truncated
extractive snippet). A guard gates the paid tool exactly as the budget domain
gates `buy`. This mirrors `saorl/budget.py` with **real bills** on the y-axis.

## Files

| File | Role |
|------|------|
| `agent_loop.py` | bounded agent (<=12 steps), corpus retrieval, deterministic rubric scorer, episode runner, planners/behavior policies |
| `billing.py` | paid tool (`call_openai`), cost accounting from response usage x pinned price sheet, `billing_log.jsonl` writer, usage/costs-API reconciliation |
| `guard.py` | the gate: `Passthrough` + `TrainedGuard`; reuses `saorl.budget_rl.learn_budget_constrained` (imported, saorl untouched); `train_three_guards` = single / corset / post_clarify |
| `price_sheet.json` | PINNED model ids + input/cached/output rates + date (provenance) |
| `build_corpus.py` | writes `corpus_docs/*.txt` (15 authored docs) + `tasks/*.json` (43 cards) |
| `corpus_docs/`, `tasks/` | self-contained corpus + task cards (FREE fallback always has a real corpus) |
| `run_dryrun.py` | end-to-end DRY RUN (<=$5 build-phase validation; hard stop-loss $3) |
| `logs/` | dry-run + (later) log-generation and live-eval outputs |

## Cost model

Per paid call: `cost = (prompt-cached)*input + cached*cached_input + completion*output`,
all `/1e6`, at `price_sheet.json` rates. `completion_tokens` already includes
`reasoning_tokens` for gpt-5* models (billed once, as output). Default paid model
`gpt-5-nano` (cheapest gpt-5 tier: \$0.05 / \$0.40 per 1M in/out as of 2026-07-26).

## Stop-loss logic

- **Build phase (this stage): hard cap \$5**; the dry run's own stop-loss is **\$3**.
  `run_dryrun.py` runs a `StopLossGuard` that denies the paid tool once measured
  spend reaches the limit (so a mid-episode call cannot overshoot), and breaks
  the episode loop when `ledger.total_cost >= stop`.
- **Log-generation phase: stop-loss at 50% of the \$100 demo cap** (i.e. \$50). Wire the same `StopLossGuard(limit=...)` around the logging
  behavior policy.
- Spend is always measured from **provider usage fields** via the pinned sheet,
  never estimated ahead of the call; the guard checks the running ledger total.


## Commands

```bash
# (0) build the self-contained corpus + task cards (idempotent)
python3 experiments/live_agent/build_corpus.py

# (0) DRY RUN — 5 episodes, passthrough+stop-loss, REAL paid calls, hard stop $3
python3 experiments/live_agent/run_dryrun.py --model gpt-5-nano --episodes 5 --stop 3.00
#   verifies: (a) billing_log == re-summed usage x price sheet, to the cent
#             (a') synthetic accounting proof (works even with no live credit)
#             (b) reconcile vs OpenAI Costs API (admin-key only; documented if 403)
#             (c) episodes serialize to saorl OfflineDataset; FQI trains 3 guards
#   prints TOTAL SPEND to the cent; writes logs/dryrun_summary.json

# component self-tests (no API cost)
python3 experiments/live_agent/agent_loop.py     # free-only episode: scorer + retrieval
python3 experiments/live_agent/guard.py          # fabricated logs: all 3 guards train + gate

# log generation (StopLossGuard limit=50.0)
python3 experiments/live_agent/run_logs.py --episodes 400 --stop 50.00 --behavior mixture
# live evaluation, one run per guard condition
python3 experiments/live_agent/run_eval.py --guard single       --episodes 60 --stop 10.00
python3 experiments/live_agent/run_eval.py --guard corset        --episodes 60 --stop 10.00
python3 experiments/live_agent/run_eval.py --guard post_clarify  --episodes 60 --stop 10.00
```

