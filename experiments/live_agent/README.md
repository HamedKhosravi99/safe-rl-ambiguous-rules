# Workstream G — live agent demonstration harness

Harness for paper **§22** (CORSET governing a real tool-using agent). A bounded
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
| `policy_candidates.md` | policy-text selection (DEFERRED: `corpus/real_pairs/pilot_pairs.jsonl` not yet produced) |
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
- **Log-generation phase: stop-loss at 50% of the \$100 demo cap** (i.e. \$50),
  per §22.6 step 4. Wire the same `StopLossGuard(limit=...)` around the logging
  behavior policy.
- Spend is always measured from **provider usage fields** via the pinned sheet,
  never estimated ahead of the call; the guard checks the running ledger total.

## Phase plan (full demo cap \$100; ~\$95 reserved AFTER policy selection)

| Phase | What | Est. \$ | Runs when |
|-------|------|---------|-----------|
| 0. Build + dry run | this harness; 5 passthrough episodes, real calls | <= \$5 (cap), \$3 stop | **done** (spent \$0.00 — see below) |
| 1. Policy selection | shortlist 3 from `corpus/real_pairs/`, pin one, freeze DSL pool + ensemble scores | ~\$0 (subscription) | after `pilot_pairs.jsonl` exists |
| 2. Log generation | 300-500 episodes x ~10 paid calls, mixture behavior policy | ~\$25 (stop-loss \$50) | after phase 1 |
| 3. Guard training | FQI on logs -> single/corset/post_clarify guards | \$0 (CPU) | after phase 2 |
| 4. Live eval | 3 conditions x 50-100 episodes | 3 x ~\$10 = ~\$30 | after phase 3 |
| 5. VoQ + ask-vs-act | compute VoQ, fire clarification, run post-clarify arm | included in phase 4 | after phase 4 |
| — Reserve | pricing drift / re-runs | ~\$35 | — |

**Do NOT run phases 2-5 during the build phase.** They are gated on the selected
policy text and would consume the reserved ~\$95.

## Commands

```bash
# (0) build the self-contained corpus + task cards (idempotent)
python3 demo/build_corpus.py

# (0) DRY RUN — 5 episodes, passthrough+stop-loss, REAL paid calls, hard stop $3
python3 demo/run_dryrun.py --model gpt-5-nano --episodes 5 --stop 3.00
#   verifies: (a) billing_log == re-summed usage x price sheet, to the cent
#             (a') synthetic accounting proof (works even with no live credit)
#             (b) reconcile vs OpenAI Costs API (admin-key only; documented if 403)
#             (c) episodes serialize to saorl OfflineDataset; FQI trains 3 guards
#   prints TOTAL SPEND to the cent; writes logs/dryrun_summary.json

# component self-tests (no API cost)
python3 demo/agent_loop.py     # free-only episode: scorer + retrieval
python3 demo/guard.py          # fabricated logs: all 3 guards train + gate

# ---- LATER PHASES (after policy selection; NOT during build) ----
# (2) log generation  (to be added: run_logs.py, StopLossGuard limit=50.0)
#     python3 demo/run_logs.py --episodes 400 --stop 50.00 --behavior mixture
# (4) live eval       (to be added: run_eval.py, one run per guard condition)
#     python3 demo/run_eval.py --guard single       --episodes 60 --stop 10.00
#     python3 demo/run_eval.py --guard corset        --episodes 60 --stop 10.00
#     python3 demo/run_eval.py --guard post_clarify  --episodes 60 --stop 10.00
```

## Build-phase dry-run result (2026-07-26)

- **Total spend: \$0.00** — 60 paid calls attempted, **0 billed**: the configured
  OpenAI key returns `429 insufficient_quota` on every chat completion (the
  models endpoint, which is free, works). Live billing is blocked until the
  account/project has spendable credit. **The harness spent nothing; well under
  the \$5 cap.**
- **(a) internal reconciliation**: trivially exact (0 live calls).
- **(a') synthetic accounting proof**: EXACT on known usage blocks (proves the
  cost math is correct and ready to bill the moment credit exists).
- **(b) Costs API**: `403` — the endpoint requires an ADMIN (org) key; the
  configured key is project/user scope. Internal reconciliation (log vs summed
  usage fields) is the primary check and holds to the cent.
- **(c) schema + FQI**: episodes serialize to `saorl.OfflineDataset`; the FQI
  learner consumes them and trains all three guards.

### To finish phase-0 validation once credit is available

Add spendable credit to the OpenAI project, then re-run
`python3 demo/run_dryrun.py` — it will bill real dollars (expected << \$1 for 5
episodes at gpt-5-nano), and checks (a)/(b) will exercise live-billed calls. No
code change is needed; the harness already handles it.
