# Live service-agent check -- RESULTS

Appendix D.1 of the paper: ARROW (called CORSET in this code) governing a real, billed tool-using agent. The agent is a Slack-style app posting research-summary messages to a channel; each **paid post** is a real billed `gpt-4o-mini` call, and a **guard** gates posting exactly as the budget domain gates `buy`. The deployment unit is the real Slack Web-API posting-rate policy `real-a02-slack-posting-rate`.

## Headline

> On a live, real-billed tool-using agent governed by the Slack posting-rate policy (conformal-retained set U=['psi_calls>=3', 'psi_calls>=4', 'psi_calls>=5', 'psi_calls>=6', 'psi_calls>=8'], |U|=5), the CORSET worst-retained guard drove the episode-level violation rate of every retained reading to 0.00 while the single-most-plausible-reading guard (k=5) left the tighter retained readings violated at up to 1.00, and the no-guard agent violated them at 1.00 -- for a total real billed cost of $0.19 ($0.194101) across 6687 gpt-4o-mini posts.

**Total billed spend: $0.19** (exact $0.194101, 6687 billed posts total; primary ledger 3960 posts reconciled to the cent, plus 2727 posts from an aborted first attempt kept for honest accounting).

## Phase 1 -- SCORE (conformal retained set)

- Policy: `real-a02-slack-posting-rate` -- "apps may post no more than one message per second per channel ... we allow bursts over that limit for short periods".
- Reading pool: nested per-window burst caps `psi_calls>=k` (k tight->loose) + null; scored by the frozen 7-persona ensemble (claude CLI, as `saorl.judge`), cached in `ensemble_scores.json`.
- q-hat = 0.5 (reused from corpus `results_paper/conformal/conformal_report.json`).
- **Retained set U = ['psi_calls>=3', 'psi_calls>=4', 'psi_calls>=5', 'psi_calls>=6', 'psi_calls>=8']**  (means: psi_calls>=3=0.585714, psi_calls>=4=0.557143, psi_calls>=5=0.642857, psi_calls>=6=0.571429, psi_calls>=8=0.55).
- **Non-singleton: True** (|U|=5) -- GATE PASSED.
- most-plausible (single guard) = `psi_calls>=5`; strict anchor (1 msg/sec) = `psi_calls>=2` (scored 0.321 < q-hat -> **outside U**, a delta-miscoverage the clarification catches).

## Phase 2 -- LOGS

- 300 channel-session episodes, mixture behavior (greedy-post / randomized-fallback), real billed posts.
- 2860 billed posts; mean episode value 10.4878; log-phase spend $0.082946.
- Serialized to `logs/offline_episodes.jsonl` (saorl OfflineDataset schema).

## Phase 3 -- GUARDS (FQI, saorl.learn_budget_constrained)

| guard | honors | effective cap | lambda | true-worst (retained, offline) |
|---|---|---|---|---|
| single | psi_calls>=5 | 5 | 1.0 | 0.278 |
| corset | psi_calls>=3,psi_calls>=4,psi_calls>=5,psi_calls>=6,psi_calls>=8 | 3 | 1.0 | 0.000 |
| post_clarify | psi_calls>=2 | 2 | 1.0 | 0.000 |

## Phase 4 -- LIVE EVAL (50 paired episodes/condition)

| condition | billed $ | task value | paid posts | retained-reading violation rate (k:rate) | anchor(k=2) viol |
|---|---|---|---|---|---|
| no_guard | 0.01743 | 11.86±0.06 | 12.0 | 3:1.00, 4:1.00, 5:1.00, 6:1.00, 8:1.00 | 1.00 |
| single | 0.00727 | 8.02±0.26 | 5.0 | 3:1.00, 4:1.00, 5:0.00, 6:0.00, 8:0.00 | 1.00 |
| corset | 0.00435 | 6.93±0.30 | 3.0 | 3:0.00, 4:0.00, 5:0.00, 6:0.00, 8:0.00 | 1.00 |
| post_clarify | 0.00289 | 6.37±0.33 | 2.0 | 3:0.00, 4:0.00, 5:0.00, 6:0.00, 8:0.00 | 0.00 |

## Phase 5 -- ASK-VS-ACT (VoQ, Thm 5)

- posterior over U (plausibility-normalized): {'psi_calls>=3': 0.2015, 'psi_calls>=4': 0.1916, 'psi_calls>=5': 0.2211, 'psi_calls>=6': 0.1966, 'psi_calls>=8': 0.1892}
- per-reading value v_psi: {'psi_calls>=3': 7.0266, 'psi_calls>=4': 7.5333, 'psi_calls>=5': 8.0885, 'psi_calls>=6': 8.6297, 'psi_calls>=8': 9.738}; v_robust (CORSET) = 7.0266.
- **VoQ (plausibility) = 1.16**, VoQ (robust) = 0.0, kappa = 0.5 return units.
- **Decision: ASK (fire clarification)** (fires = True). VoQ_robust = 0 because the worst-case answer cell is the tightest retained reading, which the CORSET policy already honors; the plausibility prior puts mass on looser readings, so the expected VoQ is strictly positive.
- Clarification transcript: `transcripts/clarification.txt`; post-clarify honors the strict anchor (k=2), realized value 6.4698.

## Deliverables (under demo/)

- `policy_source.json`
- `ensemble_scores.json`
- `conformal_set_report.json`
- `logs/offline_episodes.jsonl`
- `guards/single.json`
- `guards/corset.json`
- `guards/post_clarify.json`
- `logs/eval_no_guard.jsonl`
- `logs/eval_single.jsonl`
- `logs/eval_corset.jsonl`
- `logs/eval_post_clarify.jsonl`
- `transcripts/clarification.txt`
- `billing_log.jsonl`
- `billing_reconciliation.json`
- `fig_demo.pdf`
- `gen_demo.tex`
- `RESULTS.md`

## Honest caveats

- **This is a demonstration, not a coverage experiment.** It shows the full CORSET pipeline (conformal set -> FQI guards -> live billed eval -> ask-vs-act) end-to-end on one real policy; it is a single deployment unit, not a calibrated coverage study over many rules.
- **The strict 1 msg/sec anchor fell just below q-hat and is outside U** -- a genuine delta-miscoverage. CORSET's guarantee is over the *retained* set; the clarification query is exactly what recovers the true reading a plausibility-only agent would miss. (This is honest, not a bug: conformal coverage is 1-delta, not 1.)
- **VoQ(robust) = 0** because the worst-case answer cell equals the tightest retained reading, which CORSET already honors; the positive signal comes from the plausibility prior over looser readings.
- **Costs are project/user-scope**, reconciled internally (billing_log vs re-derived usage x pinned price sheet, exact to the cent); the org-scope Costs API needs an admin key (documented 403).
- **Hard stop-loss $15** was never approached (spent $0.19).

