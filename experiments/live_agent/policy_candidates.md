# Policy-text selection for the Workstream-G demo (paper §22.3 component 2, §22.6 step 1)

## Status: DEFERRED (blocked on the corpus miner)

The demo's deployment unit `ell` is a **real, public, provenance-pinned
usage/spending policy** whose text contains at least one genuinely vague
quantifier, so the interpretation set is non-singleton. Candidates are to be
shortlisted from **Tier-1+ finds in `corpus/real_pairs/`** once the miner agent
lands its output.

As of this build (checked 2026-07-26):

    corpus/real_pairs/            -> exists but EMPTY
    corpus/real_pairs/pilot_pairs.jsonl -> DOES NOT EXIST

So no candidates can be shortlisted yet. This file records the **selection
criteria** and the **exact shape the policy text must have** so that the moment
`pilot_pairs.jsonl` (or the Tier-1 finds) exists, selection is mechanical.

## What the harness needs from the policy text

The guard axis is already wired for **two DSL families** (both in the frozen
budget grammar, `saorl/budget.py` + `saorl/dsl.py`):

| Axis | State feature (per step) | Reading form | Natural for |
|------|--------------------------|--------------|-------------|
| spend cap | `spend` (real billed \$) | `spend >= theta` | policies with a \$ limit ("stay under \$X") |
| call quota | `calls` (paid-call count) | `calls >= q` | policies with a call/request limit ("avoid excessive calls", "N requests/day") |

Because the metered mini/nano tool costs ~\$1e-4 per call, a **call-quota**
reading is the more legible demo axis (an episode makes up to 12 paid calls, so
quotas at 5/8/10 are meaningful and cheap). A **spend-cap** reading is used if
the selected policy states a dollar figure; the accumulation is then over the
whole condition's episodes (the figure's y-axis is cumulative billed \$).

## Selection criteria (all required)

1. **Public + pinnable**: URL + version/commit/timestamp + license + content
   hash recordable into `demo/policy_source.json` (per §22.7).
2. **At least one genuinely vague quantifier** — e.g. "keep usage modest",
   "avoid excessive calls", "stay well under the limit", "reasonable number of
   requests". This is what makes the conformal interpretation set non-singleton
   (§22.8 gate: a singleton set shows nothing).
3. **A documented formal anchor**: a numeric limit/config elsewhere in the same
   source (the rate-limit table, the quota config, the budget guardrail value)
   that serves as the externally-anchored target — used ONLY for (a) post-hoc
   scoring of which reading was correct and (b) resolving the clarification
   answer. The vague quantifier and the anchor must plausibly refer to the same
   resource so the readings nest.
4. **Category A or C** (API quota docs / CI-or-agent budget policy / cloud
   spending guardrail), matching the Workstream-A provenance taxonomy.
5. **Readings nest tight->loose** so the retained band is an interval (mirrors
   `saorl/budget.py`'s $30/$45/$60 nested caps) — required for the CORSET
   worst-retained guard to be strictly more conservative than the single-reading
   guard.

## Shortlisting procedure (run once `pilot_pairs.jsonl` exists)

1. Filter `corpus/real_pairs/pilot_pairs.jsonl` to rows with
   `tier >= 1` AND `category in {A, C}` AND a vague quantifier field present.
2. Keep only rows whose formal anchor is a **spend or per-window count** (so it
   maps onto the budget-domain DSL: `spend>=X` / `calls>=q`).
3. Rank by (vagueness strength) x (anchor clarity); take the top 3.
4. For each of the 3, draft the frozen DSL pool (3-5 nested readings spanning
   clearly-too-tight through clearly-too-loose around the anchor) and confirm
   the frozen 7-persona ensemble yields a **non-singleton** conformal set at the
   corpus q-hat. Reject any candidate whose set collapses to a singleton.
5. Record the winner in `demo/policy_source.json`; freeze
   `demo/pool.json` + cached ensemble scores before any scoring is read.

## Wiring already in place (no policy text required)

`demo/guard.py` exposes `make_reading(theta, axis)`,
`demo_call_quota_readings()` (placeholder nested quotas 5/8/10), and
`train_three_guards(episodes, readings=...)`. When the real policy is selected,
replace `demo_call_quota_readings()` with the frozen pool's three reading sets
(`retained` = conformal survivors, `single` = most-plausible, `anchored` =
post-clarification target) and everything downstream is unchanged.

## Shortlist (2026-07-26, from corpus/real_pairs/pilot_pairs.jsonl)

1. **PRIMARY: real-a02-slack-posting-rate (Tier 1)** — "apps may post no
   more than one message per second per channel ... we allow bursts over
   that limit for short periods". Exact anchor (1 msg/sec/channel) +
   vague burst clause => non-singleton reading set over
   calls-per-window quotas (strict 1/s; burst-tolerant k-in-w windows).
   Recognizable org; docs provenance pinned in pilot_pairs.jsonl.
2. **BACKUP: real-a07-dockerhub-fair-use (Tier 2)** — "excessive data
   transfer, pull rates" prose + separately documented numeric pull
   limits as anchor.
3. Also viable: real-a08-telegram (T1; weak same-prose anchor, flagged),
   real-a03-stripe (T2), real-a06-mediawiki (T2; no numeric anchor).

Decision rule (README): proceed with PRIMARY unless its reading set
under the corpus q-hat is singleton; then BACKUP.

## BLOCKER: OpenAI account credit

All phases beyond $0 are blocked on account credit (key valid,
insufficient_quota). $10-20 suffices at nano prices; cap $100.
