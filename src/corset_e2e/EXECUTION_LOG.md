# CORSET-E2E execution log (plan: `paper/make8/corset_e2e_iclr8_execution_plan.pdf`)

Freeze chain: v1 manifest @057094b → v1 test executed → v1 outcome + v2
manifest @d95d90c → v2 test executed. Each manifest was committed before the
test split it governs was evaluated.

## What was executed

| WP | Scope | Status |
|----|-------|--------|
| E.1 | visible/hidden stores, grouped cluster splits | done |
| A | frozen bounded DSL, finite grids, eligibility codes | done |
| B | target-independent complete enumeration + 5 leak audits | done |
| C | deterministic additive score, dev-only weight fit, conformal calibration | done |
| D | explicit out-of-DSL abstention path | done |
| E | source-grounded end-to-end frozen test (×2 versions) | done |
| F | non-nested neural benchmark (3 families, ~30 configs, prescreen) | **not executed** |
| G | vector-cost offline RL training | **not executed** |
| H | 1,000-episode fixed-policy certificates | **not executed** |
| I | local authoritative query sweep | **not executed** |

F–I require building three simulator families, prescreening ~30 task
configurations with online oracles, generating 10⁵–10⁶-transition offline
datasets, and training 500k-step vector-cost learners at ≥5 seeds across
≥12 tasks and ≥2 learner families. That is a multi-day GPU campaign, not a
session's work, and none of it was started. Nothing in this log or in the
committed artifacts claims otherwise.

## Data

13 pinned repositories → 9,776 alert rules → 1,015 source-file clusters.
Split by HMAC of the cluster path under a committed salt: dev = the two
already-analyzed repositories (1,094 rules), cal 4,292, test 4,390 rules
over 464 clusters. The plan asked for ≥30 untouched test clusters and ≥150
in-DSL rules; there are 464 clusters, but only 315 grammar-reachable rules.

## Outcomes (test split, executed once per version)

| Metric | Gate | v1 | v2 |
|---|---|---|---|
| leak audit | 5/5 mandatory | **PASS** | **PASS** |
| ρ_gen | ≥0.90, lower bound ≥0.85 | 0.359 [0.191, 0.479] **FAIL** | 0.438 [0.227, 0.640] **FAIL** |
| ρ_ret \| gen | clears nominal 0.90 | 0.974 [0.936, 1.000] PASS | 0.848 [0.789, 0.898] **FAIL** |
| ρ_e2e | reported directly | 0.025 | 0.027 |
| OOD recall | ≥0.80 | 0.859 PASS | 0.884 PASS |
| false abstention | ≤0.25 | 0.752 **FAIL** | 0.826 **FAIL** |
| \|Û\| median | reported | 2,554,034 | 1,863,454 |
| \|Max(Û)\| median | reported | 3,972 | 1,236 |

Calibration-split conditional retention is 0.9076 (v1) and 0.9049 (v2)
against a 0.90 nominal level, so the conformal construction is calibrated
correctly. What fails is candidate generation, not retention.

**Registered verdict: the primary generation gate fails in both versions.**
Per the plan's failure branch, the revised paper may not claim a viable
end-to-end generator, and the current manuscript should keep its existing
careful scope rather than overclaim.

## Diagnosis

* **Every generation failure is the metric axis.** 202/202 sampled v1
  GEN_MISS units were metric-not-licensed; the five frozen grid axes never
  caused a miss. That is exactly what complete enumeration buys, and it
  localises the whole problem to one axis.
* **ρ_gen tracks metric-namespace overlap.** v1: 0.881 on
  cluster-monitoring-operator (Kubernetes family, like the dev repos) against
  0.05–0.10 on victoriametrics, tidb, and loki. Only 17.8% of v1 test target
  metrics existed in the dev catalog.
* **v2 confirmed the mechanism but could not fix it with this data.** Giving
  each unit its own deployment namespace raised ρ_gen 0.359 → 0.438. It could
  not go further because the namespaces harvested here are impoverished
  (12–133 metrics per repository), since 80% of rules do not parse and so
  contribute no metric names. A real deployment lists thousands of metrics
  from its monitoring API; that catalog is the missing input, not a modelling
  choice.
* **Grammar reach is the other wall.** Only 315 of 4,390 test rules (7.2%)
  are expressible at all, and 81% of parser failures are `on`/`ignoring`
  vector matching — the single highest-value DSL extension.
* **Completeness costs efficiency.** Median |Û| is ~1.9M readings, falling to
  ~1.2k after dominance pruning. Target-independence and small retained sets
  are in direct tension, and the previous target-seeded pools bought their
  small sets partly with that seeding.

## Two bugs the discipline caught

1. **Eligibility inflation.** My first eligibility classifier also required
   the target metric to be in the generator's licensed set. That would have
   relabelled every generation failure as out-of-DSL and dropped it from the
   denominator, reporting ρ_gen ≈ 1.0 by construction — the exact defect the
   plan exists to remove. Eligibility now measures grammar reach only, and a
   licensed-set miss counts against ρ_gen.
2. **A real leak in v2.** Computing a unit's catalog as
   `repo_metrics − own_cluster_metrics` makes the *exclusion* set a function
   of the hidden target, so swapping the target changed the pool. The
   target-swap audit failed 5/150 and caught it. Rewritten as a union over
   the repository's other clusters, which never reads the unit's own cluster;
   now 150/150.

## Next engineering cycle (must be developed on dev, then re-frozen)

1. Supply the deployment metric namespace as declared metadata (the plan's
   A.3 allowlist already permits it) instead of harvesting it from parseable
   alert rules.
2. Extend the grammar to `on`/`ignoring` vector matching — 81% of the
   unreachable mass.
3. Re-tune the OOD threshold: false abstention 0.75–0.83 is far above the
   0.25 gate, because the bottom-candidate signal currently keys on lexical
   metric evidence, which is weak for in-DSL and out-of-DSL units alike.
4. Only then re-run calibration and a new frozen test as v3.

---

# Cycle: v11 deterministic retrieval licenser (2026-09-01)

External review (post-R6) pressed the confessed bottleneck: ARROW starts
after candidate generation, and generation is the dominant loss. This
cycle attacks the identified mechanism rather than adding a stage.

1. **Provenance repair first.** v4_recall.json had no producing script;
   analysis/dev_miss_autopsy.py now re-derives all five fields exactly
   (availability 0.498/0.803/0.832, genmiss_recovered=124 under the
   definition: v3-run GEN_MISS whose metric the v4 catalog contains).
   The .tokens harvester was decoded to ~90% set agreement
   (source_grounded/harvest_tokens.py; committed catalogs stay the
   artifacts of record); all 13 clone SHAs pinned
   (catalog_src_manifest.json). The v6 endpoint (0.600@16) now has a
   verifier over its archived raw selections
   (analysis/verify_v6_archive.py).

2. **Root cause localized.** Complete mode makes rho_gen pure metric
   licensing. Per-repo aggregates of the archived v4 rows show the
   collapse tracks namespace size (victoriametrics 0.050 at 17,956
   names; tidb 0.643). Availability (0.83) and catalog precision (v5)
   were already exonerated. The frozen license ranks by subtoken-overlap
   fraction with alphabetical tie-breaks: it cannot separate one name
   from 1e4.

3. **A design failure caught on dev, not on test.** The first v11 ranker
   (IDF-weighted COVERAGE) lost to the frozen license under the dev
   stress regime (0.24-0.31 vs 0.35 recall-given-available). The miss
   dump showed why: coverage normalization hands top ranks to two-token
   harvest debris ('error_value', 'label_values') that generic rule text
   fully covers, and the normalization table amplifies it. Replaced by
   MATCHED-EVIDENCE-MASS scoring (num/(num+1+0.25*unmatched)) before
   anything touched cal/test.

4. **Registered outcome (single test pass, REGISTRATION_V11 branch rule 2).**
   rho_gen@64 = 0.5746 [0.439, 0.691] vs the frozen v4 arm's 0.352 and the
   best lexical arm's 0.438; the frozen ranker on the SAME union catalog
   moves only to 0.368, so +20.6 of the +22.2 points are the ranker.
   Channels are complementary (name-only 0.482, context-only 0.505, both
   0.575; normalization +1.9). Retention on the near-doubled licensed set
   (181 vs 111 units): 0.873 [0.834, 0.926], interval covering nominal
   0.90; composed 50.2% where expressible (v6 semantic pipeline: 48.6%),
   3.6% [2.0, 5.5] end to end (was 2.4%). The archived LLM licenser keeps
   the matched-breadth lead (0.600@16 vs 0.470@16); v11 UNION v6 = 0.743 =
   89.3% of the 0.832 availability ceiling. Gains land where diagnosed:
   victoriametrics 0.050 -> 0.600, mimir 0.395 -> 0.674; gitlab-runbooks
   stays hardest (0.272 -> 0.408; templated same-vocabulary crowding, the
   risk the registration named). Set cost: retained-set median unchanged,
   maximal antichain median 13,408 -> 18,150. Leak audit: all five pass,
   target-swap 300/300. Expressibility (7.2% of the corpus in-grammar) is
   now the dominant end-to-end loss by an order of magnitude, and the
   paper says so.

# Cycle: V13 exact policy sufficiency (2026-09-01)

The third review independently reinvented Gate-2/3's (B) -- forall-
optimal-face safety -- as "policy sufficiency" and asked for it as the
central object. The robust arm stays closed (Gate 3); the true-P exact
test was open, registered (V13), and run: B=C at d>=0.05 on all
clearing instances (asserted via 100 random tie-breaks per reading),
B<C at d<=0.02 (0/22 at 0.005; face-worst 1.51x budget; tie-break
safety 64-77%). Live crossed pool: face is a singleton, obeys the pool.
The no-negatives variant now carries the face theorem, the two-stage
decide, and the counts; the submission branch carries artifacts only.
