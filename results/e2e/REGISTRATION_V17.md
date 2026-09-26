# REGISTRATION V17 — external semantic validation on ARTEMIS expert-plausible sets

Written 2026-09-02, before any retained set, coverage or recall number on
these units exists. Implements revision item P3/W3 of
paper/final/ICLR2027_revision_plan_revised.md: replace the (infeasible)
human study with an evaluation against independently expert-enumerated
plausible formal readings that were not used to build ARROW's protocol.

## Benchmark and units

ARTEMIS artifact (Mendoza, Mavridou, Katis, Trippel, ICSE 2026),
https://github.com/dmmendo/ARTEMIS, pinned at the commit recorded in
results/e2e/artemis_external.json. Units are the requirements of the three
multi-reference FRETish groups: Ventilator (121), Robotics/RobotExplain (46),
LMCPS = FSM-AP (9) + FSM-S (4) + REG (2) = 15; 182 in all. DeepSTL and Thales
are single-reference and are not used. The expert set P_l of a unit is the
product of the option lists in benchmarks/*/PlausibleSpecs.xlsx expanded
exactly as the artifact's own accuracy computation does
(data_loader.load_labels with max_N_DURATION = 1), mapped to LTL through the
artifact's FRETish template dictionary. Nothing in P_l is read by the
candidate-generation or scoring steps below.

## Generate (target-blind by construction)

Candidates are the artifact's archived LLM translations, produced by its
authors without access to P_l. Two pools are declared:

* **P1 (primary): the flagship generator.** nl2structnl-reflect,
  gemini-2.5-flash, 50 trials per unit (available for all 182 units).
* **P2 (secondary, widened): the union** of every archive that exists for all
  182 units: {nl2structnl, nl2ltl, nl2ltltemplate, nl2spec, NL2TL, synthtl}
  x {gemini-2.5-flash, gpt-4.1} at 10 trials, {NL2TL-FT, deepstl} x
  gemini-2.5-flash at 10 trials, plus P1 -- 190 trials per unit.

A trial is a candidate only if its output_LTL parses as LTL (the artifact
applies the same filter through Spot). Candidates are merged into semantic
classes by exact LTL equivalence (L(phi & !psi) = L(psi & !phi) = empty),
computed by corset_e2e/external/ltl_equiv.py, validated before any outcome
is read against the artifact's archived Spot verdicts (the per-trial
`is_equal` column of every *_metrics.json of the files above).

## Score and Keep

The frozen score of a class is its **choice frequency**: the fraction of the
pool's valid trials that land in the class. It is model-free, deterministic
given the artifact, and is the calibrated form of the paper's majority-vote
baseline. The paper's persona ensemble is NOT used here (no new model calls);
conformal validity does not depend on the score, and the paper says so.

Keep is leave-one-unit-out split conformal at delta_sem = 0.10 over the
pooled 182 units, conditioned as Lemma 1 is: a unit enters calibration only
if some plausible reading is in its pool (the others are generation misses,
reported as the estimated delta_gen of this benchmark). Conformity score of
a calibration unit = the largest class frequency among its plausible
readings in the pool (so the guarantee is Pr[P_l ∩ U_l ≠ ∅ | proposed] ≥
0.9). Secondary: the smallest such frequency (guarantee: every proposed
plausible reading retained). Threshold = the k-th smallest calibration
score, k = floor(delta (n+1)); ties retained.

## Baselines (all on the same pool, same classes)

* top-1: the class of the FIRST valid trial of the flagship generator;
* self-consistency singleton: the most frequent class (ties: the class
  containing the earliest trial in file order);
* ARROW Keep: the calibrated retained set;
* unfiltered pool: every class.

## Metrics (per unit, then averaged; also by |P_l| bin 1 / 2-5 / 6-10 / >10 and by group)

A. any-plausible coverage 1[P ∩ U ≠ ∅]; B. expert-reading recall
|P ∩ U| / |P| (P as enumerated; also over proposed P); C. all-plausible
coverage 1[P ⊆ U]; D. retained classes |U| and efficiency |U| / |pool
classes|. Semantic, never textual, matching throughout.

## Branch rules (fixed now)

1. If realized any-plausible coverage on proposed units falls below 0.90
   minus two binomial standard errors, the LOO calibration is reported as
   failing (exchangeability across requirements does not hold) -- reported,
   not repaired.
2. If median efficiency exceeds 0.9, the frequency score is declared
   uninformative on this benchmark and the paper says so.
3. Every recall gap between the retained set and the two singletons is
   reported wherever it lands, per bin. The hypothesis that the gap grows
   with |P_l| is a hypothesis; it is not a target and nothing is tuned.
4. If the checker disagrees with more than 1% of the archived Spot
   verdicts, the run is stopped until the disagreements are explained.

No parameter is chosen after seeing these units. The claim licensed by a
positive result is external semantic validity against independent expert
enumeration -- not calibration to human intent.
