# REGISTRATION V20 — Natural2CTL: external single-reference target recovery at scale

Written 2026-09-02, before any candidate is generated. Implements the
optional secondary benchmark of revision item W3: "can Generate+Keep
recover independently curated formal targets at a larger scale?"
Natural2CTL (Zrelli et al., REFSQ 2024; github.com/RimZrelli/PublicDataset,
Natural2CTL.csv, 2,094 rows) pairs each natural-language requirement with
ONE expert-validated CTL formula and is therefore a target-recovery
benchmark, not an ambiguity benchmark; the paper says so.

## Units

A stratified random sample of 200 rows (numpy seed 20260902), strata =
the dataset's pattern label collapsed to its family (Pattern 6.1 response,
4.1 universality, 2.1 existence, everything else), proportional
allocation, restricted to rows whose gold CTL parses under the grammar
below (rows that do not parse are counted and excluded before sampling).

## Generate (target-blind except for the atom vocabulary)

For each unit, K = 10 independent `claude -p` calls (the same local CLI
the paper's frozen scorer uses; version recorded). The prompt gives the
requirement text, the CTL grammar (AG, AF, AX, EG, EF, EX, A[.U.], E[.U.],
A[.W.], !, &, |, ->, <->), and the list of atomic-proposition names
appearing in the gold formula (as ARTEMIS supplies its variable
dictionary), and asks for ONE CTL formula and nothing else. The gold
formula itself is never shown. Outputs that do not parse are dropped and
counted.

## Semantic matching

Candidates and gold are compared by CTL model checking on a fixed battery
of 400 random finite Kripke structures over the unit's atom set (2-5
states, total transition relation, random labelling, seed 20260902):
two formulas AGREE when they take the same truth value at every state of
every structure. Disagreement is a sound proof of inequivalence;
agreement is bounded evidence of equivalence and is labelled as such.
Exact syntactic identity after canonicalization is reported separately.

## Score, Keep, baselines, metrics

Score = choice frequency of the agreement class in the K samples; Keep =
leave-one-unit-out split conformal at delta_sem = 0.10 over units whose
gold is in the pool (generation misses reported as delta_gen_hat);
baselines = top-1 (first parsed sample) and self-consistency (most
frequent class). Metrics per unit: target generated, target retained,
retained classes; by pattern family.

## Branch rules

Everything is reported wherever it lands. If the frequency score is
uninformative (median efficiency > 0.9) the paper says so. No prompt,
sample size or battery is changed after seeing results.

Output: results_e2e/natural2ctl_external.json; macros via make_gen_artemis.py.
