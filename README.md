# Safe RL under ambiguous written rules — code, archived results, and paper generators

Code and data behind *Ambiguity Relevance for Reinforcement Learning Over Written Rules (ARROW)*.
Every number, table, and figure in the paper is produced by a script in this repository from an
archived result file under `results/`; `scripts/reproduce_paper.py` regenerates all of them and
checks them byte-for-byte against the fragments compiled into the submitted PDF.

## Repository layout

```
.
├── src/
│   ├── saorl/                 # the library: monitoring controller and occupancy LPs (exact_lp.py,
│   │   │                      #   benchmark_sg/control_mdp.py, control_suite.py), the ARROW decision
│   │   │                      #   (benchmark_sg/policy_sufficiency.py), SAFE-KEEP (benchmark_sg/safe_keep.py),
│   │   │                      #   the certified fixed-data planner (benchmark_sg/certified_protect.py),
│   │   │                      #   clarification studies (query_loop.py, safe_collapse.py), rule parsing and
│   │   │                      #   corpus tooling (parse.py, candidates.py, fetch.py), learning arms
│   │   │                      #   (experiments.py, budget_rl.py, cmapss*.py) and every experiment driver
│   │   └── benchmark_sg/
│   └── corset_e2e/            # open-domain generation pipeline: candidate generation, calibration,
│                              #   coverage analyses, the ARTEMIS external study
├── experiments/
│   ├── live_agent/            # the tool-using agent under a published rate limit (Appendix E), with logs
│   └── theory_extension/      # finite-data certificate experiment on the 28 compiled rules
├── scripts/
│   ├── paper/                 # make_gen_*.py, make_figs_*.py: archived results -> LaTeX fragments and figures
│   └── reproduce_paper.py     # runs them all in dependency order and diffs against paper/reference/
├── results/                   # archived experiment outputs read by the generators
│   ├── e2e/                   #   exact decision suites, clarification, class ladder, coverage, ARTEMIS
│   ├── conformal/             #   learning arms (agnostic50, main50, ...), LP certificates (lp/), corpus manifests
│   ├── dsrl/                  #   published offline safe-RL learners on DSRL/OSRL
│   ├── safe_keep_final/, final_pipeline/   # SAFE-KEEP on 579 families and the fixed-data pipeline (Table 4)
│   ├── theory_extension/      #   offline certificate on the 28 rules (Section 5.3)
│   ├── paper_extra/, corpus_eval/, fragments/
├── data/rule_corpora/         # the third-party rule files the pipeline reads, at pinned commits (NOTICE.md inside)
├── paper/
│   ├── reference/             # the exact fragments (generated/) and figures (figure/) compiled into the PDF
│   ├── generated/, figure/    # outputs of scripts/reproduce_paper.py (ignored by git)
├── tests/                     # unit tests (construction, gridworld, offline learner, C-MAPSS)
└── requirements.txt
```

Naming: the method was called CORSET during development, so code, logs, and comments use
`corset`/`saorl` for what the paper calls ARROW, `Decide` for the ARROW decision (Algorithm 1),
`Refine` for question selection, `Protect` for the downstream constrained learner, and `Check`
for the fresh-evaluation safety test.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # numpy, scipy, matplotlib, PyYAML
export PYTHONPATH=src                  # the packages live in src/
python3 -m pytest tests -q             # 22 unit tests
```

Re-running the learning arms additionally needs `torch`, `scikit-learn`, and `pandas`.

## Reproducing the paper's tables and figures

```bash
python3 scripts/reproduce_paper.py
```

This runs the 30 generators in `scripts/paper/` in dependency order, writes the fragments to
`paper/generated/` and the figures to `paper/figure/`, and reports the comparison with
`paper/reference/`. At release time all 52 fragments are byte-identical and the six PDF
figures regenerate (PDFs embed a timestamp, so they are not byte-compared). Run a subset with
`python3 scripts/reproduce_paper.py --only make_gen_v13 make_figs_mpl`.

## Re-running experiments

Every archived result lists its driver in the tables below. Drivers run from the repository
root with `PYTHONPATH=src`. For example

```bash
python3 src/saorl/benchmark_sg/policy_sufficiency.py      # exact ARROW verdicts behind Tables 1 and 2
python3 src/saorl/benchmark_sg/safe_keep_final_run.py     # SAFE-KEEP on 579 families (Table 4), ~30 s
python3 src/saorl/benchmark_sg/final_pipeline_run.py      # fixed-data pipeline, 1,600 records (Table 4), ~30 s
python3 experiments/theory_extension/real_rules_experiment.py   # offline certificate on the 28 rules
```

The exact-LP experiments take seconds to minutes. The learning arms (`src/saorl/experiments.py`,
50 seeds per cell) and the DSRL wrappers take hours and need the `learners` extra; the C-MAPSS
replay needs the public NASA C-MAPSS files in `data/cmapss/` (not redistributed). The live agent
(`experiments/live_agent/`) bills a metered API and is archived rather than meant to be re-run.

## Map: main-text tables and figures → code

| Display | Fragment(s) compiled into the paper | Generator | Archived data | Experiment driver that produced the data |
|---|---|---|---|---|
| Table 1 (22 monitoring rules, tied optima) | `gen_v28.tex` (rows), `gen_v13.tex` (the 22 count) | `scripts/paper/make_gen_v28.py`, `make_gen_v13.py` | `results/e2e/safe_face_select.json`, `results/e2e/policy_sufficiency.json` | `src/saorl/benchmark_sg/safe_face_select.py` (default optimum, 100 random tie-breaks, ARROW's counterexample, full-set optimum), `src/saorl/benchmark_sg/policy_sufficiency.py` |
| Table 2 (28 rules, value test vs ARROW by budget) | `gen_v13.tex` | `scripts/paper/make_gen_v13.py` | `results/e2e/policy_sufficiency.json`, `results/e2e/face_live.json` | `src/saorl/benchmark_sg/policy_sufficiency.py`, `experiments/live_agent/face_live.py` |
| Table 3 panel A (violation rates, four domains) | `gen_e5_side.tex` (joined by `make_gen_e5_side.py` from `gen_learn_table.tex`) | `scripts/paper/make_gen_revision.py` | `results/conformal/lp/deploy_certificate.json`, `shadow_price.json`, `certificate_audit.json` | `src/saorl/deploy_certificate.py`, `src/saorl/shadow_price.py`, `src/saorl/certificate_audit.py` (learning arms from `src/saorl/experiments.py`) |
| Table 3 panel B (four constraint mechanisms) | `gen_e5_side.tex` (from `gen_e5b.tex` ← `gen_agnostic.tex`) | `scripts/paper/make_gen_e5b.py`, `scripts/paper/make_corset_tables.py` | `results/conformal/agnostic50/experiments_*.json`, `results/conformal/main50/experiments_*.json` | `src/saorl/experiments.py` (BCQ-FQI, CQL, PID-Lagrangian, CPQ arms) |
| Figure 1 panel A (safe stopping vs identification) | `fig_e3_questions.pdf` | `scripts/paper/make_figs_mpl.py` | `results/e2e/safe_collapse.json` | `src/saorl/benchmark_sg/safe_collapse.py` |
| Figure 1 panel B (return recovered by the first question) | `fig_e3_panelb.pdf` | `scripts/paper/make_figs_mpl.py` (reads `gen_query_table.tex` from `make_gen_applic.py`) | `results/e2e/query_loop.json` | `src/saorl/benchmark_sg/query_loop.py` |
| Table 4 (SAFE-KEEP on 579 families) and the fixed-data pipeline numbers in Section 5.3 | `gen_safekeep.tex` | `scripts/paper/make_gen_safekeep.py` | `results/safe_keep_final/safekeep_final.json`, `results/final_pipeline/pipeline_rows.json` | `src/saorl/benchmark_sg/safe_keep_final_run.py`, `src/saorl/benchmark_sg/final_pipeline_run.py` (library: `safe_keep.py`, `certified_protect.py`, `parse.py`, `candidates.py`) |
| Section 5.3 offline-certificate numbers (15,600 tests, 22/27 and 27/27 certified, slope 1.09, correlation 0.995) | `gen_finite.tex`, `gen_applic.tex` | `scripts/paper/make_gen_finite.py`, `make_gen_applic.py` | `results/theory_extension/real_rules_finite_data.json`, `real_rules_finite_data_records.json`, `real_rules_exact.json` | `experiments/theory_extension/real_rules_experiment.py` |

Numbers quoted in the running text come from the same macro files; the fragment table below
lists every fragment the paper inputs.

## Map: appendix tables and figures → code

| Display | Label | Fragment(s) / figure file | Generator script |
|---|---|---|---|
| Table 5 | `tab:runtime` (Every exact run, re-derived) | `gen_runtime_table.tex` | `scripts/paper/make_gen_revision.py` |
| Figure 2 | `fig:c1` (Candidate-set scaling of exact Decide) | `fig_c1_scale.pdf`, `gen_revision.tex` | `scripts/paper/make_figs_appendix.py`, `scripts/paper/make_gen_revision.py` |
| Table 6 | `tab:exact` (What single-reading optimization hides, solved exactly) | `gen_csuite_bind.tex` | `scripts/paper/make_gen_csuite_bind.py` |
| Table 7 | `tab:thirdcorpus` (Eight further organizations, same pipeline) | `gen_revision.tex`, `gen_third_corpus.tex` | `scripts/paper/make_gen_revision.py` |
| Table 8 | `tab:sweep` (Threshold sweep of the live crossed pool) | `gen_sweep_table.tex` | `scripts/paper/make_gen_revision.py` |
| Figure 3 | `fig:d1` (Return recovered by oracle-answered questions) | `fig_d1_query.pdf` | `scripts/paper/make_figs_appendix.py` |
| Table 9 | `tab:refine` (Safe collapse versus identification by question count) | `gen_v43_table.tex` | `scripts/paper/make_gen_v43.py` |
| Table 10 | `tab:e4` (Finite-data certification on the 28 compiled rules) | `gen_finite.tex`, `gen_finite_panelc.tex`, `gen_v13.tex` | `scripts/paper/make_gen_finite.py`, `scripts/paper/make_gen_v13.py` |
| Figure 4 | `fig:marginlaw` (Certification cost follows the margin law) | `fig_margin_scaling.tex` | `scripts/paper/make_fig_margin_scaling.py` |
| Table 11 | `tab:v48finite` (Finite-sample residue) | `gen_v48_finite.tex` | `scripts/paper/make_gen_v47_51_tables.py` |
| Table 12 | `tab:agnostic` (The effect is not an artifact of the inner optimizer) | `gen_agnostic.tex` | `scripts/paper/make_corset_tables.py` |
| Table 13 | `tab:dsrl` (Five published offline-safe-RL learners on DSRL/OSRL) | `gen_dsrl.tex` | `scripts/paper/make_corset_tables.py` |
| Table 14 | `tab:slack` (A live agent under a published rate limit) | transcribed from `experiments/live_agent/gen_demo.tex`, generated by `experiments/live_agent/make_table.py` from `experiments/live_agent/logs/eval_summary.json` | `experiments/live_agent/make_table.py` |
| Table 15 | `tab:sweepbudget` (Budget sweep: both arms re-learned at every budget) | `gen_sweep_budget.tex`, `gen_sweep_macros.tex` | `scripts/paper/make_gen_extra.py` |
| Figure 5 | `fig:e2` (Tail-level sweep: headline policies re-scored) | `fig_e2_tail.pdf`, `gen_sweep_macros.tex` | `scripts/paper/make_figs_appendix.py`, `scripts/paper/make_gen_extra.py` |
| Table 16 | `tab:e6` (Simple alternatives do not reproduce ARROW) | `gen_revision.tex`, `gen_v14.tex` | `scripts/paper/make_gen_revision.py`, `scripts/paper/make_gen_v14.py` |
| Table 17 | `tab:baselines` (Practical baselines at the operating budget) | `gen_baseline_table.tex`, `gen_revision.tex` | `scripts/paper/make_gen_revision.py` |
| Figure 6 | `fig:e1` (Exact price of the pointwise-maximum surrogate) | `fig_e1_surrogate.pdf`, `gen_v47_51.tex` | `scripts/paper/make_figs_appendix.py`, `scripts/paper/make_gen_v47_51.py` |
| Table 18 | `tab:selfcons` (Majority-voting several translations still commits to one reading) | `gen_selfcons_macros.tex`, `gen_selfcons_table.tex` | `scripts/paper/make_gen_extra.py` |
| Table 19 | `tab:pipeline-compare` (The sequential pipeline against ARROW) | `gen_v47_51.tex`, `gen_v47_pipeline.tex` | `scripts/paper/make_gen_v47_51.py`, `scripts/paper/make_gen_v47_51_tables.py` |
| Table 20 | `tab:positioning` (Positioning relative to prior work) | hand-written comparison of prior work (no data) | — |

## Every LaTeX fragment the paper inputs

| Fragment | Generator | Archived data read |
|---|---|---|
| `gen_agnostic.tex` | `scripts/paper/make_corset_tables.py` | `results/corpus_eval/` `bot_augmented_report.json`, `test_report_openai.json`; `paper/` `corset_iclr_appendix_short.tex`; `results/conformal/agnostic50/` `experiments_20260724-004227.json` … |
| `gen_applic.tex` | `scripts/paper/make_gen_applic.py` | `experiments/theory_extension/` `real_rules_finite_data.json`, `real_rules_finite_data_records.json`; `results/conformal/lp/` `evaluator_scale_50k.json`, `exposure_ceiling.json`; `results/e2e/` `artemis_units.json`, `query_loop.json` |
| `gen_artemis.tex` | `scripts/paper/make_gen_artemis.py` | `results/e2e/` `artemis_external.json`, `natural2ctl_external.json` |
| `gen_avail.tex` | `scripts/paper/make_gen_avail.py` | `results/corpus_eval/` `test_report_openai.json`; `results/conformal/e0/` `e0_report.json`; `results/conformal/lp/` `certified_at_scale.json`, `certified_at_scale_d05.json` … |
| `gen_baseline_table.tex` | `scripts/paper/make_gen_revision.py` | `results/conformal/lp/` (4 files); `results/conformal/` `selfconsistency.json`; `results/e2e/` (8 files) |
| `gen_certprice.tex` | `scripts/paper/make_corset_tables.py` | `results/corpus_eval/` `bot_augmented_report.json`, `test_report_openai.json`; `paper/` `corset_iclr_appendix_short.tex`; `results/conformal/agnostic50/` `experiments_20260724-004227.json` … |
| `gen_certscale_caption.tex` | `scripts/paper/make_corset_tables.py` | `results/corpus_eval/` `bot_augmented_report.json`, `test_report_openai.json`; `paper/` `corset_iclr_appendix_short.tex`; `results/conformal/agnostic50/` `experiments_20260724-004227.json` … |
| `gen_control_suite_caption.tex` | `scripts/paper/make_corset_tables.py` | `results/corpus_eval/` `bot_augmented_report.json`, `test_report_openai.json`; `paper/` `corset_iclr_appendix_short.tex`; `results/conformal/agnostic50/` `experiments_20260724-004227.json` … |
| `gen_cov.tex` | `scripts/paper/make_gen_cov.py` | `results/e2e/` `coverage_anatomy.json`, `e2e_report_g1.json`, `e2e_report_g1_mondrian.json` |
| `gen_csuite_bind.tex` | `scripts/paper/make_gen_csuite_bind.py` |  |
| `gen_dsrl.tex` | `scripts/paper/make_corset_tables.py` | `results/corpus_eval/` `bot_augmented_report.json`, `test_report_openai.json`; `paper/` `corset_iclr_appendix_short.tex`; `results/conformal/agnostic50/` `experiments_20260724-004227.json` … |
| `gen_dsrl_seedinfo.tex` | `scripts/paper/make_corset_tables.py` | `results/corpus_eval/` `bot_augmented_report.json`, `test_report_openai.json`; `paper/` `corset_iclr_appendix_short.tex`; `results/conformal/agnostic50/` `experiments_20260724-004227.json` … |
| `gen_dsrlcount.tex` | `scripts/paper/make_corset_tables.py` | `results/corpus_eval/` `bot_augmented_report.json`, `test_report_openai.json`; `paper/` `corset_iclr_appendix_short.tex`; `results/conformal/agnostic50/` `experiments_20260724-004227.json` … |
| `gen_e0_caption.tex` | `scripts/paper/make_corset_tables.py` | `results/corpus_eval/` `bot_augmented_report.json`, `test_report_openai.json`; `paper/` `corset_iclr_appendix_short.tex`; `results/conformal/agnostic50/` `experiments_20260724-004227.json` … |
| `gen_e2e.tex` | `scripts/paper/make_gen_e2e.py` | `results/e2e/catalog_v4/` (7 files); `results/e2e/` (7 files) |
| `gen_e5_prose.tex` | `scripts/paper/make_gen_e5_prose.py` |  |
| `gen_e5_side.tex` | `scripts/paper/make_gen_e5_side.py` |  |
| `gen_e9_margin.tex` | `scripts/paper/make_corset_tables.py` | `results/corpus_eval/` `bot_augmented_report.json`, `test_report_openai.json`; `paper/` `corset_iclr_appendix_short.tex`; `results/conformal/agnostic50/` `experiments_20260724-004227.json` … |
| `gen_extension_caption.tex` | `scripts/paper/make_corset_tables.py` | `results/corpus_eval/` `bot_augmented_report.json`, `test_report_openai.json`; `paper/` `corset_iclr_appendix_short.tex`; `results/conformal/agnostic50/` `experiments_20260724-004227.json` … |
| `gen_finite.tex` | `scripts/paper/make_gen_finite.py` | `experiments/theory_extension/` `real_rules_exact.json`, `real_rules_finite_data.json`, `real_rules_finite_data_records.json`; `results/e2e/` `control_suite_uncapped.json` |
| `gen_finite_panelc.tex` | `scripts/paper/make_gen_finite.py` | `experiments/theory_extension/` `real_rules_exact.json`, `real_rules_finite_data.json`, `real_rules_finite_data_records.json`; `results/e2e/` `control_suite_uncapped.json` |
| `gen_frontier_macros.tex` | `scripts/paper/make_gen_extra.py` | `results/conformal/` `selfconsistency.json`; `results/e2e/` `e2e_report_v4.json`; `results/paper_extra/pareto/` `pareto_risk_20260724-013218.json` |
| `gen_funnel.tex` | `scripts/paper/make_gen_extra.py` | `results/conformal/` `selfconsistency.json`; `results/e2e/` `e2e_report_v4.json`; `results/paper_extra/pareto/` `pareto_risk_20260724-013218.json` |
| `gen_ladder.tex` | `scripts/paper/make_gen_ladder.py` | `results/e2e/` `class_ladder.json` |
| `gen_livecross.tex` | `scripts/paper/make_gen_livecross.py` | `results/e2e/` `live_crossing_keep.json`, `live_crossing_keep_stability.json`, `live_crossing_replay.json` |
| `gen_lmbase_caption.tex` | `scripts/paper/make_corset_tables.py` | `results/corpus_eval/` `bot_augmented_report.json`, `test_report_openai.json`; `paper/` `corset_iclr_appendix_short.tex`; `results/conformal/agnostic50/` `experiments_20260724-004227.json` … |
| `gen_matchedsize_macros.tex` | `scripts/paper/make_corset_tables.py` | `results/corpus_eval/` `bot_augmented_report.json`, `test_report_openai.json`; `paper/` `corset_iclr_appendix_short.tex`; `results/conformal/agnostic50/` `experiments_20260724-004227.json` … |
| `gen_prov.tex` | `scripts/paper/make_gen_prov.py` | `results/e2e/` (8 files) |
| `gen_r6.tex` | `scripts/paper/make_gen_r6.py` | `experiments/live_agent/` `ensemble_scores_runcaps_stability.json`; `results/conformal/benchmark_sg/_src/kyverno-policies/argo-cel/application-field-validation/` `application-field-validation.yaml`; `results/conformal/benchmark_sg/_src/kyverno-policies/argo-cel/application-prevent-default-project/` `application-prevent-default-project.yaml` … |
| `gen_revision.tex` | `scripts/paper/make_gen_revision.py` | `results/conformal/lp/` (4 files); `results/conformal/` `selfconsistency.json`; `results/e2e/` (8 files) |
| `gen_runtime_table.tex` | `scripts/paper/make_gen_revision.py` | `results/conformal/lp/` (4 files); `results/conformal/` `selfconsistency.json`; `results/e2e/` (8 files) |
| `gen_safekeep.tex` | `scripts/paper/make_gen_safekeep.py` | `results/final_pipeline/` `pipeline_rows.json`; `results/safe_keep_final/` `safekeep_final.json` |
| `gen_screencost.tex` | `scripts/paper/make_corset_tables.py` | `results/corpus_eval/` `bot_augmented_report.json`, `test_report_openai.json`; `paper/` `corset_iclr_appendix_short.tex`; `results/conformal/agnostic50/` `experiments_20260724-004227.json` … |
| `gen_selfcons_macros.tex` | `scripts/paper/make_gen_extra.py` | `results/conformal/` `selfconsistency.json`; `results/e2e/` `e2e_report_v4.json`; `results/paper_extra/pareto/` `pareto_risk_20260724-013218.json` |
| `gen_selfcons_table.tex` | `scripts/paper/make_gen_extra.py` | `results/conformal/` `selfconsistency.json`; `results/e2e/` `e2e_report_v4.json`; `results/paper_extra/pareto/` `pareto_risk_20260724-013218.json` |
| `gen_shadow_caption.tex` | `scripts/paper/make_corset_tables.py` | `results/corpus_eval/` `bot_augmented_report.json`, `test_report_openai.json`; `paper/` `corset_iclr_appendix_short.tex`; `results/conformal/agnostic50/` `experiments_20260724-004227.json` … |
| `gen_sweep_budget.tex` | `scripts/paper/make_gen_extra.py` | `results/conformal/` `selfconsistency.json`; `results/e2e/` `e2e_report_v4.json`; `results/paper_extra/pareto/` `pareto_risk_20260724-013218.json` |
| `gen_sweep_macros.tex` | `scripts/paper/make_gen_extra.py` | `results/conformal/` `selfconsistency.json`; `results/e2e/` `e2e_report_v4.json`; `results/paper_extra/pareto/` `pareto_risk_20260724-013218.json` |
| `gen_sweep_table.tex` | `scripts/paper/make_gen_revision.py` | `results/conformal/lp/` (4 files); `results/conformal/` `selfconsistency.json`; `results/e2e/` (8 files) |
| `gen_third_corpus.tex` | `scripts/paper/make_gen_revision.py` | `results/conformal/lp/` (4 files); `results/conformal/` `selfconsistency.json`; `results/e2e/` (8 files) |
| `gen_unified_numbers.tex` | `scripts/paper/make_unified_numbers.py` | `results/conformal/benchmark_sg/` `control_suite.json`, `report.json`, `report_extension.json`; `results/conformal/e0/` `e0_report.json`; `results/conformal/lp/` (4 files) … |
| `gen_v11b.tex` | `scripts/paper/make_gen_v11b.py` | `results/e2e/` (4 files) |
| `gen_v13.tex` | `scripts/paper/make_gen_v13.py` | `results/e2e/` `face_live.json`, `policy_sufficiency.json` |
| `gen_v14.tex` | `scripts/paper/make_gen_v14.py` | `results/e2e/` `face_ladder.json` |
| `gen_v28.tex` | `scripts/paper/make_gen_v28.py` | `results/e2e/` `policy_sufficiency.json`, `safe_face_select.json` |
| `gen_v41.tex` | `scripts/paper/make_gen_v41.py` | `results/e2e/` `decision_equivalence_readout.json`, `k_scaling.json` |
| `gen_v43.tex` | `scripts/paper/make_gen_v43.py` | `results/e2e/` `basis_size.json`, `learner_slack.json`, `safe_collapse.json` |
| `gen_v43_table.tex` | `scripts/paper/make_gen_v43.py` | `results/e2e/` `basis_size.json`, `learner_slack.json`, `safe_collapse.json` |
| `gen_v47_51.tex` | `scripts/paper/make_gen_v47_51.py` | `results/e2e/` (10 files) |
| `gen_v47_pipeline.tex` | `scripts/paper/make_gen_v47_51_tables.py` | `results/e2e/` (7 files) |
| `gen_v48_finite.tex` | `scripts/paper/make_gen_v47_51_tables.py` | `results/e2e/` (7 files) |

## Archived results → experiment scripts

| Archived result | Produced by |
|---|---|
| `policy_sufficiency.json` | `src/saorl/benchmark_sg/policy_sufficiency.py`, `src/saorl/benchmark_sg/safe_face_select.py`, `src/saorl/benchmark_sg/baseline_table.py` |
| `safe_face_select.json` | `src/saorl/benchmark_sg/safe_face_offline.py`, `src/saorl/benchmark_sg/occupancy_arrow.py`, `src/saorl/benchmark_sg/certified_safe_face.py` |
| `face_live.json` | `experiments/live_agent/face_live.py` |
| `face_ladder.json` | `src/saorl/benchmark_sg/face_ladder.py`, `src/saorl/benchmark_sg/decide_runtime.py` |
| `deploy_certificate.json` | `src/saorl/evaluator_scale.py`, `src/saorl/exposure_ceiling.py`, `src/saorl/certificate_audit.py` |
| `shadow_price.json` | `src/saorl/shadow_price.py` |
| `certificate_audit.json` | `src/saorl/certificate_audit.py`, `src/saorl/benchmark_sg/pipeline_compare.py` |
| `baseline_table.json` | `src/saorl/benchmark_sg/baseline_table.py`, `src/saorl/benchmark_sg/pipeline_compare.py` |
| `surrogate_price.json` | `src/saorl/benchmark_sg/collapse_readout.py`, `src/saorl/benchmark_sg/surrogate_price.py`, `src/saorl/benchmark_sg/decide_runtime.py` |
| `safe_collapse.json` | `src/saorl/benchmark_sg/safe_collapse.py` |
| `query_loop.json` | `src/saorl/benchmark_sg/query_loop.py` |
| `real_rules_finite_data.json` | `experiments/theory_extension/real_rules_experiment.py` |
| `real_rules_finite_data_records.json` | `experiments/theory_extension/real_rules_experiment.py` |
| `real_rules_exact.json` | `experiments/theory_extension/real_rules_experiment.py` |
| `control_suite_uncapped.json` | `src/saorl/benchmark_sg/semantic_risk_frontier.py`, `experiments/theory_extension/delta_vs_gamma.py`, `experiments/theory_extension/real_rules_experiment.py` |
| `class_ladder.json` | `src/saorl/benchmark_sg/class_ladder.py`, `src/saorl/benchmark_sg/faithfulness_provenance.py`, `src/saorl/benchmark_sg/decide_runtime.py` |
| `k_scaling.json` | `src/saorl/benchmark_sg/k_scaling.py` |
| `decision_equivalence_readout.json` | `src/corset_e2e/analysis/decision_equivalence_readout.py` |
| `learner_slack.json` | `src/saorl/benchmark_sg/learner_slack.py` |
| `basis_size.json` | `src/saorl/benchmark_sg/basis_size.py` |
| `coverage_funnel.json` | `src/corset_e2e/analysis/coverage_funnel.py` |
| `third_corpus.json` | `src/saorl/benchmark_sg/third_corpus.py` |
| `live_threshold_sweep.json` | `experiments/live_agent/threshold_sweep.py` |
| `evaluator_scale.json` | `src/saorl/evaluator_scale.py` |
| `decide_runtime.json` | `src/saorl/benchmark_sg/decide_runtime.py` |
| `price_instances.json` | `src/saorl/benchmark_sg/price_instances.py` |
| `v4_recall.json` | `src/corset_e2e/analysis/dev_miss_autopsy.py` |
| `control_suite.json` | `src/saorl/e10_real_game.py`, `src/saorl/t31_switchgap.py`, `src/saorl/gate1_screen.py` |
| `e0_report.json` | `src/saorl/benchmark_sg/e0_run.py` |
| `artemis_units.json` | `src/corset_e2e/external/artemis_decide.py`, `src/corset_e2e/external/artemis_arrow.py`, `src/corset_e2e/external/artemis_load.py` |
| `artemis_external.json` | `src/corset_e2e/external/artemis_decide.py`, `src/corset_e2e/external/artemis_arrow.py` |
| `gold_ast.json` | `src/corset_e2e/analysis/run_v16_full.py`, `src/corset_e2e/analysis/emission_gap_g2.py`, `src/corset_e2e/analysis/derive_g2_from_dev.py` |
| `pipeline_compare.json` | `src/saorl/benchmark_sg/pipeline_compare.py` |
| `collapse_utility.json` | `src/saorl/benchmark_sg/collapse_readout.py`, `src/saorl/benchmark_sg/collapse_utility.py` |
| `answer_noise_check.json` | `src/saorl/benchmark_sg/answer_noise_frontier.py`, `src/saorl/benchmark_sg/answer_noise_check.py` |
| `union_calibrated_v51.json` | `src/corset_e2e/analysis/union_calibrated_v51.py` |
| `selfconsistency.json` | `src/saorl/baseline_llm_selfconsistency.py` |
| `guarantee_accounting.json` | `src/corset_e2e/analysis/guarantee_accounting.py` |
| `faithful_recalibration.json` | `src/corset_e2e/analysis/faithful_recalibration.py` |
| `live_crossing_replay.json` | `experiments/live_agent/replay_crossing.py`, `experiments/live_agent/face_live.py` |
| `coverage_anatomy.json` | `src/corset_e2e/analysis/coverage_anatomy.py` |
| `downstream_provenance.json` | `src/saorl/benchmark_sg/faithfulness_provenance.py`, `src/corset_e2e/analysis/guarantee_accounting.py` |
| `exposure_ceiling.json` | `src/saorl/exposure_ceiling.py` |
| `evaluator_scale_50k.json` | `src/saorl/exposure_ceiling.py` |
| `results/safe_keep_final/safekeep_final.json` | `src/saorl/benchmark_sg/safe_keep_final_run.py` (recovered driver, re-run reproduces the archive up to the runtime field) |
| `results/final_pipeline/pipeline_rows.json` | `src/saorl/benchmark_sg/final_pipeline_run.py` (recovered driver, re-run reproduces all 1,600 rows within LP tolerance and yields identical paper macros) |
| `results/conformal/agnostic50/`, `main50/`, `budget50/`, `cmapss4/`, `risk50/` … | `src/saorl/experiments.py` (learning arms, 50 seeds per cell) |
| `results/dsrl/**` | `src/saorl/dsrl_*.py` wrappers around the published offline safe-RL learners |

