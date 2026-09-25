# Safe RL under ambiguous written rules — code, archived results, and paper generators

Code and data behind *When Is One Reading Enough? Decision-Relevant Ambiguity in Constrained
Reinforcement Learning* (ARROW). Every number, table, and figure in the paper is produced by a
script in this repository from an archived result file under `results/` or `experiments/`;
`scripts/reproduce_paper.py` regenerates all of them and checks them byte-for-byte against the
fragments compiled into the submitted PDF.

The repository ships what the submitted paper reports and the code that produced it. Studies that
earlier drafts reported and the paper no longer does (the CORSET-era learning arms, the generic
certificate variants, the coverage and taxonomy analyses of the open-domain generator, the
tail-level sweep, the Natural2CTL study) were removed together with their archives and generators.

## Repository layout

```
.
├── src/
│   ├── saorl/                 # the library: monitoring controller and occupancy LPs (exact_lp.py,
│   │   │                      #   benchmark_sg/control_mdp.py, control_suite.py), the ARROW decision
│   │   │                      #   (benchmark_sg/policy_sufficiency.py), SAFE-KEEP (benchmark_sg/safe_keep.py),
│   │   │                      #   the certified fixed-data planner (benchmark_sg/certified_protect.py),
│   │   │                      #   clarification studies (query_loop.py, safe_collapse.py), the agent-service
│   │   │                      #   domain and its learners (benchmark_sg/scope_agent*.py), rule parsing and
│   │   │                      #   corpus tooling (benchmark_sg/parse.py, candidates.py, fetch.py), the learning arms behind
│   │   │                      #   the budget sweep (experiments.py, pareto_risk.py, budget_rl.py, cmapss*.py)
│   │   │                      #   and every experiment driver the tables below name
│   │   └── benchmark_sg/
│   └── corset_e2e/            # open-domain generation pipeline: candidate generation, calibration and the
│                              #   analyses the paper still reports, the ARTEMIS external study (external/artemis_*.py)
├── experiments/
│   ├── live_agent/            # the tool-using agent under a published rate limit (Appendix D.1.2): tasks, corpus,
│   │                          #   guards, transcripts and logs of the 50 paired billed sessions per condition
│   ├── theory_extension/      # offline certificates on the 28 compiled rules: real_rules_experiment.py
│   │                          #   (the archived generic certificate sweep) and certificate_v53.py (registered
│   │                          #   V53 study: generic uniform, occupancy-weighted and robust-dual certificates on
│   │                          #   identical draws, and the witness-based information floor; Table 14)
│   └── dsrl_learners/         # wrappers that ran the published offline safe-RL learners (results/dsrl/, Table 19)
├── scripts/
│   ├── paper/                 # 26 generators (make_gen_*.py, make_fig*.py, make_corset_tables.py,
│   │                          #   make_unified_numbers.py): archived results -> LaTeX fragments and figures
│   └── reproduce_paper.py     # runs them all in dependency order and diffs against paper/reference/
├── results/                   # archived experiment outputs read by the generators
│   ├── e2e/                   #   exact decision suites, clarification, class ladder, the third corpus, ARTEMIS,
│   │                          #   the agent-service archives (scope_agent*.json), the generation-pipeline reports the
│   │                          #   remaining analyses read, and the pre-registration notes REGISTRATION_V17, V18,
│   │                          #   V23, V50-V53
│   ├── compiler_audit/        #   compiler-faithfulness audit behind SAFE-KEEP's implication relation
│   ├── conformal/             #   the compiled-suite reports (benchmark_sg/), the E0 corpus study (e0/), the LP
│   │                          #   certificates the headline macros read (lp/), the price report (price/) and the three
│   │                          #   learning-arm archives (risk50/, budget50/, main50/) that the certificate drivers read
│   ├── dsrl/                  #   published offline safe-RL learners on DSRL/OSRL, one JSON per run
│   ├── safe_keep_final/, final_pipeline/   # SAFE-KEEP on 579 families and the fixed-data pipeline (Table 15)
│   ├── theory_extension/      #   offline certificates on the 28 rules: the archived generic sweep and
│   │                          #   certificate_v53.json (Table 14), with their logs
│   ├── paper_extra/pareto/    #   the maintenance budget sweep (Table 12)
│   └── fragments/
├── data/rule_corpora/         # the third-party rule files the pipeline reads, at pinned commits (NOTICE.md inside)
├── paper/
│   ├── reference/             # the 36 fragments (generated/, 33 input by the paper) and 3 figures (figure/) compiled into the PDF
│   └── generated/, figure/    # outputs of scripts/reproduce_paper.py (ignored by git)
├── tests/                     # unit tests (construction, gridworld, offline learner, C-MAPSS)
└── requirements.txt
```

Naming: the method was called CORSET during development, so code, logs, and comments use
`corset`/`saorl` for what the paper calls ARROW, `Decide` for the ARROW decision (lines 7-23 of
Algorithm 1), `Refine` for question selection, `Protect` for the downstream constrained learner,
and `Check` for the fresh-evaluation safety test.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # numpy, scipy, matplotlib, PyYAML
export PYTHONPATH=src                  # the packages live in src/
python3 -m pytest tests -q             # 22 unit tests; the three C-MAPSS tests skip unless data/cmapss/ is present
```

Re-running the learning arms additionally needs `torch`, `scikit-learn`, and `pandas`.

## Reproducing the paper's tables and figures

```bash
python3 scripts/reproduce_paper.py
```

This runs the 26 generators in `scripts/paper/` in dependency order, writes the fragments to
`paper/generated/` and the figures to `paper/figure/`, and reports the comparison with
`paper/reference/`. At release time all 36 archived fragments are byte-identical and the
three PDF figures regenerate (PDFs embed a timestamp, so they are not byte-compared); the whole
run takes about 20 seconds. Run a subset with
`python3 scripts/reproduce_paper.py --only make_gen_v53 make_fig_v53`.

## Re-running experiments

Every archived result lists its driver in the tables below. Drivers run from the repository
root with `PYTHONPATH=src`. For example

```bash
python3 src/saorl/benchmark_sg/policy_sufficiency.py        # exact ARROW verdicts behind Table 1
python3 src/saorl/benchmark_sg/safe_keep_final_run.py       # SAFE-KEEP on 579 families (Table 15), ~30 s
python3 src/saorl/benchmark_sg/final_pipeline_run.py        # fixed-data pipeline, 1,600 records (Table 15), ~30 s
python3 experiments/theory_extension/certificate_v53.py     # three offline certificates on the 28 rules (Table 14), ~4 min
python3 src/saorl/benchmark_sg/scope_agent.py               # exact surface of the agent-service family (Table 2, Tables 14 and 17)
python3 src/corset_e2e/external/artemis_per_method.py       # every archived ARTEMIS method on its own samples (Table 3)
```

The exact-LP experiments take seconds to minutes. The learning arms (`src/saorl/pareto_risk.py`
for the budget sweep, and `src/saorl/benchmark_sg/scope_agent_learn.py`, `scope_agent_native.py`,
`scope_agent_check.py` for the agent-service learners) and the DSRL wrappers take hours and need
the `learners` extra; the C-MAPSS replay needs the public NASA C-MAPSS files in `data/cmapss/`
(not redistributed); the ARTEMIS scripts need the ARTEMIS artifact in `external_data/artemis/`
(or `ARTEMIS_DIR`), also not redistributed. The live agent (`experiments/live_agent/`) bills a
metered API and is archived rather than meant to be re-run.

## Map: main-text tables and figures → code

Page numbers refer to the submitted PDF.

| Display | Page | What it reports | Generator | Archived data |
|---|---|---|---|---|
| **Table 1** | 8 | Number of the 28 compiled monitoring rules judged reducible to one reading by the value test and by ARROW, per budget | `scripts/paper/make_gen_v13.py` | `results/e2e/policy_sufficiency.json` |
| **Figure 1** | 8 | Clarification progress on 87 service-monitoring candidate sets | `scripts/paper/make_figs_mpl.py` (`fig_e3_questions.pdf`) | `results/e2e/safe_collapse.json` |
| **Table 2** | 8 | Return as a share of the exact full-set optimum on the agent-service benchmark: combined conservative cost versus the sufficient reading | `scripts/paper/make_gen_v50.py` | `results/e2e/scope_agent.json`, `results/e2e/scope_agent_learn.json` |
| **Table 3** | 9 | Plausible readings retained on 182 ARTEMIS requirements: nine language-to-specification methods and three selection rules on the union pool | `scripts/paper/make_gen_artemis_methods.py` (`gen_artemis_methods_table_short.tex`), `scripts/paper/make_gen_artemis.py` | `results/e2e/artemis_per_method.json`, `results/e2e/artemis_external.json` |

## Map: appendix tables, figures and algorithm → code

| Display | Page | Appendix | What it reports | Generator | Archived data |
|---|---|---|---|---|---|
| **Table 4** | 17 | A | Positioning by problem solved, safety input and guarantee | hand-written | -- |
| **Table 5** | 19 | B.1 | Data sources and populations (external sources and our three constructions, with the results each feeds) | hand-written; counts from the fragment macros | see the fragment rows below |
| **Table 6** | 19 | B.2 | Decision models: what each reading charges, horizon, budget and policy class | hand-written | -- |
| **Algorithm 1** | 21 | B.3 | End-to-end pipeline with ARROW inlined (lines 7-23) | hand-written | -- |
| **Table 7** | 22 | B.3 | Protocols of the exact and fixed-data studies (population, settings, repeats) | hand-written; counts from the fragment macros | see the fragment rows below |
| **Table 8** | 23 | B.3 | Learner settings of the agent-service study (one rule per setting, shared by all learners and both arms) | hand-written; two counts from `scripts/paper/make_gen_v50.py` | `results/e2e/scope_agent_learn.json` |
| **Table 9** | 37 | D | Appendix roadmap (question, evidence, theory link, what it establishes) | hand-written; one count from `scripts/paper/make_gen_v52.py` | `results/e2e/scope_agent_check.json` |
| **Table 10** | 38 | D.1 | What single-reading optimization hides, solved exactly (17 of the 28 compiled rules) | `scripts/paper/make_gen_csuite_bind.py` joining fragments of `scripts/paper/make_corset_tables.py` | `results/conformal/benchmark_sg/control_suite.json`, `results/e2e/exact_nonnested.json` |
| **Table 11** | 38 | D.1 | Simple alternatives address different parts of the problem | `scripts/paper/make_gen_revision.py`, `scripts/paper/make_gen_v14.py`, `scripts/paper/make_gen_sota.py` | `results/e2e/baseline_table.json`, `results/e2e/face_ladder.json` |
| **Table 12** | 39 | D.1 | Budget sweep on the maintenance domains: both policies re-learned at every budget | `scripts/paper/make_gen_extra.py` | `results/paper_extra/pareto/pareto_risk_*.json` |
| **Table 13** | 39 | D.1 | Stopping versus identification by question count (hyperedge cutting at its FREE stop, the SAFE-stop exception in the caption) | `scripts/paper/make_gen_v43.py` | `results/e2e/safe_collapse.json`, `results/e2e/basis_size.json`, `results/e2e/learner_slack.json` |
| **Table 14** | 40 | D.2 | Three data scales per sufficient-reading class: n* of each certificate and the witness-based information floor kl(1-delta,delta)/Ibar, Ibar the divergence of the positive-margin witness chain and an upper bound on I*, a valid lower bound for delta-sound procedures that certify under the true chain with probability at least 1-delta | `scripts/paper/make_gen_v53.py` (`gen_v53_classes.tex`) | `results/theory_extension/certificate_v53.json` |
| **Figure 2** | 41 | D.2 | Generic certificate, decision-specific certificate and information floor against 1/kappa^2 for the six budget-margin classes | `scripts/paper/make_fig_v53.py` (`fig_v53_scales.pdf`) | `results/theory_extension/certificate_v53.json` |
| **Table 15** | 41 | D.2 | Safe-Keep audit: uncovered removals (Score-Keep audited under the same relation), retained-set sizes, implications confirmed | `scripts/paper/make_gen_safekeep.py` | `results/safe_keep_final/safekeep_final.json`, `results/safe_keep_final/score_keep_cover.json`, `results/final_pipeline/pipeline_rows.json` |
| **Figure 3** | 42 | D.3 | Exact price of the single-cost surrogate by budget, median-to-maximum band | `scripts/paper/make_figs_appendix.py` (`fig_e1_surrogate.pdf`) | `results/e2e/collapse_readout.json` |
| **Table 16** | 43 | D.3 | Exact price of the single-cost surrogate by budget (median, mean, p90, max, counts above 1%, 2%, 5%) | `scripts/paper/make_gen_v47_51_tables.py` (`gen_v48_surrogate.tex`) | `results/e2e/collapse_readout.json` |
| **Table 17** | 43 | D.3 | Per-instance gains of the ARROW-selected reading over the single-cost surrogate (9 sufficient instances, 6 learners) | `scripts/paper/make_gen_v50.py` (`gen_v50_perinst.tex`) | `results/e2e/scope_agent.json`, `results/e2e/scope_agent_learn.json` |
| **Table 18** | 44 | D.3 | Native multi-constraint learners: K separate constraints versus the sufficient reading | `scripts/paper/make_gen_v51.py` | `results/e2e/scope_agent_native.json`, `results/e2e/scope_agent_native_smalln.json` |
| **Table 19** | 44 | D.3 | Portability across five released offline safe-RL learners on DSRL/OSRL (**GPU**: V100) | `scripts/paper/make_corset_tables.py` (`gen_dsrl.tex`, `gen_dsrl_seedinfo.tex`) | `results/dsrl/` |
| **Table 20** | 45 | D.3 | Checked deployment on all learned agent-service policies: per learner, unsafe and pass counts under each training cost and unsafe deployments, plus the pooled wrong-reading policies | `scripts/paper/make_gen_v52.py`, `scripts/paper/make_gen_v50.py` | `results/e2e/scope_agent_check.json`, `results/e2e/scope_agent_learn.json` |
| **Table 21** | 45 | D.4 | Expert-plausible readings preserved (union pool of every generator's archived translations) | `scripts/paper/make_gen_artemis.py` | `results/e2e/artemis_external.json` |

Table 19 is the one GPU result (five released offline safe-RL learners on DSRL/OSRL, run on V100s
through the wrappers in `experiments/dsrl_learners/`); everything else was produced on CPU.
Results reported in the appendix text rather than in a table, with their generators: the data
sources (Appendix B.1), the eight-repository replication totals (`make_gen_revision.py`: `gen_revision.tex` for the
totals, `gen_third_corpus_compact.tex` for the per-repository rows), the live service-agent check (`make_gen_live.py`,
`gen_live.tex`), the certificate-versus-log-size sweep (`make_gen_v53.py`, `gen_v53_table.tex`)
and the earlier-setting sensitivity runs of the learning study (`make_gen_v50.py` from the archived
`scope_agent_learn_*.json`). The appendix figures (`make_fig_v53.py`, `make_figs_appendix.py`) are Figures 2 and 3; their
numbers are also Tables 14 and 16.

## Every LaTeX fragment the paper inputs

The 36 files under `paper/reference/generated/` are the `\input` and `\tblinput` targets of
the paper source, 33 of them input by the current version (the table bodies `gen_third_corpus_compact.tex`,
`gen_live_agent.tex` and `gen_v53_table.tex` remain generated and archived; their totals are quoted in
the appendix text through the macro files). Macro files carry the numbers the running text and captions print; table files are
row bodies. The "archived data read" column was recorded by tracing every file each generator opens.

| Fragment | Generator | Archived data read |
|---|---|---|
| `gen_applic.tex` | `scripts/paper/make_gen_applic.py` | `results/e2e/` `artemis_units.json`, `query_loop.json` |
| `gen_artemis.tex` | `scripts/paper/make_gen_artemis.py` | `results/e2e/` `artemis_external.json` |
| `gen_artemis_decide.tex` | `scripts/paper/make_gen_artemis_decide.py` | `results/e2e/` `artemis_decide.json` |
| `gen_artemis_methods.tex` | `scripts/paper/make_gen_artemis_methods.py` | `results/e2e/` `artemis_per_method.json` |
| `gen_artemis_methods_table_short.tex` | `scripts/paper/make_gen_artemis_methods.py` | `results/e2e/` `artemis_per_method.json` |
| `gen_control_suite_caption.tex` | `scripts/paper/make_corset_tables.py` | `results/conformal/benchmark_sg/` `control_suite.json` |
| `gen_csuite_bind.tex` | `scripts/paper/make_gen_csuite_bind.py` | joins `gen_control_suite.tex` and `gen_bind.tex`, both from `make_corset_tables.py` (`results/conformal/benchmark_sg/control_suite.json`, `results/e2e/exact_nonnested.json`) |
| `gen_dsrl.tex` | `scripts/paper/make_corset_tables.py` | `results/dsrl/**` (every per-run JSON file of the released-learner sweeps and top-ups) |
| `gen_dsrl_seedinfo.tex` | `scripts/paper/make_corset_tables.py` | `results/dsrl/**` |
| `gen_dsrlcount.tex` | `scripts/paper/make_corset_tables.py` | `results/dsrl/**` (the count of per-run JSON files) |
| `gen_finite.tex` | `scripts/paper/make_gen_finite.py` | `results/e2e/` `control_suite_uncapped.json`; `results/theory_extension/` `real_rules_exact.json`, `real_rules_finite_data.json`, `real_rules_finite_data_records.json` |
| `gen_live.tex` | `scripts/paper/make_gen_live.py` | `experiments/live_agent/logs/` `eval_corset.jsonl`, `eval_single.jsonl`, `eval_summary.json` |
| `gen_live_agent.tex` | `scripts/paper/make_gen_live.py` | `experiments/live_agent/logs/` `eval_corset.jsonl`, `eval_single.jsonl`, `eval_summary.json` |
| `gen_r6.tex` | `scripts/paper/make_gen_r6.py` | `data/rule_corpora/kyverno-policies/**` (471 policy files); `results/conformal/benchmark_sg/` `calib_vs_fixed.json`, `fetch_manifest.json`, `report.json`; `results/e2e/` `surrogate_price.json`; `experiments/live_agent/` `ensemble_scores_runcaps_stability.json` |
| `gen_revision.tex` | `scripts/paper/make_gen_revision.py` | `results/e2e/` `baseline_table.json`, `decide_runtime.json`, `surrogate_price.json`, `third_corpus.json` |
| `gen_safekeep.tex` | `scripts/paper/make_gen_safekeep.py` | `results/final_pipeline/` `pipeline_rows.json`; `results/safe_keep_final/` `safekeep_final.json`, `score_keep_cover.json` |
| `gen_sota.tex` | `scripts/paper/make_gen_sota.py` | `results/e2e/` `baseline_table.json` |
| `gen_sweep_budget.tex` | `scripts/paper/make_gen_extra.py` | `results/paper_extra/pareto/` `pareto_risk_20260724-013218.json` |
| `gen_sweep_macros.tex` | `scripts/paper/make_gen_extra.py` | `results/paper_extra/pareto/` `pareto_risk_20260724-013218.json` |
| `gen_third_corpus_compact.tex` | `scripts/paper/make_gen_revision.py` | `results/e2e/` `baseline_table.json`, `decide_runtime.json`, `surrogate_price.json`, `third_corpus.json` |
| `gen_unified_numbers.tex` | `scripts/paper/make_unified_numbers.py` | `results/conformal/benchmark_sg/` `control_suite.json`, `report.json`, `report_extension.json`; `results/conformal/e0/` `e0_report.json`; `results/conformal/lp/` `deploy_certificate.json`, `expected_cost_certificate.json`, `shadow_price.json`, `switch_gap_exact.json`; `results/e2e/` `antichain_reduction.json`, `e16_exact_screen.json`, `e2e_report_v4.json`, `e4_screen.json`, `exact_nonnested.json`, `policy_class_budget.json`, `screen_cost.json`, `v6_calibrated.json`, `w4_neutral_grids.json`, `w7_bridge_analysis.json` |
| `gen_v13.tex` | `scripts/paper/make_gen_v13.py` | `results/e2e/` `face_live.json`, `policy_sufficiency.json` |
| `gen_v14.tex` | `scripts/paper/make_gen_v14.py` | `results/e2e/` `face_ladder.json` |
| `gen_v43.tex` | `scripts/paper/make_gen_v43.py` | `results/e2e/` `basis_size.json`, `learner_slack.json`, `safe_collapse.json` |
| `gen_v43_table.tex` | `scripts/paper/make_gen_v43.py` | `results/e2e/` `basis_size.json`, `learner_slack.json`, `safe_collapse.json` |
| `gen_v47_51.tex` | `scripts/paper/make_gen_v47_51.py` | `results/e2e/` `answer_noise_check.json`, `collapse_readout.json`, `collapse_utility.json`, `coverage_funnel.json`, `e2e_test_rows_v11_complete.json`, `guarantee_accounting.json`, `pipeline_compare.json`, `pipeline_compare_delta.json`, `union_calibrated_v51.json`, `v16_failure_taxonomy.json` |
| `gen_v48_surrogate.tex` | `scripts/paper/make_gen_v47_51_tables.py` | `results/e2e/` `collapse_readout.json` |
| `gen_v50_macros.tex` | `scripts/paper/make_gen_v50.py` | `results/e2e/` `scope_agent.json`, `scope_agent_learn.json` |
| `gen_v50_perinst.tex` | `scripts/paper/make_gen_v50.py` | `results/e2e/` `scope_agent.json`, `scope_agent_learn.json` |
| `gen_v50_table_main.tex` | `scripts/paper/make_gen_v50.py` | `results/e2e/` `scope_agent.json`, `scope_agent_learn.json` |
| `gen_v51.tex` | `scripts/paper/make_gen_v51.py` | `results/e2e/` `scope_agent_native.json`, `scope_agent_native_smalln.json` |
| `gen_v52.tex` | `scripts/paper/make_gen_v52.py` | `results/e2e/` `scope_agent_check.json` |
| `gen_v52_table_compact.tex` | `scripts/paper/make_gen_v52.py` | `results/e2e/` `scope_agent_check.json` |
| `gen_v53.tex` | `scripts/paper/make_gen_v53.py` | `results/theory_extension/` `certificate_v53.json`, `real_rules_exact.json` |
| `gen_v53_classes.tex` | `scripts/paper/make_gen_v53.py` | `results/theory_extension/` `certificate_v53.json`, `real_rules_exact.json` |
| `gen_v53_table.tex` | `scripts/paper/make_gen_v53.py` | `results/theory_extension/` `certificate_v53.json`, `real_rules_exact.json` |

Two intermediate fragments are produced but not input by the paper: `gen_control_suite.tex` and
`gen_bind.tex` (`make_corset_tables.py`), joined by `make_gen_csuite_bind.py` into
`gen_csuite_bind.tex`. `gen_v48_surrogate.tex` (`make_gen_v47_51_tables.py`) is input as Table 16 and
`make_figs_appendix.py` plots the same numbers as Figure 3.

## Archived results → experiment scripts

| Archived result | Produced by |
|---|---|
| `policy_sufficiency.json` | `src/saorl/benchmark_sg/policy_sufficiency.py`, `src/saorl/benchmark_sg/safe_face_select.py`, `src/saorl/benchmark_sg/baseline_table.py` |
| `safe_face_select.json` | `src/saorl/benchmark_sg/safe_face_offline.py` |
| `face_live.json` | `experiments/live_agent/face_live.py` (reads `results/e2e/live_crossing_replay.json`) |
| `face_ladder.json` | `src/saorl/benchmark_sg/face_ladder.py`, `src/saorl/benchmark_sg/decide_runtime.py` |
| `deploy_certificate.json` | `src/saorl/deploy_certificate.py` |
| `expected_cost_certificate.json` | `src/saorl/expected_cost_certificate.py` |
| `certificate_audit.json` | `src/saorl/certificate_audit.py` |
| `shadow_price.json` | `src/saorl/shadow_price.py` |
| `baseline_table.json` | `src/saorl/benchmark_sg/baseline_table.py` |
| `surrogate_price.json` | `src/saorl/benchmark_sg/collapse_readout.py`, `src/saorl/benchmark_sg/surrogate_price.py`, `src/saorl/benchmark_sg/decide_runtime.py` |
| `safe_collapse.json` | `src/saorl/benchmark_sg/safe_collapse.py` |
| `query_loop.json` | `src/saorl/benchmark_sg/query_loop.py` |
| `real_rules_finite_data.json` | `experiments/theory_extension/real_rules_experiment.py` |
| `real_rules_finite_data_records.json` | `experiments/theory_extension/real_rules_experiment.py` |
| `real_rules_exact.json` | `experiments/theory_extension/real_rules_experiment.py` |
| `control_suite_uncapped.json` | `src/saorl/benchmark_sg/semantic_risk_frontier.py`, `experiments/theory_extension/real_rules_experiment.py` |
| `class_ladder.json` | `src/saorl/benchmark_sg/class_ladder.py`, `src/saorl/benchmark_sg/faithfulness_provenance.py`, `src/saorl/benchmark_sg/decide_runtime.py` |
| `learner_slack.json` | `src/saorl/benchmark_sg/learner_slack.py` |
| `basis_size.json` | `src/saorl/benchmark_sg/basis_size.py` |
| `coverage_funnel.json` | `src/corset_e2e/analysis/coverage_funnel.py` |
| `third_corpus.json` | `src/saorl/benchmark_sg/third_corpus.py` |
| `decide_runtime.json` | `src/saorl/benchmark_sg/decide_runtime.py` |
| `control_suite.json` | `src/saorl/benchmark_sg/control_suite.py` |
| `switch_gap_exact.json` | `src/saorl/t31_switchgap.py` |
| `e0_report.json` | `src/saorl/benchmark_sg/e0_run.py` |
| `artemis_units.json` | `src/corset_e2e/external/artemis_decide.py`, `src/corset_e2e/external/artemis_arrow.py`, `src/corset_e2e/external/artemis_load.py` |
| `artemis_external.json` | `src/corset_e2e/external/artemis_decide.py`, `src/corset_e2e/external/artemis_arrow.py` |
| `gold_ast.json` | `src/corset_e2e/analysis/emission_gap_g2.py` |
| `pipeline_compare.json` | `src/saorl/benchmark_sg/pipeline_compare.py` (reads `results/conformal/lp/certificate_audit.json` and the `main50`/`budget50` learning-arm archives) |
| `collapse_utility.json` | `src/saorl/benchmark_sg/collapse_readout.py`, `src/saorl/benchmark_sg/collapse_utility.py` |
| `answer_noise_check.json` | `src/saorl/benchmark_sg/answer_noise_frontier.py`, `src/saorl/benchmark_sg/answer_noise_check.py` |
| `union_calibrated_v51.json` | `src/corset_e2e/analysis/union_calibrated_v51.py` |
| `guarantee_accounting.json` | `src/corset_e2e/analysis/guarantee_accounting.py` |
| `results/safe_keep_final/safekeep_final.json` | `src/saorl/benchmark_sg/safe_keep_final_run.py` (recovered driver, re-run reproduces the archive up to the runtime field) |
| `results/final_pipeline/pipeline_rows.json` | `src/saorl/benchmark_sg/final_pipeline_run.py` (recovered driver, re-run reproduces all 1,600 rows within LP tolerance and yields identical paper macros) |
| `scope_agent.json` (exact surface of the agent-service family: candidate geometry, exact optima, sufficient instances) | `src/saorl/benchmark_sg/scope_agent.py` |
| `scope_agent_learn.json` (six single-signal learners under the union surrogate and the ARROW reading, rule-defined settings; REGISTRATION_V50 and its Amendment 2), `scope_agent_learn_coarsegrid.json`, `scope_agent_learn_tau005.json` (the earlier inherited settings, kept for the sensitivity paragraph), `scope_agent_learn_pid_gains_x0.5.json`, `scope_agent_learn_pid_gains_x2.json` (PID gain sensitivity) | `src/saorl/benchmark_sg/scope_agent_learn.py`, `scripts/paper/run_pid_gain_sensitivity.py` |
| `scope_agent_native.json` , `scope_agent_native_smalln.json` (native multi-constraint learners, learner-aware tolerance; REGISTRATION_V51) | `src/saorl/benchmark_sg/scope_agent_native.py` |
| `scope_agent_check.json` (CHECK on every learned agent-service policy; REGISTRATION_V52) | `src/saorl/benchmark_sg/scope_agent_check.py` |
| `artemis_per_method.json` (every archived ARTEMIS generator on its own samples) | `src/corset_e2e/external/artemis_per_method.py` |
| `artemis_decide.json` (entailment structure of the retained sets; REGISTRATION_V23) | `src/corset_e2e/external/artemis_decide.py` |
| `experiments/live_agent/logs/eval_summary.json` (50 paired live sessions per condition) | `experiments/live_agent/run_eval.py` (billed API calls; archived, not meant to be re-run) |
| `results/dsrl/**` (per-run JSON files of the released offline safe-RL learners: the sweeps and top-ups that Table 19 reads, and the vector-constraint, E3 and non-dominated runs that the run count in Appendix B.1 includes) | `experiments/dsrl_learners/dsrl_sweep.py`, `dsrl_vector.py`, `dsrl_e3.py`, `dsrl_nondominated.py` (wrappers around the published OSRL/DSRL learners; the SLURM job files are omitted); `make_dsrl_table.py` and `aggregate_vector.py` summarize them |
| `results/theory_extension/certificate_v53.json` (three offline certificates on identical draws, the witness divergence and floor per class; REGISTRATION_V53) | `experiments/theory_extension/certificate_v53.py` (about four minutes on one CPU core; `--istar-only` refreshes and re-verifies the witness section alone; `certificate_v53_stdout.txt` is its log) |

Files under `results/e2e/` and `results/conformal/` that no row above names (`ledger.json`,
`gold_ast.json`, the `catalog_v4/` token files, the `v6_selections/` outputs, the `e2e_report_*`
and `v16_*` reports, `e2e_test_rows_v11_complete.json`, the E0 and LP reports) are inputs or
outputs of the drivers named above and of the generation-pipeline analyses that
`make_unified_numbers.py` and `make_gen_v47_51.py` read; they are kept because a listed driver
reads them.
