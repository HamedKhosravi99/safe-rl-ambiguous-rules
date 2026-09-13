# Safe RL under ambiguous written rules — code, archived results, and paper generators

Code and data behind *When Is One Reading Enough? Decision-Relevant Ambiguity in Constrained
Reinforcement Learning* (ARROW). Every number, table, and figure in the paper is produced by a
script in this repository from an archived result file under `results/` or `experiments/`;
`scripts/reproduce_paper.py` regenerates all of them and checks them byte-for-byte against the
fragments compiled into the submitted PDF.

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
│   │   │                      #   corpus tooling (parse.py, candidates.py, fetch.py), learning arms
│   │   │                      #   (experiments.py, budget_rl.py, cmapss*.py) and every experiment driver
│   │   └── benchmark_sg/
│   └── corset_e2e/            # open-domain generation pipeline: candidate generation, calibration,
│                              #   coverage analyses, the ARTEMIS external study (external/artemis_*.py)
├── experiments/
│   ├── live_agent/            # the tool-using agent under a published rate limit (Table 10), with logs
│   ├── theory_extension/      # offline certificates on the 28 compiled rules: real_rules_experiment.py
│   │                          #   (archived generic certificate) and certificate_v53.py (registered V53 study:
│   │                          #   generic uniform, occupancy-weighted and robust-dual certificates on identical
│   │                          #   draws, budget tightening, decision-information radius)
│   └── dsrl_learners/         # wrappers that ran the published offline safe-RL learners (results/dsrl/)
├── scripts/
│   ├── paper/                 # make_gen_*.py, make_fig*.py: archived results -> LaTeX fragments and figures
│   └── reproduce_paper.py     # runs them all in dependency order and diffs against paper/reference/
├── results/                   # archived experiment outputs read by the generators
│   ├── e2e/                   #   exact decision suites, clarification, class ladder, coverage, ARTEMIS,
│   │                          #   the agent-service archives (scope_agent*.json), and the pre-registration
│   │                          #   notes REGISTRATION_V17, V18-V21, V23, V50-V53
│   ├── compiler_audit/        #   compiler-faithfulness audit behind SAFE-KEEP's implication relation
│   ├── conformal/             #   learning arms (agnostic50, main50, ...), LP certificates (lp/), corpus manifests
│   ├── dsrl/                  #   published offline safe-RL learners on DSRL/OSRL
│   ├── safe_keep_final/, final_pipeline/   # SAFE-KEEP on 579 families and the fixed-data pipeline (Table 16)
│   ├── theory_extension/      #   offline certificates on the 28 rules: the archived generic sweep and
│   │                          #   certificate_v53.json (Tables 12-15, Figure 2)
│   ├── paper_extra/, corpus_eval/, fragments/
├── data/rule_corpora/         # the third-party rule files the pipeline reads, at pinned commits (NOTICE.md inside)
├── paper/
│   ├── reference/             # the exact fragments (generated/) and figures (figure/) compiled into the PDF
│   ├── generated/, figure/    # outputs of scripts/reproduce_paper.py (ignored by git)
│   └── supplementary_derivations.tex   # long-form derivations the appendix now states compactly
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

This runs the 38 generators in `scripts/paper/` in dependency order, writes the fragments to
`paper/generated/` and the figures to `paper/figure/`, and reports the comparison with
`paper/reference/`. At release time all 61 fragments the paper inputs are byte-identical and the
three PDF figures regenerate (PDFs embed a timestamp, so they are not byte-compared); the whole
run takes about 20 seconds. Run a subset with
`python3 scripts/reproduce_paper.py --only make_gen_v53 make_fig_v53`.

## Re-running experiments

Every archived result lists its driver in the tables below. Drivers run from the repository
root with `PYTHONPATH=src`. For example

```bash
python3 src/saorl/benchmark_sg/policy_sufficiency.py        # exact ARROW verdicts behind Table 2
python3 src/saorl/benchmark_sg/safe_keep_final_run.py       # SAFE-KEEP on 579 families (Table 16), ~30 s
python3 src/saorl/benchmark_sg/final_pipeline_run.py        # fixed-data pipeline, 1,600 records (Table 16), ~30 s
python3 experiments/theory_extension/certificate_v53.py     # three offline certificates on the 28 rules (Tables 12-15), ~4 min
python3 src/saorl/benchmark_sg/scope_agent.py               # exact surface of the agent-service family (Table 3, Tables 17-18)
python3 src/corset_e2e/external/artemis_per_method.py       # every archived ARTEMIS method on its own samples (Table 4, Table 23)
```

The exact-LP experiments take seconds to minutes. The learning arms (`src/saorl/experiments.py`
and `src/saorl/benchmark_sg/scope_agent_learn.py`, `scope_agent_native.py`, `scope_agent_check.py`)
and the DSRL wrappers take hours and need the `learners` extra; the C-MAPSS replay needs the
public NASA C-MAPSS files in `data/cmapss/` (not redistributed). The live agent
(`experiments/live_agent/`) bills a metered API and is archived rather than meant to be re-run.

## Map: main-text tables and figures → code

Page numbers refer to the submitted PDF.

| Display | Page | What it reports | Generator | Archived data |
|---|---|---|---|---|
| **Table 1** | 7 | Experimental roadmap (four questions, evidence, theory link) | hand-written | -- |
| **Table 2** | 8 | ARROW detects decision-relevant ambiguity beyond the value test | `scripts/paper/make_gen_v13.py` | `results/e2e/face_live.json`, `results/e2e/policy_sufficiency.json` |
| **Figure 1** | 8 | Decision-focused clarification reaches sufficiency before identification | `scripts/paper/make_figs_mpl.py` (`fig_e3_questions.pdf`) | `results/e2e/safe_collapse.json` |
| **Table 3** | 8 | Return recovered by ARROW on the sufficient agent-service instances (single-signal learners) | `scripts/paper/make_gen_v50.py` | `results/e2e/scope_agent.json`, `results/e2e/scope_agent_learn.json` |
| **Table 4** | 9 | Comparison with nine language-to-specification methods on 182 ARTEMIS requirements | `scripts/paper/make_gen_artemis_methods.py`, `scripts/paper/make_gen_artemis.py` | `results/e2e/artemis_per_method.json`, `results/e2e/artemis_external.json` |

## Map: appendix tables and figures → code

| Display | Page | Appendix | What it reports | Generator | Archived data |
|---|---|---|---|---|---|
| **Table 5** | 14 | A | Positioning by problem solved, safety input, output and guarantee | hand-written | -- |
| **Table 6** | 16 | B.1 | Retention and deployment combinations used by the pipeline | hand-written | -- |
| **Table 7** | 29 | D | Appendix roadmap (question, evidence, theory link, what it establishes) | hand-written; one count from `scripts/paper/make_gen_v52.py` | `results/e2e/scope_agent_check.json` |
| **Table 8** | 29 | D.1 | What single-reading optimization hides, solved exactly (compiled suite) | `scripts/paper/make_gen_csuite_bind.py` joining fragments of `scripts/paper/make_corset_tables.py` | `results/conformal/benchmark_sg/control_suite.json`, `results/e2e/exact_nonnested.json` |
| **Table 9** | 30 | D.1 | Cross-organization replication: 465 pools from eight further organizations, value-screen vs ARROW clears at d=0.005 | `scripts/paper/make_gen_revision.py` (`gen_third_corpus_compact.tex`) | `results/e2e/third_corpus.json` |
| **Table 10** | 30 | D.1 | Live service agent under a published rate limit (50 paired billed sessions per condition) | `scripts/paper/make_gen_live.py` | `experiments/live_agent/logs/eval_summary.json` |
| **Table 11** | 31 | D.1 | Safe collapse versus identification by question count | `scripts/paper/make_gen_v43.py` | `results/e2e/safe_collapse.json`, `results/e2e/basis_size.json`, `results/e2e/learner_slack.json` |
| **Table 12** | 31 | D.2 | Three offline certificates on identical chain draws (uniform / occupancy-weighted / robust dual), certified counts and false certificates by log size (V53) | `scripts/paper/make_gen_v53.py` (`gen_v53_table.tex`) | `results/theory_extension/certificate_v53.json` |
| **Table 13** | 31 | D.2 | Three data scales per sufficient-reading class: n* of each certificate and the information floor kl(1-delta,delta)/I* | `scripts/paper/make_gen_v53.py` (`gen_v53_classes.tex`) | `results/theory_extension/certificate_v53.json` |
| **Figure 2** | 32 | D.2 | Generic certificate, decision-specific certificate and information floor against 1/kappa^2 | `scripts/paper/make_fig_v53.py` (`fig_v53_scales.pdf`) | `results/theory_extension/certificate_v53.json` |
| **Table 14** | 32 | D.2 | Smallest certified budget tightening for the hard class, with exact return prices | `scripts/paper/make_gen_v53.py` (`gen_v53_tightening.tex`) | `results/theory_extension/certificate_v53.json` |
| **Table 15** | 32 | D.2 | Counts from sampled load-chain paths (secondary of V53) | `scripts/paper/make_gen_v53.py` (`gen_v53_secondary.tex`) | `results/theory_extension/certificate_v53.json` |
| **Table 16** | 33 | D.2 | Safe-Keep audit: reduction, coverage and deployment records | `scripts/paper/make_gen_safekeep.py` | `results/safe_keep_final/safekeep_final.json`, `results/final_pipeline/pipeline_rows.json` |
| **Figure 3** | 33 | D.3 | Exact price of the pointwise-maximum surrogate by budget | `scripts/paper/make_figs_appendix.py` (`fig_e1_surrogate.pdf`) over `scripts/paper/make_gen_v47_51_tables.py` | `results/e2e/collapse_readout.json` |
| **Table 17** | 34 | D.3 | Single-signal learners under the union surrogate and the sufficient reading (all instances) | `scripts/paper/make_gen_v50.py` | `results/e2e/scope_agent.json`, `results/e2e/scope_agent_learn.json` |
| **Table 18** | 34 | D.3 | Per-instance gains of the ARROW-selected reading over the union surrogate (9 sufficient instances, 4 learners) | `scripts/paper/make_gen_v50.py` (`gen_v50_perinst.tex`) | `results/e2e/scope_agent.json`, `results/e2e/scope_agent_learn.json` |
| **Table 19** | 35 | D.3 | Native multi-constraint learners: K separate constraints versus the sufficient reading | `scripts/paper/make_gen_v51.py` | `results/e2e/scope_agent_native.json` |
| **Table 20** | 35 | D.3 | Checked deployment on all learned agent-service policies | `scripts/paper/make_gen_v52.py`, `scripts/paper/make_gen_v50.py` | `results/e2e/scope_agent_check.json`, `results/e2e/scope_agent.json` |
| **Table 21** | 36 | D.3 | Budget sweep on the maintenance domains: both arms re-learned at every budget | `scripts/paper/make_gen_extra.py` | `results/conformal/selfconsistency.json`, `results/e2e/e2e_report_v4.json` |
| **Table 22** | 37 | D.3 | Portability across five released offline safe-RL learners on DSRL/OSRL (**GPU**: PACE, V100) | `scripts/paper/make_corset_tables.py` (`gen_dsrl.tex`, `gen_dsrl_seedinfo.tex`) | `results/dsrl/` |
| **Table 23** | 38 | D.4 | Every archived ARTEMIS method on its own samples | `scripts/paper/make_gen_artemis_methods.py` | `results/e2e/artemis_per_method.json` |
| **Table 24** | 38 | D.4 | Expert-plausible readings preserved (union pool, 190 samples) | `scripts/paper/make_gen_artemis.py` | `results/e2e/artemis_external.json` |
| **Table 25** | 38 | D.4 | Simple alternatives address different parts of the problem | `scripts/paper/make_gen_revision.py`, `scripts/paper/make_gen_v14.py` | `results/e2e/baseline_table.json`, `results/e2e/face_ladder.json` |

Table 22 is the one GPU result (five released offline safe-RL learners on DSRL/OSRL, run on V100s
through the wrappers in `experiments/dsrl_learners/`); everything else was produced on CPU.

## Every LaTeX fragment the paper inputs

| Fragment | Generator | Archived data read |
|---|---|---|
| `gen_applic.tex` | `scripts/paper/make_gen_applic.py` | `results/conformal/lp/` `evaluator_scale_50k.json`, `exposure_ceiling.json`; `results/e2e/` `artemis_units.json`, `query_loop.json`; `results/theory_extension/` `real_rules_finite_data.json`, `real_rules_finite_data_records.json` |
| `gen_artemis.tex` | `scripts/paper/make_gen_artemis.py` | `results/e2e/` `artemis_external.json`, `natural2ctl_external.json` |
| `gen_artemis_decide.tex` | `scripts/paper/make_gen_artemis_decide.py` | `results/e2e/` `artemis_decide.json` |
| `gen_artemis_methods.tex` | `scripts/paper/make_gen_artemis_methods.py` | `results/e2e/` `artemis_per_method.json` |
| `gen_artemis_methods_app.tex` | `scripts/paper/make_gen_artemis_methods.py` | `results/e2e/` `artemis_per_method.json` |
| `gen_artemis_methods_table.tex` | `scripts/paper/make_gen_artemis_methods.py` | `results/e2e/` `artemis_per_method.json` |
| `gen_avail.tex` | `scripts/paper/make_gen_avail.py` | `results/conformal/e0/` `e0_report.json`; `results/conformal/lp/` `certified_at_scale.json`, `certified_at_scale_d05.json`; `results/corpus_eval/` `test_report_openai.json`; … |
| `gen_certprice.tex` | `scripts/paper/make_corset_tables.py` | `results/conformal/agnostic50/` `experiments_20260724-004227.json`; `results/conformal/benchmark_sg/` `calib_vs_fixed.json`, `control_suite.json`, `matched_size.json` …; `results/conformal/budget50/` `experiments_20260723-235413.json`; … |
| `gen_certscale_caption.tex` | `scripts/paper/make_corset_tables.py` | `results/conformal/agnostic50/` `experiments_20260724-004227.json`; `results/conformal/benchmark_sg/` `calib_vs_fixed.json`, `control_suite.json`, `matched_size.json` …; `results/conformal/budget50/` `experiments_20260723-235413.json`; … |
| `gen_control_suite_caption.tex` | `scripts/paper/make_corset_tables.py` | `results/conformal/agnostic50/` `experiments_20260724-004227.json`; `results/conformal/benchmark_sg/` `calib_vs_fixed.json`, `control_suite.json`, `matched_size.json` …; `results/conformal/budget50/` `experiments_20260723-235413.json`; … |
| `gen_cov.tex` | `scripts/paper/make_gen_cov.py` | `results/e2e/` `coverage_anatomy.json`, `e2e_report_g1.json`, `e2e_report_g1_mondrian.json` |
| `gen_csuite_bind.tex` | `scripts/paper/make_gen_csuite_bind.py` | joins `gen_control_suite.tex` and `gen_bind.tex` (both from `make_corset_tables.py`) |
| `gen_dsrl.tex` | `scripts/paper/make_corset_tables.py` | `results/conformal/agnostic50/` `experiments_20260724-004227.json`; `results/conformal/benchmark_sg/` `calib_vs_fixed.json`, `control_suite.json`, `matched_size.json` …; `results/conformal/budget50/` `experiments_20260723-235413.json`; … |
| `gen_dsrl_seedinfo.tex` | `scripts/paper/make_corset_tables.py` | `results/conformal/agnostic50/` `experiments_20260724-004227.json`; `results/conformal/benchmark_sg/` `calib_vs_fixed.json`, `control_suite.json`, `matched_size.json` …; `results/conformal/budget50/` `experiments_20260723-235413.json`; … |
| `gen_dsrlcount.tex` | `scripts/paper/make_corset_tables.py` | `results/conformal/agnostic50/` `experiments_20260724-004227.json`; `results/conformal/benchmark_sg/` `calib_vs_fixed.json`, `control_suite.json`, `matched_size.json` …; `results/conformal/budget50/` `experiments_20260723-235413.json`; … |
| `gen_e0_caption.tex` | `scripts/paper/make_corset_tables.py` | `results/conformal/agnostic50/` `experiments_20260724-004227.json`; `results/conformal/benchmark_sg/` `calib_vs_fixed.json`, `control_suite.json`, `matched_size.json` …; `results/conformal/budget50/` `experiments_20260723-235413.json`; … |
| `gen_e2e.tex` | `scripts/paper/make_gen_e2e.py` | `results/e2e/catalog_v4/` token files; `results/e2e/` `e2e_report_v4.json`, `e2e_test_rows_v4_complete.json` … |
| `gen_e5_prose.tex` | `scripts/paper/make_gen_e5_prose.py` | reads `gen_learn_table.tex` and `gen_e5_side.tex` (fragments) |
| `gen_e9_margin.tex` | `scripts/paper/make_corset_tables.py` | `results/conformal/agnostic50/` `experiments_20260724-004227.json`; `results/conformal/benchmark_sg/` `calib_vs_fixed.json`, `control_suite.json`, `matched_size.json` …; `results/conformal/budget50/` `experiments_20260723-235413.json`; … |
| `gen_extension_caption.tex` | `scripts/paper/make_corset_tables.py` | `results/conformal/agnostic50/` `experiments_20260724-004227.json`; `results/conformal/benchmark_sg/` `calib_vs_fixed.json`, `control_suite.json`, `matched_size.json` …; `results/conformal/budget50/` `experiments_20260723-235413.json`; … |
| `gen_finite.tex` | `scripts/paper/make_gen_finite.py` | `results/e2e/` `control_suite_uncapped.json`; `results/theory_extension/` `real_rules_exact.json`, `real_rules_finite_data.json`, `real_rules_finite_data_records.json` |
| `gen_frontier_macros.tex` | `scripts/paper/make_gen_extra.py` | `results/conformal/` `selfconsistency.json`; `results/e2e/` `e2e_report_v4.json`; `results/paper_extra/pareto/` `pareto_risk_20260724-013218.json` |
| `gen_funnel.tex` | `scripts/paper/make_gen_extra.py` | `results/conformal/` `selfconsistency.json`; `results/e2e/` `e2e_report_v4.json`; `results/paper_extra/pareto/` `pareto_risk_20260724-013218.json` |
| `gen_ladder.tex` | `scripts/paper/make_gen_ladder.py` | `results/e2e/` `class_ladder.json` |
| `gen_live.tex` | `scripts/paper/make_gen_live.py` | `experiments/live_agent/logs/` `eval_summary.json` |
| `gen_live_agent.tex` | `scripts/paper/make_gen_live.py` | `experiments/live_agent/logs/` `eval_summary.json` |
| `gen_livecross.tex` | `scripts/paper/make_gen_livecross.py` | `results/e2e/` `live_crossing_keep.json`, `live_crossing_keep_stability.json`, `live_crossing_replay.json` |
| `gen_lmbase_caption.tex` | `scripts/paper/make_corset_tables.py` | `results/conformal/agnostic50/` `experiments_20260724-004227.json`; `results/conformal/benchmark_sg/` `calib_vs_fixed.json`, `control_suite.json`, `matched_size.json` …; `results/conformal/budget50/` `experiments_20260723-235413.json`; … |
| `gen_matchedsize_macros.tex` | `scripts/paper/make_corset_tables.py` | `results/conformal/agnostic50/` `experiments_20260724-004227.json`; `results/conformal/benchmark_sg/` `calib_vs_fixed.json`, `control_suite.json`, `matched_size.json` …; `results/conformal/budget50/` `experiments_20260723-235413.json`; … |
| `gen_prov.tex` | `scripts/paper/make_gen_prov.py` | `results/e2e/` `downstream_provenance.json`, `g2_test_reach.json`, `gold_ast.json` … |
| `gen_r6.tex` | `scripts/paper/make_gen_r6.py` | `data/rule_corpora/kyverno-policies/argo-cel/application-field-validation/` `application-field-validation.yaml`; `data/rule_corpora/kyverno-policies/argo-cel/application-prevent-default-project/` `application-prevent-default-project.yaml`; `data/rule_corpora/kyverno-policies/argo-cel/application-prevent-updates-project/` `application-prevent-updates-project.yaml`; … |
| `gen_revision.tex` | `scripts/paper/make_gen_revision.py` | `results/conformal/lp/` `certificate_audit.json`, `deploy_certificate.json`, `evaluator_scale.json` …; `results/conformal/` `selfconsistency.json`; `results/e2e/` `baseline_table.json`, `decide_runtime.json`, `faithful_recalibration.json` … |
| `gen_safekeep.tex` | `scripts/paper/make_gen_safekeep.py` | `results/final_pipeline/` `pipeline_rows.json`; `results/safe_keep_final/` `safekeep_final.json` |
| `gen_screencost.tex` | `scripts/paper/make_corset_tables.py` | `results/conformal/agnostic50/` `experiments_20260724-004227.json`; `results/conformal/benchmark_sg/` `calib_vs_fixed.json`, `control_suite.json`, `matched_size.json` …; `results/conformal/budget50/` `experiments_20260723-235413.json`; … |
| `gen_selfcons_macros.tex` | `scripts/paper/make_gen_extra.py` | `results/conformal/` `selfconsistency.json`; `results/e2e/` `e2e_report_v4.json`; `results/paper_extra/pareto/` `pareto_risk_20260724-013218.json` |
| `gen_shadow_caption.tex` | `scripts/paper/make_corset_tables.py` | `results/conformal/agnostic50/` `experiments_20260724-004227.json`; `results/conformal/benchmark_sg/` `calib_vs_fixed.json`, `control_suite.json`, `matched_size.json` …; `results/conformal/budget50/` `experiments_20260723-235413.json`; … |
| `gen_sota.tex` | `scripts/paper/make_gen_sota.py` | `results/e2e/` `baseline_table.json` |
| `gen_sweep_budget.tex` | `scripts/paper/make_gen_extra.py` | `results/conformal/` `selfconsistency.json`; `results/e2e/` `e2e_report_v4.json`; `results/paper_extra/pareto/` `pareto_risk_20260724-013218.json` |
| `gen_sweep_macros.tex` | `scripts/paper/make_gen_extra.py` | `results/conformal/` `selfconsistency.json`; `results/e2e/` `e2e_report_v4.json`; `results/paper_extra/pareto/` `pareto_risk_20260724-013218.json` |
| `gen_third_corpus_compact.tex` | `scripts/paper/make_gen_revision.py` | `results/conformal/lp/` `certificate_audit.json`, `deploy_certificate.json`, `evaluator_scale.json` …; `results/conformal/` `selfconsistency.json`; `results/e2e/` `baseline_table.json`, `decide_runtime.json`, `faithful_recalibration.json` … |
| `gen_unified_numbers.tex` | `scripts/paper/make_unified_numbers.py` | `results/conformal/benchmark_sg/` `control_suite.json`, `report.json`, `report_extension.json`; `results/conformal/e0/` `e0_report.json`; `results/conformal/lp/` `deploy_certificate.json`, `expected_cost_certificate.json`, `shadow_price.json` …; … |
| `gen_v11b.tex` | `scripts/paper/make_gen_v11b.py` | `results/e2e/` `e2e_report_v11.json`, `ksweep_v11.json`, `pool_axes_witnesses.json` … |
| `gen_v13.tex` | `scripts/paper/make_gen_v13.py` | `results/e2e/` `face_live.json`, `policy_sufficiency.json` |
| `gen_v14.tex` | `scripts/paper/make_gen_v14.py` | `results/e2e/` `face_ladder.json` |
| `gen_v28.tex` | `scripts/paper/make_gen_v28.py` | `results/e2e/` `policy_sufficiency.json`, `safe_face_select.json` |
| `gen_v41.tex` | `scripts/paper/make_gen_v41.py` | `results/e2e/` `decision_equivalence_readout.json`, `k_scaling.json` |
| `gen_v43.tex` | `scripts/paper/make_gen_v43.py` | `results/e2e/` `basis_size.json`, `learner_slack.json`, `safe_collapse.json` |
| `gen_v43_table.tex` | `scripts/paper/make_gen_v43.py` | `results/e2e/` `basis_size.json`, `learner_slack.json`, `safe_collapse.json` |
| `gen_v47_51.tex` | `scripts/paper/make_gen_v47_51.py` | `results/e2e/` `answer_noise_check.json`, `collapse_readout.json`, `collapse_utility.json` … |
| `gen_v50_macros.tex` | `scripts/paper/make_gen_v50.py` | `results/e2e/` `scope_agent.json`, `scope_agent_learn.json` |
| `gen_v50_perinst.tex` | `scripts/paper/make_gen_v50.py` | `results/e2e/` `scope_agent.json`, `scope_agent_learn.json` |
| `gen_v50_table_full.tex` | `scripts/paper/make_gen_v50.py` | `results/e2e/` `scope_agent.json`, `scope_agent_learn.json` |
| `gen_v50_table_main.tex` | `scripts/paper/make_gen_v50.py` | `results/e2e/` `scope_agent.json`, `scope_agent_learn.json` |
| `gen_v51.tex` | `scripts/paper/make_gen_v51.py` | `results/e2e/` `scope_agent_native.json`, `scope_agent_native_smalln.json` |
| `gen_v52.tex` | `scripts/paper/make_gen_v52.py` | `results/e2e/` `scope_agent_check.json` |
| `gen_v52_table_compact.tex` | `scripts/paper/make_gen_v52.py` | `results/e2e/` `scope_agent_check.json` |
| `gen_v53.tex` | `scripts/paper/make_gen_v53.py` | `results/theory_extension/` `certificate_v53.json`, `real_rules_exact.json` |
| `gen_v53_classes.tex` | `scripts/paper/make_gen_v53.py` | `results/theory_extension/` `certificate_v53.json`, `real_rules_exact.json` |
| `gen_v53_secondary.tex` | `scripts/paper/make_gen_v53.py` | `results/theory_extension/` `certificate_v53.json`, `real_rules_exact.json` |
| `gen_v53_table.tex` | `scripts/paper/make_gen_v53.py` | `results/theory_extension/` `certificate_v53.json`, `real_rules_exact.json` |
| `gen_v53_tightening.tex` | `scripts/paper/make_gen_v53.py` | `results/theory_extension/` `certificate_v53.json`, `real_rules_exact.json` |

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
| `results/dsrl/**` | `experiments/dsrl_learners/dsrl_sweep.py`, `dsrl_vector.py`, `dsrl_e3.py`, `dsrl_nondominated.py` (wrappers around the published OSRL/DSRL learners; the SLURM job files are omitted) |
| `scope_agent.json` (exact surface of the agent-service family: candidate geometry, exact optima, sufficient instances) | `src/saorl/benchmark_sg/scope_agent.py` |
| `scope_agent_learn.json` (four single-signal learners under the union surrogate and the ARROW reading; REGISTRATION_V50) | `src/saorl/benchmark_sg/scope_agent_learn.py` |
| `scope_agent_native.json`, `scope_agent_native_smalln.json` (native multi-constraint learners, learner-aware tolerance; REGISTRATION_V51) | `src/saorl/benchmark_sg/scope_agent_native.py` |
| `scope_agent_check.json` (CHECK on every learned agent-service policy; REGISTRATION_V52) | `src/saorl/benchmark_sg/scope_agent_check.py` |
| `artemis_per_method.json` (every archived ARTEMIS generator on its own samples) | `src/corset_e2e/external/artemis_per_method.py` |
| `artemis_decide.json` (entailment structure of the retained sets; REGISTRATION_V23) | `src/corset_e2e/external/artemis_decide.py` |
| `results/theory_extension/certificate_v53.json` (three offline certificates on identical draws, tightening sweep, decision-information radius; REGISTRATION_V53) | `experiments/theory_extension/certificate_v53.py` (about four minutes on one CPU core; `certificate_v53_stdout.txt` is its log) |
| `experiments/live_agent/logs/eval_summary.json` (50 paired live sessions per condition) | `experiments/live_agent/run_eval.py` (billed API calls; archived, not meant to be re-run) |

