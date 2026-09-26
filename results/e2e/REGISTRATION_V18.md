# REGISTRATION V18 — a third corpus: monitoring rules from eight other organizations

Written 2026-09-02, before any pool of this corpus is built. Implements
revision item P10 / W7-4 ("a third policy corpus"). The pinned public-rule
analysis of the paper is one organization's monitoring rules
(kube-prometheus) and one organization's admission rules (kyverno/policies).
This adds the exact decision-level pipeline on the alert rules of eight
other organizations whose repositories are already pinned on disk for the
generation study (results/e2e/catalog_src_manifest.json SHAs):
mimir and loki (Grafana Labs), cluster-monitoring-operator (Red Hat
OpenShift), victoriametrics, rook and ceph, thanos, tidb (PingCAP).
Excluded, with the reason stated: gitlab-runbooks (machine expansion of a
few SLO templates over components; 90% of the open-domain corpus mass,
its inclusion would make the corpus one organization again) and
awesome-prometheus-alerts (a cross-vendor collection, not an
organization's deployment, and the generator's development split).
tempo ships its rules as jsonnet, not YAML, and has no parseable alert.

## Pipeline (identical to the kube-prometheus analysis; nothing retuned)

Alert rules with summary/description text whose expression parses into
the six-tuple grammar (parse_prom_expr), deduplicated on (alert name,
structural key); threshold bank taken from this corpus; candidate pools,
fixture universes and behaviour classes exactly as build_prom_pool; then
  * non-dominance (pool_dominance);
  * compiled class: eligibility (_eligible), compile_instance, the exact
    value screen at d in {0.005, 0.05}, the face certificate W_psi at the
    same budgets (one LP per competing reading);
  * free class: exact MILP screen at d = 0.05; ladder rungs k = 0, 1, 2
    (subset cap 60, as published); face-level frontier;
  * faithfulness verdict of every recorded reading (verdict_of) and the
    compiled screen restated on the faithful subset.

## Endpoints

Per repository and pooled: rules parsed / with text / deduplicated;
non-dominance rate; compiled eligible / nested-clear / unanalyzed; screen
fires at 0.05 and 0.005; value-clearing vs face-clearing at both budgets;
free-class fires; ladder some/every-pair counts; frontier distribution;
faithful share and the faithful restatement of the compiled screen.

## Branch rules

Everything is reported wherever it lands, next to the kube-prometheus
numbers, as a descriptive replication: no budget, cap, grammar or fixture
is changed after seeing results. If the compiled screen fires on a
materially larger share here than on kube-prometheus, the paper says the
rare-firing finding is corpus-dependent; if it does not, it says the
finding replicates on eight further organizations. Either sentence is
written only after the run.

Output: results/e2e/third_corpus.json; macros via make_gen_revision.py.
