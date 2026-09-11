# Third-party content

`results_conformal/benchmark_sg/_src/` contains the specific rule files that the
experiments read, copied from public repositories at the commits pinned in
`results_conformal/benchmark_sg/fetch_manifest.json` and
`results_conformal/e0/fetch_manifest_e0.json` (repository URLs and commit hashes
are recorded there). Only the files the pipeline consumes are included, and each
keeps its original relative path. The originals are distributed under their own
licenses (Apache-2.0 for kyverno/policies, kube-prometheus, VictoriaMetrics,
grafana/loki, grafana/tempo, thanos, rook, mimir, tidb, cluster-monitoring-operator;
MIT for samber/awesome-prometheus-alerts; LGPL/other for ceph; see each project).
`saorl/benchmark_sg/fetch.py` re-fetches the corpora at the pinned commits.

`saorl/cmapss*.py` read the public NASA C-MAPSS turbofan dataset, which is not
redistributed here; the replay-MDP results derived from it are archived under
`results_conformal/`.
