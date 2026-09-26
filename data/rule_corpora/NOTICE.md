# Third-party content

This directory holds the third-party rule files the pipeline reads, copied from public
repositories at pinned commits and kept at their original relative paths. Only the files
the experiments consume are included. The corpora shipped here are
awesome-prometheus-alerts, ceph, kube-prometheus, kyverno-policies, loki, rook, tempo, victoriametrics.

Provenance. The kube-prometheus and kyverno/policies files are listed with repository
URLs, commit hashes and per-file hashes in `results/conformal/benchmark_sg/fetch_manifest.json`,
and `src/saorl/benchmark_sg/fetch.py` re-fetches both at those commits. The
awesome-prometheus-alerts, ceph, loki, rook, tempo, victoriametrics files are listed in `results/conformal/e0/fetch_manifest_e0.json`.
The loki, victoriametrics, rook, ceph rule files belong to the eight-repository replication
(`src/saorl/benchmark_sg/third_corpus.py`), whose pinned commits for all eight
repositories are recorded under `commits` in `results/e2e/third_corpus.json`.

Not shipped. The mimir, cluster-monitoring-operator, thanos, tidb rule files of that replication are not
included. The per-repository results derived from them are archived in
`results/e2e/third_corpus.json` and the harvested rule catalogs in `results/e2e/catalog_v4/`,
so every number in the paper regenerates without them. Re-running the replication on those
four repositories requires checking each out at its recorded commit under this directory.

Licenses. The originals are distributed under their own licenses (Apache-2.0 for
kyverno/policies, kube-prometheus, VictoriaMetrics, grafana/loki, grafana/tempo, thanos,
rook, mimir, tidb, cluster-monitoring-operator; MIT for samber/awesome-prometheus-alerts;
LGPL/other for ceph; see each project).

`src/saorl/cmapss.py` and `src/saorl/cmapss_env.py` read the public NASA C-MAPSS turbofan
dataset, which is not redistributed here; the replay-MDP results derived from it are
archived under `results/conformal/`.
