"""P3.4 (Addendum A5): additional pinned PromQL source repositories.

Fetches the three repos frozen in Addendum A5, parses their rendered
alert rules (both PrometheusRule-CRD `spec.groups` and plain rule-file
`groups` formats), and evaluates the SAME leave-one-out protocol with the
SAME transformation library and deterministic scorer as the frozen
study. The original 132-pool benchmark, its manifest, and its numbers
are untouched: this writes fetch_manifest_ext.json and
report_extension.json only, and reports new-repos-only plus combined
independent-source-family counts.

Run: SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.extension
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import yaml

from .dominance import pool_dominance
from .evaluate import build_prom_pool
from .fetch import (FileEntry, _clone_or_reuse, _collect_files, _head_sha,
                    _last_commit_ts, SRC_DIR)
from .parse import (PromReading, PromTarget, dur_to_s, parse_prom_expr,
                    parse_prometheus, prom_threshold_bank, _strip_template)
from .run_benchmark import DELTA, _loo_eval, _prom_skeleton, _record

_ROOT = Path(__file__).resolve().parents[3]
_MAN_EXT = _ROOT / "results/conformal" / "benchmark_sg" / "fetch_manifest_ext.json"
_REPORT = _ROOT / "results/conformal" / "benchmark_sg" / "report_extension.json"

# Addendum A5 repo list (frozen)
REPOS_EXT = {
    "thanos": dict(
        url="https://github.com/thanos-io/thanos",
        family="prometheus",
        globs=["examples/alerts/*.yaml"],
    ),
    "mimir": dict(
        url="https://github.com/grafana/mimir",
        family="prometheus",
        globs=["operations/mimir-mixin-compiled/alerts.yaml",
               "operations/mimir-mixin-compiled-baremetal/alerts.yaml"],
    ),
    "cluster-monitoring-operator": dict(
        url="https://github.com/openshift/cluster-monitoring-operator",
        family="prometheus",
        globs=["assets/**/*prometheus-rule*.yaml",
               "assets/**/*prometheusrule*.yaml"],
    ),
}


def fetch_ext() -> dict:
    out = dict(repos={}, files=[])
    for name, spec in REPOS_EXT.items():
        repo_dir = _clone_or_reuse(name, spec["url"])
        sha = _head_sha(repo_dir)
        entries = _collect_files(repo_dir, name, spec)
        out["repos"][name] = dict(url=spec["url"], family=spec["family"],
                                  commit=sha, n_files=len(entries),
                                  local_dir=str(repo_dir))
        out["files"].extend(e.__dict__ for e in entries)
    out["n_files_total"] = len(out["files"])
    _MAN_EXT.write_text(json.dumps(out, indent=1))
    return out


def _rule_groups(doc) -> list:
    """Both formats: PrometheusRule CRD (spec.groups) and plain rule file
    (groups). A file may contain multiple YAML documents."""
    if not isinstance(doc, dict):
        return []
    if "spec" in doc and isinstance(doc.get("spec"), dict):
        return doc["spec"].get("groups", []) or []
    return doc.get("groups", []) or []


def parse_ext(man: dict) -> Tuple[List[PromTarget], dict, int]:
    targets: List[PromTarget] = []
    skipped = defaultdict(int)
    oodsl_with_text = 0
    seen = set()
    for fe in man["files"]:
        commit = man["repos"][fe["repo"]]["commit"]
        try:
            docs = list(yaml.safe_load_all(open(fe["abs_path"])))
        except yaml.YAMLError:
            skipped["yaml_error"] += 1
            continue
        for doc in docs:
            for grp in _rule_groups(doc):
                gname = (grp or {}).get("name", "")
                for r in (grp or {}).get("rules", []) or []:
                    if "alert" not in r or "expr" not in r:
                        skipped["not_alert"] += 1
                        continue
                    parsed = parse_prom_expr(str(r["expr"]))
                    ann = r.get("annotations", {}) or {}
                    text = _strip_template(" ".join(
                        str(ann.get(k, "")) for k in ("summary", "description",
                                                      "message")))
                    if parsed is None:
                        skipped["out_of_dsl_expr"] += 1
                        if text:
                            oodsl_with_text += 1
                        continue
                    if not text:
                        skipped["no_text"] += 1
                        continue
                    sev = str((r.get("labels", {}) or {}).get("severity", ""))
                    reading = PromReading(
                        metric=parsed["metric"], selectors=parsed["selectors"],
                        comparator=parsed["comparator"],
                        threshold=parsed["threshold"],
                        rate_window_s=parsed["rate_window_s"],
                        aggregation=parsed["aggregation"],
                        agg_by=parsed["agg_by"],
                        for_s=dur_to_s(r.get("for")) or 0.0, severity=sev)
                    key = (fe["repo"], r["alert"], reading.struct_key())
                    if key in seen:
                        skipped["duplicate"] += 1
                        continue
                    seen.add(key)
                    targets.append(PromTarget(
                        name=r["alert"], group=gname, reading=reading,
                        text=text, repo=fe["repo"], commit=commit,
                        file=fe["rel_path"], sub_source=gname,
                        last_commit_ts=fe.get("last_commit_ts"),
                        raw_expr=re.sub(r"\s+", " ", str(r["expr"]).strip())))
    return targets, dict(skipped), oodsl_with_text


def main() -> None:
    man = fetch_ext()
    targets, skipped, oodsl_with_text = parse_ext(man)
    print(f"extension: {len(targets)} parsed targets; skipped: {skipped}; "
          f"oodsl_with_text: {oodsl_with_text}")
    if not targets:
        _REPORT.write_text(json.dumps(dict(error="no targets", skipped=skipped)))
        return

    # A5: same transformation library / scorer; threshold bank built from the
    # extension's own targets (parallel to the frozen study's construction).
    bank = prom_threshold_bank(targets)
    recs = [_record(build_prom_pool(t, bank), t, _prom_skeleton(t))
            for t in targets]
    # same oodsl-test convention as _family_report: the test half of the
    # oodsl-with-text rules count as automatic misses in overall coverage
    oodsl_test = (oodsl_with_text - max(1, int(round(oodsl_with_text * 0.5)))
                  if oodsl_with_text else 0)
    # leave-one-out tests every in-DSL rule, so the overall denominator
    # takes the whole out-of-DSL arm rather than its later half
    loo = _loo_eval(recs, oodsl_with_text, DELTA)

    # dominance across pools, mirroring dominance_all
    per_pool = [pool_dominance(r.vectors) for r in recs]
    n = len(per_pool)
    dom_stats = dict(
        n_pools=n,
        collapse_rate=round(sum(d["has_dominating_member"] for d in per_pool) / n, 4),
        frac_genuinely_non_dominated=round(
            sum(not d["has_dominating_member"] for d in per_pool) / n, 4),
        mean_n_maximal=round(sum(d["n_maximal"] for d in per_pool) / n, 3),
        nu_pi_positive_frac=round(
            sum(d.get("nu_pi", 0) > 0 for d in per_pool) / n, 4),
    )

    orig_targets, _ = parse_prometheus()
    orig_families = {t.file for t in orig_targets}
    ext_families = {(t.repo, t.file) for t in targets}
    per_repo = defaultdict(int)
    for t in targets:
        per_repo[t.repo] += 1

    rep = dict(
        manifest="EVIDENCE_FREEZE_MANIFEST.md Addendum A5 @ bbe3ac3",
        provenance={k: dict(url=v["url"], commit=v["commit"])
                    for k, v in man["repos"].items()},
        counts=dict(parsed=len(targets), skipped=skipped,
                    oodsl_with_text=oodsl_with_text, oodsl_test=oodsl_test,
                    per_repo=dict(per_repo),
                    n_source_files=len(ext_families)),
        primary_iid_loo=loo,
        dominance=dom_stats,
        combined_clusters=dict(
            original_prom_source_files=len(orig_families),
            extension_source_files=len(ext_families),
            combined=len(orig_families) + len(ext_families),
            note="independent source-file clusters, V3 metric; original "
                 "132-pool study untouched"),
    )
    _REPORT.write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("counts", "dominance")}, indent=1,
                     default=str))
    print("LOO in-DSL coverage:", loo.get("in_dsl_coverage"),
          "CP", loo.get("in_dsl_cov_cp"), "mean set", loo.get("mean_set_size"))
    print("combined clusters:", rep["combined_clusters"]["combined"])
    print("wrote", _REPORT)


if __name__ == "__main__":
    main()
