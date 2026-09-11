"""E0 prospective campaign driver (Addendum A6 @ 669ef94).

Stage discipline (freeze order): cluster assignment is a pure function of
the file PATH (HMAC salt1), so --stage cal parses ONLY calibration-assigned
files, computes and freezes q_sem, and exits; --stage test may run only
after the freeze file exists, and parses test-assigned files once.

Primary chain: retention mode (G0 branch 3) — pools by the frozen
transformation library (prom_candidates + E0-D threshold bank +
prom_fixtures), score S-det, unit = hash-selected semantic representative
per source-file cluster. Secondary: the target-independent generator's
recall/containment on E0-T representatives (rho_DSL instrument).

Run:
  SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.e0_run --stage cal
  SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.e0_run --stage test
"""
from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from .dominance import pool_dominance
from .evaluate import build_prom_pool
from .extension import _rule_groups
from .fetch import _clone_or_reuse, _collect_files, _head_sha
from .g0_audit import dev_targets
from .gen_independent import build_vocab, generate, proj
from .parse import (PromReading, PromTarget, dur_to_s, parse_prom_expr,
                    prom_threshold_bank, _strip_template)
from .run_benchmark import _prom_skeleton, _record

_ROOT = Path(__file__).resolve().parents[3]
_OUT = _ROOT / "results/conformal" / "e0"
_MAN = _OUT / "fetch_manifest_e0.json"
_QF = _OUT / "qhat_freeze.json"
_REPORT = _OUT / "e0_report.json"

SALT1 = bytes.fromhex("b64d061971c5e80dc95942d0c0e2ee0e")
SALT2 = bytes.fromhex("6dff3ac4564a27d4434d321b1eb46fff")
SALT3 = bytes.fromhex("6ed763131521c6c39ad7ea959c2dd22d")
DELTA = 0.1

TIER_A = {
    "awesome-prometheus-alerts": dict(
        url="https://github.com/samber/awesome-prometheus-alerts",
        family="prometheus",
        globs=["dist/rules/**/*.yml", "rules/**/*.yml"]),
    "ceph": dict(url="https://github.com/ceph/ceph", family="prometheus",
                 globs=["monitoring/ceph-mixin/prometheus_alerts.yml"]),
    "victoriametrics": dict(
        url="https://github.com/VictoriaMetrics/VictoriaMetrics",
        family="prometheus", globs=["deployment/docker/rules/*.yml"]),
    "gitlab-runbooks": dict(
        url="https://gitlab.com/gitlab-com/runbooks",
        family="prometheus",
        globs=["rules/*.yml", "legacy-prometheus-rules/*.yml"]),
    "loki": dict(url="https://github.com/grafana/loki", family="prometheus",
                 globs=["production/loki-mixin-compiled/alerts.yaml",
                        "operations/loki-mixin-compiled/alerts.yaml"]),
    "tempo": dict(url="https://github.com/grafana/tempo",
                  family="prometheus",
                  globs=["operations/tempo-mixin-compiled/alerts.yaml"]),
    "rook": dict(url="https://github.com/rook/rook", family="prometheus",
                 globs=["deploy/examples/monitoring/*rules*.yml",
                        "deploy/examples/monitoring/*rules*.yaml"]),
    "tidb": dict(url="https://github.com/pingcap/tidb",
                 family="prometheus",
                 globs=["metrics/alertmanager/*.rules.yml"]),
}
TIER_B = {
    "cockroach": dict(url="https://github.com/cockroachdb/cockroach",
                      family="prometheus",
                      globs=["monitoring/rules/*.rules.yml"]),
    "scylla-monitoring": dict(
        url="https://github.com/scylladb/scylla-monitoring",
        family="prometheus", globs=["prometheus/prom_rules/*.yml"]),
    "m3": dict(url="https://github.com/m3db/m3", family="prometheus",
               globs=["integrations/prometheus/*.yml"]),
}


def _assign(cluster_id: str) -> str:
    h = hmac_mod.new(SALT1, cluster_id.encode(), hashlib.sha256).digest()
    return "T" if h[0] % 3 == 2 else "C"


def _rep_key(salt: bytes, cluster_id: str, name: str, idx: int) -> bytes:
    return hmac_mod.new(salt, f"{cluster_id}|{name}|{idx}".encode(),
                        hashlib.sha256).digest()


def fetch_tier(tier: dict, man: dict):
    for name, spec in tier.items():
        if name in man["repos"]:
            continue
        try:
            repo_dir = _clone_or_reuse(name, spec["url"])
            entries = _collect_files(repo_dir, name, spec)
            man["repos"][name] = dict(url=spec["url"],
                                      commit=_head_sha(repo_dir),
                                      n_files=len(entries))
            man["files"].extend(e.__dict__ for e in entries)
        except Exception as ex:  # noqa: BLE001 -- repo may be unavailable
            man["repos"][name] = dict(url=spec["url"], error=str(ex)[:200])


def parse_file(fe: dict, commit: str):
    """Yield PromTarget per in-DSL alert with text in one file."""
    out, idx = [], 0
    try:
        docs = list(yaml.safe_load_all(open(fe["abs_path"],
                                            errors="replace")))
    except yaml.YAMLError:
        return out
    for doc in docs:
        for grp in _rule_groups(doc):
            for r in (grp or {}).get("rules", []) or []:
                if "alert" not in r or "expr" not in r:
                    continue
                parsed = parse_prom_expr(str(r["expr"]))
                ann = r.get("annotations", {}) or {}
                text = _strip_template(" ".join(
                    str(ann.get(k, "")) for k in
                    ("summary", "description", "message")))
                if parsed is None or not text:
                    continue
                reading = PromReading(
                    metric=parsed["metric"], selectors=parsed["selectors"],
                    comparator=parsed["comparator"],
                    threshold=parsed["threshold"],
                    rate_window_s=parsed["rate_window_s"],
                    aggregation=parsed["aggregation"],
                    agg_by=parsed["agg_by"],
                    for_s=dur_to_s(r.get("for")) or 0.0,
                    severity=str((r.get("labels", {}) or {}).get(
                        "severity", "")))
                out.append((idx, PromTarget(
                    name=r["alert"], group=(grp or {}).get("name", ""),
                    reading=reading, text=text, repo=fe["repo"],
                    commit=commit, file=fe["rel_path"],
                    sub_source=(grp or {}).get("name", ""),
                    last_commit_ts=fe.get("last_commit_ts"),
                    raw_expr=re.sub(r"\s+", " ", str(r["expr"]).strip()))))
                idx += 1
    return out


def clusters_of(man: dict, side: str):
    """side in {C, T}: parse only files assigned to that side."""
    cl = defaultdict(list)
    for fe in man["files"]:
        cid = f"{fe['repo']}/{fe['rel_path']}"
        if _assign(cid) != side:
            continue
        commit = man["repos"][fe["repo"]].get("commit", "")
        for idx, t in parse_file(fe, commit):
            cl[cid].append((idx, t))
    return {cid: rules for cid, rules in cl.items() if rules}


def sem_rep(cid, rules):
    return min(rules, key=lambda it: _rep_key(SALT2, cid, it[1].name,
                                              it[0]))[1]


def ctrl_eligible(t: PromTarget) -> bool:
    r = t.reading
    return (r.comparator in (">", ">=") and (r.aggregation or "none") ==
            "none" and 0 <= r.for_s <= 3600 and r.threshold > 0)


def ctrl_rep(cid, rules):
    el = [(i, t) for i, t in rules if ctrl_eligible(t)]
    if not el:
        return None
    return min(el, key=lambda it: _rep_key(SALT3, cid, it[1].name,
                                           it[0]))[1]


def gold_class_score(t: PromTarget, bank) -> tuple:
    rec = _record(build_prom_pool(t, bank), t, _prom_skeleton(t))
    return rec


def stage_cal():
    _OUT.mkdir(parents=True, exist_ok=True)
    man = dict(repos={}, files=[])
    fetch_tier(TIER_A, man)
    cl_C = clusters_of(man, "C")
    if len(cl_C) < 47:  # projected total < ~70 -> ladder
        fetch_tier(TIER_B, man)
        cl_C = clusters_of(man, "C")
    _MAN.write_text(json.dumps(man, indent=1, default=str))
    devs = dev_targets()
    bank = prom_threshold_bank(devs)
    reps = {cid: sem_rep(cid, rules) for cid, rules in cl_C.items()}
    scores = {}
    for cid, t in reps.items():
        rec = gold_class_score(t, bank)
        scores[cid] = rec.gold_score
    n = len(scores)
    k = max(1, int((n + 1) * DELTA))
    q = sorted(scores.values())[k - 1]
    level = 1 - k / (n + 1)
    _QF.write_text(json.dumps(dict(
        n_cal_clusters=n, k=k, qhat_sem=q, level=round(level, 4),
        per_cluster_scores={c: round(s, 6) for c, s in
                            sorted(scores.items())},
        repos={r: v for r, v in man["repos"].items()},
    ), indent=1))
    print(f"E0-C: {n} clusters, k={k}, qhat_sem={q:.6f}, "
          f"level={level:.4f}")
    print("repos:", {r: v.get("n_files", v.get("error", "?")[:40])
                     for r, v in man["repos"].items()})


def stage_test():
    assert _QF.exists(), "freeze q_sem first (--stage cal)"
    qf = json.loads(_QF.read_text())
    q = qf["qhat_sem"]
    man = json.loads(_MAN.read_text())
    devs = dev_targets()
    bank = prom_threshold_bank(devs)
    vocab = build_vocab(devs)
    cl_T = clusters_of(man, "T")
    reps = {cid: sem_rep(cid, rules) for cid, rules in cl_T.items()}

    # primary: cluster-level retention on semantic representatives
    ret_bits, set_sizes, raw_sizes, abst = {}, [], [], 0
    dom_stats, nondom = [], 0
    per_repo = defaultdict(lambda: [0, 0])
    for cid, t in reps.items():
        rec = gold_class_score(t, bank)
        hit = rec.gold_score >= q
        ret_bits[cid] = bool(hit)
        keep = [i for i, s in enumerate(rec.class_scores) if s >= q]
        set_sizes.append(len(keep))
        raw_sizes.append(len(rec.class_scores))
        abst += (len(keep) == 0)
        d = pool_dominance(rec.vectors)
        dom_stats.append(d)
        nondom += (not d["has_dominating_member"])
        per_repo[t.repo][1] += 1
        per_repo[t.repo][0] += int(hit)

    n = len(ret_bits)
    hits = sum(ret_bits.values())
    from scipy.stats import beta as _beta
    lo = _beta.ppf(0.025, hits, n - hits + 1) if hits else 0.0
    hi = _beta.ppf(0.975, hits + 1, n - hits) if hits < n else 1.0

    # rule-level empirical (all T in-DSL rules) + min-score variant
    rule_bits, cluster_rule_bits, min_bits = [], defaultdict(list), {}
    for cid, rules in cl_T.items():
        gs = []
        for _i, t in rules:
            rec = gold_class_score(t, bank)
            b = rec.gold_score >= q
            rule_bits.append(b)
            cluster_rule_bits[cid].append(b)
            gs.append(rec.gold_score)
        min_bits[cid] = min(gs) >= q
    # cluster bootstrap over rule-level bits
    rng = np.random.default_rng(0)
    cids = list(cluster_rule_bits)
    boots = []
    for _ in range(2000):
        pick = rng.choice(len(cids), len(cids), replace=True)
        bits = [b for j in pick for b in cluster_rule_bits[cids[j]]]
        boots.append(np.mean(bits))
    # secondary: independent-generator recall/containment on reps
    g_rec = g_cont = 0
    for cid, t in reps.items():
        pool, _m = generate(dict(text=t.text, name=t.name,
                                 severity=t.reading.severity), vocab)
        keys = {proj(r) for r in pool}
        if proj(t.reading) in keys:
            g_rec += 1
            sc = [  # containment: gold retained under q among generated?
                s for s in [None]]  # placeholder; retention needs pool scores
    # containment for the generator = recall x (gold scored >= q under its
    # own generated pool). Compute directly:
    g_cont = 0
    from .score import score_prom

    class _T:  # duck target for score_prom
        def __init__(self, text):
            self.text = text
    for cid, t in reps.items():
        pool, _m = generate(dict(text=t.text, name=t.name,
                                 severity=t.reading.severity), vocab)
        keys = {proj(r): i for i, r in enumerate(pool)}
        gi = keys.get(proj(t.reading))
        if gi is not None:
            s = score_prom(_T(t.text), pool[gi])[0]
            if s >= q:
                g_cont += 1

    # seed sensitivity: 20 alternative salt pairs (disclosure only)
    sens = []
    for j in range(20):
        s1 = hashlib.sha256(f"alt1-{j}".encode()).digest()[:16]
        s2 = hashlib.sha256(f"alt2-{j}".encode()).digest()[:16]
        qs, sizes_j = [], []
        # re-split + re-pick over the union of parsed C and T clusters is
        # not possible without re-parsing C here; sensitivity is over the
        # representative pick within T clusters at the frozen q (set sizes)
        for cid, rules in cl_T.items():
            rp = min(rules, key=lambda it: hmac_mod.new(
                s2, f"{cid}|{it[1].name}|{it[0]}".encode(),
                hashlib.sha256).digest())[1]
            rec = gold_class_score(rp, bank)
            sizes_j.append(sum(1 for s in rec.class_scores if s >= q))
        sens.append(round(float(np.mean(sizes_j)), 3))

    ctrl = {cid: ctrl_rep(cid, rules) for cid, rules in cl_T.items()}
    ctrl = {c: t for c, t in ctrl.items() if t is not None}
    (_OUT / "ctrl_reps.json").write_text(json.dumps(
        {c: dict(name=t.name, repo=t.repo, file=t.file,
                 raw_expr=t.raw_expr, text=t.text,
                 reading=dict(metric=t.reading.metric,
                              comparator=t.reading.comparator,
                              threshold=t.reading.threshold,
                              for_s=t.reading.for_s,
                              window=t.reading.rate_window_s))
         for c, t in sorted(ctrl.items())}, indent=1))

    rep = dict(
        manifest="A6 @ 669ef94; qhat freeze " + qf.get("frozen_at", ""),
        n_test_clusters=n, retention_hits=hits,
        conditional_retention=round(hits / n, 4),
        exact_binomial_95=[round(lo, 4), round(hi, 4)],
        target_level=qf["level"], qhat_sem=q,
        n_cal_clusters=qf["n_cal_clusters"], k=qf["k"],
        abstention_rate=round(abst / n, 4),
        mean_set=round(float(np.mean(set_sizes)), 2),
        mean_raw=round(float(np.mean(raw_sizes)), 2),
        set_reduction=round(1 - float(np.mean(set_sizes)) /
                            float(np.mean(raw_sizes)), 4),
        nondominated_frac=round(nondom / n, 4),
        rule_level=dict(n=len(rule_bits),
                        coverage=round(float(np.mean(rule_bits)), 4),
                        cluster_boot_95=[round(float(np.percentile(
                            boots, 2.5)), 4), round(float(np.percentile(
                                boots, 97.5)), 4)]),
        min_score_all_covered=round(
            sum(min_bits.values()) / len(min_bits), 4),
        generator_secondary=dict(
            recall=round(g_rec / n, 4), containment=round(g_cont / n, 4),
            dev_reference=dict(capped=0.2078, ceiling=0.695)),
        per_repo={k: dict(hit=v[0], n=v[1]) for k, v in per_repo.items()},
        cluster_sizes=dict(
            C=qf.get("n_cal_clusters"),
            T_dist=sorted(Counter(len(r) for r in cl_T.values()).items())),
        seed_sensitivity_mean_set=dict(
            frozen=round(float(np.mean(set_sizes)), 3),
            alt_salts=sens),
        n_ctrl_eligible_reps=len(ctrl),
    )
    _REPORT.write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in
                      ("n_test_clusters", "conditional_retention",
                       "exact_binomial_95", "target_level",
                       "abstention_rate", "mean_set", "set_reduction",
                       "nondominated_frac", "min_score_all_covered",
                       "generator_secondary", "n_ctrl_eligible_reps")},
                     indent=1))
    print("rule-level:", rep["rule_level"])
    print("per-repo:", rep["per_repo"])


if __name__ == "__main__":
    stage = sys.argv[sys.argv.index("--stage") + 1] \
        if "--stage" in sys.argv else "cal"
    (stage_cal if stage == "cal" else stage_test)()
