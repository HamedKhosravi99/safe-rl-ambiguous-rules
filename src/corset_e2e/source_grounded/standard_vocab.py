"""A frozen external vocabulary of standard exporter metric names.

The dev vocabulary gate failed at C_V = 73.3%, and the misses were
dominated by series that ship with an exporter binary rather than being
written into any repository file -- Prometheus's own `up`, node_exporter
gauges, kube-state-metrics series. A catalogue that harvests names from
files cannot see a metric nobody writes down, however good its retrieval.

This adds those names from PUBLISHED INVENTORIES, not from the corpus and
not from the observed misses. The distinction is the whole point, so it is
enforced structurally:

  * each source is a pinned upstream commit, recorded with its URL and the
    SHA-256 of the exact bytes downloaded;
  * each source has ONE declared extraction rule, applied wholesale to
    every file of that source -- there is no per-metric decision anywhere
    in this module, so a name is either produced by the rule or absent;
  * the resulting vocabulary is IDENTICAL for every target, and is a
    function of the pinned commits alone.

A disclosure that belongs with the artifact rather than in a footnote: the
author of this module had already seen the dev misses (`up`,
node_filesystem_readonly, node_memory_MemTotal_bytes,
kube_horizontalpodautoscaler_spec_*) when writing it. That cannot be
undone, so independence is not claimed on the author's say-so; it rests on
the rules above, and on the reported fraction of this vocabulary that no
gold rule ever uses, which is what a curated list would not have.

Run: PYTHONPATH=. python3 corset_e2e/source_grounded/standard_vocab.py
Writes results/e2e/standard_vocab.json
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(ROOT, "results/e2e")

# ---- sources, pinned by repository and resolved to a commit at build time
SOURCES = {
    "kube-state-metrics": dict(
        repo="kubernetes/kube-state-metrics", ref="main",
        include=lambda p: p.startswith("docs/metrics/") and p.endswith(".md"),
        # the docs are markdown TABLES, one metric per row in column one;
        # the first rule looked for backticks and found 3 names in 40 files
        rule=r"(?m)^\|\s*(kube_[a-z0-9_]+)\s*\|"),
    "node_exporter": dict(
        repo="prometheus/node_exporter", ref="master",
        include=lambda p: p.startswith("collector/") and p.endswith(".go")
        and not p.endswith("_test.go"),
        # collectors ASSEMBLE names: BuildFQName(namespace, subsystem, "x")
        # with `subsystem` a per-file constant. Matching literals alone found
        # 10 names in 155 files. Handled by `assemble` below.
        rule=r'"(node_[a-z0-9_]+)"', assemble="node_exporter"),
    "prometheus": dict(
        repo="prometheus/prometheus", ref="main",
        include=lambda p: p.startswith("docs/") and p.endswith(".md"),
        rule=r"`(up|scrape_[a-z0-9_]+|prometheus_[a-z0-9_]+)`"),
}
# Minimum length 2, not 4. The first version required four characters and
# silently discarded `up` -- the single most-used metric in the corpus and
# one the Prometheus extraction rule names explicitly -- so a declared rule
# was being overridden by an undeclared length filter.
NAME_OK = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{1,}$")
TIMEOUT = 60


def _get(url: str) -> bytes:
    out = subprocess.run(["curl", "-sSL", "-m", str(TIMEOUT), url],
                         capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"fetch failed: {url}")
    return out.stdout


def resolve_commit(repo: str, ref: str) -> str:
    d = json.loads(_get(f"https://api.github.com/repos/{repo}/commits/{ref}"))
    return d["sha"]


def build_source(name: str, cfg: dict) -> dict:
    commit = resolve_commit(cfg["repo"], cfg["ref"])
    tree = json.loads(_get(
        f"https://api.github.com/repos/{cfg['repo']}/git/trees/{commit}"
        f"?recursive=1"))
    paths = sorted(t["path"] for t in tree.get("tree", [])
                   if t["type"] == "blob" and cfg["include"](t["path"]))
    rx = re.compile(cfg["rule"])
    sub_rx = re.compile(r'\bsubsystem\s*=\s*"([a-z0-9_]+)"')
    fq_rx = re.compile(r'BuildFQName\(\s*namespace\s*,\s*'
                       r'(?:subsystem|"([a-z0-9_]*)")\s*,\s*"([a-z0-9_]+)"')
    names, files = set(), []
    for p in paths:
        raw = _get(f"https://raw.githubusercontent.com/{cfg['repo']}/"
                   f"{commit}/{p}")
        files.append(dict(path=p, sha256=hashlib.sha256(raw).hexdigest(),
                          bytes=len(raw)))
        text = raw.decode("utf8", "ignore")
        for m in rx.finditer(text):
            n = m.group(1)
            if NAME_OK.match(n):
                names.add(n)
        if cfg.get("assemble") == "node_exporter":
            subs = sub_rx.findall(text)
            default_sub = subs[0] if subs else ""
            for lit, leaf in fq_rx.findall(text):
                sub = lit if lit else default_sub
                n = "_".join(x for x in ("node", sub, leaf) if x)
                if NAME_OK.match(n):
                    names.add(n)
    return dict(repo=cfg["repo"], ref=cfg["ref"], commit=commit,
                rule=cfg["rule"], n_files=len(files), n_names=len(names),
                files=files, names=sorted(names))


def main() -> None:
    prov = {}
    allnames = set()
    for name, cfg in SOURCES.items():
        print(f"[{name}] resolving {cfg['repo']}@{cfg['ref']} ...")
        s = build_source(name, cfg)
        prov[name] = s
        allnames |= set(s["names"])
        print(f"    commit {s['commit'][:12]}  {s['n_files']} files  "
              f"{s['n_names']} names")
    blob = "\n".join(sorted(allnames)).encode()
    rep = dict(n_names=len(allnames),
               vocabulary_sha256=hashlib.sha256(blob).hexdigest(),
               sources={k: {kk: vv for kk, vv in v.items() if kk != "names"}
                        for k, v in prov.items()},
               names=sorted(allnames))
    json.dump(rep, open(os.path.join(OUT, "standard_vocab.json"), "w"))
    print(f"\nSTANDARD VOCABULARY {len(allnames)} names, "
          f"sha256 {rep['vocabulary_sha256'][:16]}")
    print("wrote results/e2e/standard_vocab.json")


if __name__ == "__main__":
    main()
