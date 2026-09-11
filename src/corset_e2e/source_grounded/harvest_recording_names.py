"""Harvest DECLARED RECORDING-RULE NAMES, and nothing else, per cluster.

Why this channel exists. The slot autopsy showed metric availability at
6.7%, and the names that dominate the misses are recording rules --
`slo:max:hard:gitlab_component_saturation:ratio` (1,255 units),
`gitlab_component_ops:rate_6h` (450), `gitlab_component_errors:ratio_1h`
(242). The v4 catalogue harvests metric names from non-rule files only
(harvest_tokens.is_excluded rejects any path containing "alert" or
"rule"), because rule files are the artifact family the targets come from.
That exclusion works, and it also discards the entire vocabulary these
deployments alert in: a recording rule DECLARES a series name, and modern
SLO alerting is written almost exclusively against such names.

WHAT IS READ, EXACTLY. For every rule file, for every mapping in
`groups[].rules[]` that has a `record` key, this reads the STRING VALUE OF
`record` and nothing else. It never reads `expr` -- neither the target
alert's expression nor the defining right-hand side of the recording rule
itself -- and never reads `alert`, `labels`, `annotations` or `for`. The
analogy is reading a module's exported symbol table, not its source: the
generator learns that a series named `slo:max:hard:...` exists in this
deployment, and learns nothing about what it computes or which alert fires
on it.

WHAT IS EXCLUDED. Names are indexed by cluster (repo:relpath), and a unit
is served the union over its repository MINUS its own cluster, the same
leave-one-cluster-out rule the v3/v4 catalogues already use. So a unit can
never be served a name declared in the very file its target was taken
from.

This does not weaken the leak audits; it is subject to them. See
generator/leak_audit_recording.py, which checks by construction and by
target-swap that no expression text reaches the vocabulary.

Run: PYTHONPATH=. python3 corset_e2e/source_grounded/harvest_recording_names.py
Writes results/e2e/recording_names.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PKG))
SRC = os.path.join(ROOT, "data", "rule_corpora")
OUT = os.path.join(ROOT, "results/e2e")

# a declared series name; anything else in the mapping is not read
RECORD_KEY = "record"
ALLOWED_KEYS = (RECORD_KEY,)


def _groups_of(doc):
    if not isinstance(doc, dict):
        return None
    if isinstance(doc.get("spec"), dict) and "groups" in doc["spec"]:
        return doc["spec"]["groups"]
    if "groups" in doc:
        return doc["groups"]
    return None


def record_names_in_file(path: str):
    """Yield only the `record:` string values declared in this file."""
    try:
        with open(path, encoding="utf8", errors="ignore") as fh:
            docs = list(yaml.safe_load_all(fh))
    except (yaml.YAMLError, OSError, UnicodeDecodeError):
        return
    for doc in docs:
        groups = _groups_of(doc)
        if not isinstance(groups, list):
            continue
        for grp in groups:
            if not isinstance(grp, dict):
                continue
            for r in grp.get("rules", []) or []:
                if not isinstance(r, dict) or RECORD_KEY not in r:
                    continue
                name = r.get(RECORD_KEY)
                # read the declared symbol; never r["expr"], r["labels"], ...
                if isinstance(name, str) and name.strip():
                    yield name.strip()


def harvest() -> dict:
    """repo -> cluster_id -> sorted declared names."""
    out = defaultdict(lambda: defaultdict(set))
    repos = sorted(d for d in os.listdir(SRC)
                   if os.path.isdir(os.path.join(SRC, d)))
    for repo in repos:
        rroot = os.path.join(SRC, repo)
        for dirpath, dirnames, filenames in os.walk(rroot):
            dirnames[:] = [d for d in dirnames
                           if d not in (".git", "vendor", "node_modules")]
            for fn in filenames:
                if not fn.endswith((".yaml", ".yml")):
                    continue
                path = os.path.join(dirpath, fn)
                rel = os.path.relpath(path, rroot)
                cid = f"{repo}:{rel}"
                for name in record_names_in_file(path):
                    out[repo][cid].add(name)
    return {r: {c: sorted(v) for c, v in cs.items()} for r, cs in out.items()}


def names_for_unit(table: dict, repo: str, own_cluster: str) -> set:
    """Every name declared in this repo except in the unit's own cluster."""
    out = set()
    for cid, names in table.get(repo, {}).items():
        if cid == own_cluster:
            continue
        out.update(names)
    return out


def main() -> None:
    table = harvest()
    n_names = sum(len(v) for cs in table.values() for v in cs.values())
    distinct = len({x for cs in table.values() for v in cs.values() for x in v})
    print(f"repos {len(table)}  clusters {sum(len(c) for c in table.values())}")
    print(f"declared record names: {n_names} ({distinct} distinct)")
    for repo, cs in sorted(table.items()):
        d = len({x for v in cs.values() for x in v})
        if d:
            print(f"  {repo:<28s} {d:5d} distinct in {len(cs)} clusters")
    json.dump(table, open(os.path.join(OUT, "recording_names.json"), "w"),
              indent=0)
    print("\nwrote results/e2e/recording_names.json")


if __name__ == "__main__":
    main()
