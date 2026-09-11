"""Harvest LABEL KEY names from non-rule files, per repository.

The join-label slot sat at 34.1% because the only vocabulary offered was
the 35 label sets the development split exhibits, and dev's keys turn out
to be almost disjoint from the rest of the corpus: of 2,868 gold label-set
occurrences in test, dev's sets match 65 exactly, and even dev's KEY
vocabulary covers only 67. Dev writes `namespace`, `pod`, `job`; the
deployments write `component`, `env`, `environment`, `stage`, `tier`,
`type`. A label schema is deployment-specific vocabulary in exactly the
way a metric name is, so it is retrieved the same way.

WHAT IS READ. Only files that the metric catalogue is already allowed to
read: every path containing "alert" or "rule" is skipped, so this never
opens the artifact family the targets come from. Within those files, label
keys are taken from grouping clauses (`by`, `without`, `on`, `ignoring`)
and from label matchers (`{key="..."}`), which is where dashboards and
service catalogues spell out a deployment's schema.

WHAT IS PRODUCED. Keys, not sets. Candidate sets are formed downstream by
enumerating subsets of a repository's most frequent keys, which is bounded
and auditable: the corpus uses 27 distinct sets over 31 keys, so subsets of
a dozen ranked keys cover the space without any target being consulted.

Run: PYTHONPATH=. python3 corset_e2e/source_grounded/harvest_label_keys.py
Writes results/e2e/label_keys.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PKG))
SRC = os.path.join(ROOT, "data", "rule_corpora")
OUT = os.path.join(ROOT, "results/e2e")

sys.path.insert(0, os.path.join(str(ROOT), "src"))
from corset_e2e.source_grounded.harvest_tokens import is_excluded  # noqa: E402

_GROUPING = re.compile(
    r"\b(?:by|without|on|ignoring|group_left|group_right)\s*\(([^)]{0,300})\)")
# a PromQL selector block: a metric-like name followed by {...}. Matching
# bare `key="value"` anywhere would harvest XML attributes -- dashboards
# embed SVG, and the first version of this ranked `x`, `y`, `fill` and
# `stroke` above the deployment's real dimensions.
_SELECTOR = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*\s*\{([^{}]{0,400})\}")
_MATCHER = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:=~|!~|=|!=)\s*[\"']")
_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")

# keys that are syntax or configuration, not deployment dimensions
STOP = frozenset((
    "expr", "record", "alert", "for", "labels", "annotations", "name",
    "groups", "rules", "severity", "summary", "description", "message",
    "runbook", "dashboard", "title", "value", "unit", "format", "legend",
    "datasource", "interval", "refid", "target", "targets", "panels",
    "type_", "apiversion", "kind", "metadata", "spec", "http", "https"))
MAX_FILE_BYTES = 2_000_000
# markup carries attribute names that look like label matchers
SKIP_EXT = (".svg", ".html", ".htm", ".xml", ".png", ".jpg", ".jpeg", ".pdf",
            ".drawio", ".css", ".woff", ".woff2", ".ttf", ".ico")


def keys_in_text(text: str) -> Counter:
    out = Counter()
    for m in _GROUPING.finditer(text):
        for part in m.group(1).split(","):
            k = part.strip().strip("\"'")
            if _NAME.match(k) and k not in STOP:
                out[k] += 2          # a grouping clause is direct evidence
    for sel in _SELECTOR.finditer(text):
        for m in _MATCHER.finditer(sel.group(1)):
            k = m.group(1)
            if _NAME.match(k) and k not in STOP:
                out[k] += 1
    return out


def harvest() -> dict:
    out = defaultdict(Counter)
    repos = sorted(d for d in os.listdir(SRC)
                   if os.path.isdir(os.path.join(SRC, d)))
    for repo in repos:
        rroot = os.path.join(SRC, repo)
        for dirpath, dirnames, filenames in os.walk(rroot):
            dirnames[:] = [d for d in dirnames
                           if d not in (".git", "vendor", "node_modules")]
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, rroot).lower()
                if is_excluded(rel):
                    continue          # never the targets' artifact family
                if rel.endswith(SKIP_EXT):
                    continue
                try:
                    if os.path.getsize(full) > MAX_FILE_BYTES:
                        continue
                    with open(full, encoding="utf8", errors="ignore") as fh:
                        text = fh.read()
                except OSError:
                    continue
                out[repo].update(keys_in_text(text))
    return {r: dict(c.most_common()) for r, c in out.items()}


def main() -> None:
    table = harvest()
    json.dump(table, open(os.path.join(OUT, "label_keys.json"), "w"), indent=0)
    for repo, c in sorted(table.items()):
        top = list(c)[:12]
        print(f"{repo:<30s} {len(c):5d} keys   top: {top}")
    print("\nwrote results/e2e/label_keys.json")


if __name__ == "__main__":
    main()
