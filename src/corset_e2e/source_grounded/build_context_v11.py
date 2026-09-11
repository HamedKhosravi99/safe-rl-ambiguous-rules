"""v11 context profiles: what the repository SAYS about each metric.

The v3/v4/v5 campaigns established that availability is not the wall
(0.83 of gold metrics sit in the v4 catalog) and catalog precision is not
the wall (v5); the lexical license simply cannot connect rule language to
metric names that do not resemble it (REGISTRATION_V6, measured three
times).  The v6 LLM licenser proved the missing evidence exists in
target-blind data.  This module harvests that evidence deterministically:
for every catalog metric, the word context of the lines where the metric
occurs in the repository's NON-rule files (docs, dashboards, source metric
definitions) -- e.g. HELP strings, dashboard panel titles, doc prose --
becomes a token profile the v11 licenser can match rule language against.

Target-blindness is inherited from the v4 harvest rule: any path
containing 'alert' or 'rule' is excluded wholesale, so no context is ever
read from the artifact family the targets come from.  Profiles attach
ONLY to names already in the committed catalogs of record
(catalog_v4/*.tokens union the v3 expression harvest), so availability is
untouched -- this changes evidence for ranking, not the candidate space.

Run: PYTHONPATH=. python3 corset_e2e/source_grounded/build_context_v11.py
Writes results/e2e/context_v11/<repo>.ctx.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.source_grounded.harvest_tokens import (  # noqa: E402
    SRC, _TOKEN_RE, is_excluded)

OUT = os.path.join(ROOT, "results/e2e")
CTX_DIR = os.path.join(OUT, "context_v11")

_WORD_RE = re.compile(r"[a-z][a-z0-9]+")
_HEAD_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
STOP = frozenset(
    "the a an of to and or for in on is are was were be been this that these "
    "those with by if it as at we you your our their its from not no can may "
    "will would should could than then when where which who whom what how all "
    "any each per more most other some such only own same so too very s t "
    "just also into over under between during before after above below out "
    "off up down again further once here there both few because until while "
    "about against does did doing have has had having do value values true "
    "false null name names type types default example examples set new "
    "https http github com www org io".split())
PROFILE_CAP = 48         # tokens kept per metric profile
LINE_CAP = 30            # context tokens taken from any single line


def line_tokens(line: str):
    return [t for t in _WORD_RE.findall(line.lower())
            if len(t) > 1 and t not in STOP][:LINE_CAP]


def metrics_of_record(repo: str) -> set:
    names = set(open(os.path.join(OUT, "catalog_v4", f"{repo}.tokens"))
                .read().split())
    v3 = json.load(open(os.path.join(OUT, "metric_catalog_v3.json")))
    for cid, ms in v3["cluster_metrics"].items():
        if v3["cluster_repo"].get(cid) == repo:
            names |= set(ms)
    return names


def harvest_context(repo: str) -> dict:
    universe = metrics_of_record(repo)
    prof: dict = defaultdict(Counter)
    occ: Counter = Counter()
    rdir = os.path.join(SRC, repo)
    for dirpath, dirnames, filenames in os.walk(rdir):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, rdir).lower()
            if is_excluded(rel):
                continue
            try:
                text = open(full, encoding="utf8", errors="ignore").read()
            except OSError:
                continue
            is_md = rel.endswith((".md", ".markdown"))
            heading: list = []
            for line in text.splitlines():
                if is_md:
                    hm = _HEAD_RE.match(line)
                    if hm:
                        heading = line_tokens(hm.group(1))
                        continue
                hits = {m.group(1).rstrip("_") for m in _TOKEN_RE.finditer(line)}
                hits &= universe
                if not hits:
                    continue
                ctx = line_tokens(line) + heading
                for m in hits:
                    occ[m] += 1
                    prof[m].update(ctx)
    return dict(
        repo=repo, n_metrics_of_record=len(universe),
        n_with_context=len(prof),
        occurrences={m: occ[m] for m in prof},
        profiles={m: dict(c.most_common(PROFILE_CAP))
                  for m, c in prof.items()})


def main() -> None:
    os.makedirs(CTX_DIR, exist_ok=True)
    repos = sorted(f[:-len(".tokens")]
                   for f in os.listdir(os.path.join(OUT, "catalog_v4"))
                   if f.endswith(".tokens"))
    for repo in repos:
        payload = harvest_context(repo)
        path = os.path.join(CTX_DIR, f"{repo}.ctx.json")
        json.dump(payload, open(path, "w"))
        print(f"{repo}: {payload['n_with_context']}/"
              f"{payload['n_metrics_of_record']} metrics have context "
              f"({os.path.getsize(path) // 1024} KiB)")


if __name__ == "__main__":
    main()
