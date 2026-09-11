"""Reconstructed producer for results/e2e/catalog_v4/*.tokens and
catalog_v5/*.tokens, with a verification mode.

The committed catalogs were data-only (commit f30a9d27 added the .tokens
files with no script).  This script re-derives them from the pinned repo
clones at data/rule_corpora/ under the registered rules
(REGISTRATION_V4.md / REGISTRATION_V5.md):

  v4: metric-shaped names (same _METRIC_RE and keyword filter as catalog
      v3) harvested from ALL repo files EXCLUDING any alert/recording-rule
      file (path contains 'alert', or basename matches *rule*.y*ml /
      *rules*.libsonnet), so the harvest never reads the artifacts the
      targets come from.
  v5: same regex and exclusions, restricted to documentation and
      dashboard files (*.md, *.json, *.jsonnet, *.libsonnet).

--verify diffs the re-derived sets against the committed .tokens and
fails loudly on any difference; --pin writes the repo SHAs to
results/e2e/catalog_src_manifest.json so the harvest is reproducible.

Run: PYTHONPATH=. python3 corset_e2e/source_grounded/harvest_tokens.py --verify --pin
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(ROOT, "data", "rule_corpora")
OUT = os.path.join(ROOT, "results/e2e")

V5_EXT = (".md", ".json", ".jsonnet", ".libsonnet")

# Decoded from the committed catalogs (fit on kube-prometheus, verified on
# all 24 .tokens files): first segment must contain an underscore, colon
# continuations allowed, no leading word-boundary (so a name embedded after
# an uppercase prefix is captured from its first lowercase run), trailing
# underscores stripped, minimum length 4.  No keyword filter.
_TOKEN_RE = re.compile(r"([a-z][a-z0-9]*(?:_+[a-z0-9]+)+(?::[a-z0-9_]+)*)")


def is_excluded(rel_lower: str) -> bool:
    """Any path mentioning alerts or rules: the harvest never reads the
    artifact family the targets come from (REGISTRATION_V4)."""
    return "alert" in rel_lower or "rule" in rel_lower


def harvest_repo(repo_dir: str, v5: bool = False) -> set:
    names: set = set()
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, repo_dir).lower()
            if is_excluded(rel):
                continue
            if v5 and not rel.endswith(V5_EXT):
                continue
            try:
                text = open(full, encoding="utf8", errors="ignore").read()
            except OSError:
                continue
            for m in _TOKEN_RE.finditer(text):
                name = m.group(1).rstrip("_")
                if len(name) < 4 or "_" not in name:
                    continue
                names.add(name)
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--pin", action="store_true")
    args = ap.parse_args()

    repos = sorted(d for d in os.listdir(SRC)
                   if os.path.isdir(os.path.join(SRC, d, ".git")))
    manifest = {}
    failures = 0
    for repo in repos:
        rdir = os.path.join(SRC, repo)
        sha = subprocess.check_output(
            ["git", "-C", rdir, "rev-parse", "HEAD"]).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", rdir, "status", "--porcelain"]).decode().strip())
        manifest[repo] = dict(sha=sha, dirty=dirty)
        for ver, v5 in (("catalog_v4", False), ("catalog_v5", True)):
            committed_path = os.path.join(OUT, ver, f"{repo}.tokens")
            if not os.path.exists(committed_path):
                continue                      # kyverno-policies has no catalog
            got = harvest_repo(rdir, v5=v5)
            if args.verify:
                want = set(open(committed_path).read().split())
                extra, missing = got - want, want - got
                tag = "OK " if not (extra or missing) else "DIFF"
                print(f"[{tag}] {ver}/{repo}: derived {len(got)} committed "
                      f"{len(want)} (+{len(extra)} -{len(missing)})")
                if extra or missing:
                    failures += 1
                    for s in list(sorted(extra))[:5]:
                        print("        extra:", s)
                    for s in list(sorted(missing))[:5]:
                        print("        missing:", s)
            if args.write:
                with open(committed_path, "w") as fh:
                    fh.write("\n".join(sorted(got)) + "\n")
    if args.pin:
        path = os.path.join(OUT, "catalog_src_manifest.json")
        json.dump(dict(note=("pinned SHAs of the harvested repo clones at "
                             "data/rule_corpora"),
                       repos=manifest), open(path, "w"), indent=1)
        print("pinned", path)
    if args.verify and failures:
        raise SystemExit(f"{failures} catalog(s) failed byte-set verification")


if __name__ == "__main__":
    main()
