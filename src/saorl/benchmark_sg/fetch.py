"""Source-grounded benchmark, step 1: fetch pinned public artifacts.

Shallow/partial-clones two public policy repositories into the benchmark data
directory and records an exact provenance manifest (repo URL, resolved commit
SHA, and the list of rule/policy files with their last-commit timestamps, so a
downstream *chronological* split can order rules by real repository recency).

Repositories (default branch HEAD at clone time is pinned):
  * prometheus-operator/kube-prometheus   -- alert rules in manifests/*-prometheusRule.yaml
  * kyverno/policies                       -- ClusterPolicy / Policy YAMLs

We use ``git clone --filter=blob:none`` (a *partial* clone): the full commit
graph is fetched (so ``git log -1 --format=%ct -- <path>`` gives an honest
per-file last-commit time for the chronological split) while file blobs are
fetched lazily at checkout.  This is cheap and keeps real provenance.  If the
network is unavailable and a previous manifest exists, the cached clone is
reused.

No promtool / kyverno CLI binary is required or invoked here.

Run:  PYTHONPATH=. python3 -m saorl.benchmark_sg.fetch
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = _ROOT / "data" / "rule_corpora"
MANIFEST = _ROOT / "results/conformal" / "benchmark_sg" / "fetch_manifest.json"

REPOS = {
    "kube-prometheus": dict(
        url="https://github.com/prometheus-operator/kube-prometheus",
        family="prometheus",
        # PrometheusRule CRDs live here; each carries spec.groups[].rules[]
        globs=["manifests/*-prometheusRule.yaml", "manifests/*prometheusRule*.yaml"],
    ),
    "kyverno-policies": dict(
        url="https://github.com/kyverno/policies",
        family="kyverno",
        globs=["**/*.yaml"],  # filtered to ClusterPolicy/Policy in _kyverno_files
    ),
}


@dataclass
class FileEntry:
    repo: str
    family: str
    rel_path: str          # path relative to repo root
    abs_path: str          # absolute path on disk
    last_commit_ts: Optional[int]  # unix seconds of the file's last commit


def _run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True,
                          capture_output=True, text=True).stdout


def _clone_or_reuse(name: str, url: str) -> Path:
    dest = SRC_DIR / name
    if (dest / ".git").exists():
        return dest
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    # partial clone: full history metadata, lazy blobs (keeps real dates cheaply)
    try:
        _run(["git", "clone", "--filter=blob:none", "--quiet", url, str(dest)])
    except subprocess.CalledProcessError:
        # fall back to a shallow snapshot (loses per-file dates)
        _run(["git", "clone", "--depth", "1", "--quiet", url, str(dest)])
    return dest


def _head_sha(repo_dir: Path) -> str:
    return _run(["git", "rev-parse", "HEAD"], cwd=repo_dir).strip()


def _last_commit_ts(repo_dir: Path, rel_path: str) -> Optional[int]:
    try:
        out = _run(["git", "log", "-1", "--format=%ct", "--", rel_path], cwd=repo_dir).strip()
        return int(out) if out else None
    except (subprocess.CalledProcessError, ValueError):
        return None


def _is_kyverno_policy(path: Path) -> bool:
    """A ClusterPolicy/Policy manifest (skip tests, kustomize, chainsaw, charts)."""
    name = path.name
    if name in ("kyverno-test.yaml", "kustomization.yaml", "Chart.yaml", "values.yaml"):
        return False
    lowered = str(path).lower()
    if any(seg in lowered for seg in (".chainsaw", "/chainsaw", "artifacthub",
                                      "/charts/", "resource.yaml", "/.github/")):
        return False
    try:
        head = path.read_text(errors="ignore")[:4000]
    except OSError:
        return False
    return ("kind: ClusterPolicy" in head or "kind: Policy" in head) and "spec:" in head


def _collect_files(repo_dir: Path, name: str, spec: dict) -> List[FileEntry]:
    fam = spec["family"]
    entries: List[FileEntry] = []
    seen: set = set()
    for g in spec["globs"]:
        for p in sorted(repo_dir.glob(g)):
            if not p.is_file() or p in seen:
                continue
            if fam == "kyverno" and not _is_kyverno_policy(p):
                continue
            seen.add(p)
            rel = str(p.relative_to(repo_dir))
            entries.append(FileEntry(
                repo=name, family=fam, rel_path=rel, abs_path=str(p),
                last_commit_ts=_last_commit_ts(repo_dir, rel)))
    return entries


def fetch(force: bool = False) -> dict:
    """Clone (or reuse) both repos and return the provenance manifest dict."""
    repos_meta: Dict[str, dict] = {}
    all_files: List[dict] = []
    for name, spec in REPOS.items():
        repo_dir = _clone_or_reuse(name, spec["url"])
        sha = _head_sha(repo_dir)
        files = _collect_files(repo_dir, name, spec)
        repos_meta[name] = dict(url=spec["url"], family=spec["family"],
                                commit=sha, n_files=len(files),
                                local_dir=str(repo_dir))
        all_files.extend(asdict(f) for f in files)
    manifest = dict(
        repos=repos_meta,
        n_files_total=len(all_files),
        files=all_files,
        note="Partial clone (--filter=blob:none); commit == default-branch HEAD "
             "at clone time. last_commit_ts is the file's last commit (unix s) "
             "for the chronological split.",
    )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(MANIFEST, "w"), indent=1)
    return manifest


def load_manifest() -> dict:
    if not MANIFEST.exists():
        return fetch()
    return json.load(open(MANIFEST))


def main() -> None:
    m = fetch()
    print(f"wrote {MANIFEST}")
    for name, meta in m["repos"].items():
        print(f"  {name:20s} commit={meta['commit'][:12]}  files={meta['n_files']}  "
              f"({meta['family']})")
    print(f"  total candidate files: {m['n_files_total']}")


if __name__ == "__main__":
    main()
