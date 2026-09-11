"""Official-CLI cross-check for the source-grounded benchmark (plan 3.1/3.2
steps "run promtool check rules" / "run the repository's Kyverno tests").

Runs the OFFICIAL binaries -- promtool (Prometheus) and kyverno (CLI) -- over
the pinned artifacts in data/rule_corpora, confirming that
every source target we parse is a valid, executable rule and that Kyverno
policies pass their own committed test fixtures. Writes
results/conformal/benchmark_sg/cli_crosscheck.json.

Requires promtool and kyverno on PATH.
Run: python3 -m saorl.benchmark_sg.cli_crosscheck [max_kyverno]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC = ROOT / "data" / "rule_corpora"
OUT = ROOT / "results/conformal" / "benchmark_sg" / "cli_crosscheck.json"
TMP = Path("/tmp/_promrule.yaml")


def promtool_check():
    files = sorted((SRC / "kube-prometheus").rglob("*prometheusRule*.yaml"))
    ok = fail = rules = 0
    failures = []
    for f in files:
        try:
            d = yaml.safe_load(open(f))
            groups = d.get("spec", {}).get("groups", [])
        except Exception as e:
            failures.append(f"{f.name}: parse {e}"); fail += 1; continue
        TMP.write_text(yaml.safe_dump({"groups": groups}))
        r = subprocess.run(["promtool", "check", "rules", str(TMP)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            ok += 1
            m = re.search(r"(\d+)\s+rules found", r.stdout + r.stderr)
            rules += int(m.group(1)) if m else sum(len(g.get("rules", [])) for g in groups)
        else:
            fail += 1; failures.append(f"{f.name}: {(r.stdout + r.stderr).strip()[:120]}")
    return dict(n_files=len(files), files_ok=ok, files_failed=fail,
                rules_validated=rules, failures=failures)


def kyverno_test(limit=None):
    dirs = sorted({p.parent for p in (SRC / "kyverno-policies").rglob(".kyverno-test/kyverno-test.yaml")})
    if limit:
        dirs = dirs[:limit]
    ok = fail = 0
    failures = []
    for d in dirs:
        r = subprocess.run(["kyverno", "test", str(d)], capture_output=True, text=True)
        out = r.stdout + r.stderr
        failed = re.search(r"(\d+)\s+tests failed", out)
        if r.returncode == 0 and (not failed or failed.group(1) == "0"):
            ok += 1
        else:
            fail += 1
            if len(failures) < 10:
                failures.append(f"{d.name}: {out.strip().splitlines()[-1][:100] if out.strip() else 'rc=%d' % r.returncode}")
    return dict(n_dirs=len(dirs), dirs_ok=ok, dirs_failed=fail, failures=failures)


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    rep = dict(
        promtool=promtool_check(),
        kyverno=kyverno_test(limit),
        note=("Official-CLI verification of the pinned source artifacts "
              "(promtool check rules; kyverno test). Supplements the "
              "deterministic Python evaluators used for candidate execution."),
    )
    OUT.write_text(json.dumps(rep, indent=1))
    p, k = rep["promtool"], rep["kyverno"]
    print(f"promtool check rules: {p['files_ok']}/{p['n_files']} files OK, "
          f"{p['rules_validated']} rules validated, {p['files_failed']} failed")
    print(f"kyverno test: {k['dirs_ok']}/{k['n_dirs']} policy fixtures passed, "
          f"{k['dirs_failed']} failed")
    if p["failures"]:
        print("prom failures:", p["failures"][:3])
    if k["failures"]:
        print("kyv failures:", k["failures"][:3])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
