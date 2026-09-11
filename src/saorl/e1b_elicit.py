"""E1b: the k=1 arm and the permutation null (REGISTRATION_V10 addendum).

A NEW elicitation study. The published harness is not in the release and its
per-rule readings were never archived, so this cannot reproduce the published
0.37; what it can do is the contrast the objection turns on, k=1 against k=3
under one harness, one rule set, one prompt family.

Rule set: the 30 alphabetically first kube-prometheus alerts in the pinned
_src/ tree carrying summary, description and expr.

Run:  PYTHONPATH=. python3 -m saorl.e1b_elicit           # query + archive
      PYTHONPATH=. python3 -m saorl.e1b_elicit --analyze # endpoints only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RULES = ROOT / "results/e2e" / "e1b_rules.json"
RAW = ROOT / "results/e2e" / "e1b_readings.jsonl"
OUT = ROOT / "results/e2e" / "e1b_elicitation.json"

PERSONAS = [
    "a site reliability engineer who owns this alert's on-call rotation",
    "a monitoring platform engineer who reviews alerting rules for the fleet",
    "a systems engineer writing the runbook this alert links to",
]

ASK = {
    1: ("Give the SINGLE best executable reading of this alert: the one "
        "threshold and one duration you would actually deploy."),
    3: ("Give THREE defensible executable readings of this alert: three "
        "(threshold, duration) pairs a competent engineer could argue for."),
}


def harvest_rules() -> list:
    rules = {}
    pat = str(ROOT / "data" / "rule_corpora" /
              "kube-prometheus" / "**" / "*.yaml")
    for f in glob.glob(pat, recursive=True):
        txt = open(f, errors="ignore").read()
        for m in re.finditer(r"-\s*alert:\s*(\w+)(.*?)(?=\n\s*-\s*alert:|\Z)",
                             txt, re.S):
            name, body = m.group(1), m.group(2)
            s = re.search(r"summary:\s*(?:\||>-?)?\s*(.+)", body)
            d = re.search(r"description:\s*(?:\||>-?)?\s*(.+)", body)
            e = re.search(r"expr:\s*(?:\||>-?)?\s*(.+)", body)
            if s and d and e and name not in rules:
                rules[name] = dict(name=name, summary=s.group(1).strip()[:300],
                                   description=d.group(1).strip()[:400])
    sel = sorted(rules)[:30]
    return [rules[k] for k in sel]


def prompt_for(rule: dict, persona: str, k: int) -> str:
    return (
        f"You are {persona}.\n\n"
        f"Alert name: {rule['name']}\n"
        f"Summary: {rule['summary']}\n"
        f"Description: {rule['description']}\n\n"
        f"{ASK[k]}\n\n"
        "A reading is a numeric THRESHOLD (the level at which the condition "
        "is considered met) and a DURATION IN SECONDS the condition must "
        "hold before firing (0 if it should fire immediately).\n"
        f"Reply with exactly {k} line(s), each formatted as\n"
        "  threshold=<number> duration=<seconds>\n"
        "and nothing else. No prose, no units, no explanation.")


def ask(prompt: str, timeout: float = 120.0):
    try:
        out = subprocess.run(["claude", "-p", prompt], capture_output=True,
                             text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return out.stdout


def parse(reply, k: int):
    if not reply:
        return None
    pairs = re.findall(r"threshold\s*=\s*(-?[\d.eE+]+)\s+duration\s*=\s*(-?[\d.eE+]+)",
                       reply)
    vals = []
    for t, d in pairs[:k]:
        try:
            vals.append((float(t), float(d)))
        except ValueError:
            pass
    return vals if len(vals) == k else None


def nondominated(readings) -> bool:
    """No member forbids everything the others do: no reading maximal on both."""
    if len(readings) < 2:
        return False
    for i, a in enumerate(readings):
        if all(a[0] >= b[0] and a[1] >= b[1] for j, b in enumerate(readings) if j != i):
            return False
    return True


def collect() -> None:
    rules = harvest_rules()
    json.dump(rules, open(RULES, "w"), indent=1)
    done = set()
    if RAW.exists():
        for line in open(RAW):
            r = json.loads(line)
            done.add((r["rule"], r["persona"], r["k"]))
    jobs = [(r, pi, k) for r in rules for pi in range(len(PERSONAS)) for k in (1, 3)
            if (r["name"], pi, k) not in done]
    print(f"{len(rules)} rules, {len(jobs)} calls to make")
    lock = threading.Lock()
    fh = open(RAW, "a")

    def one(job):
        r, pi, k = job
        reply = ask(prompt_for(r, PERSONAS[pi], k))
        vals = parse(reply, k)
        if vals is None:                       # registered: one retry
            reply = ask(prompt_for(r, PERSONAS[pi], k))
            vals = parse(reply, k)
        with lock:
            fh.write(json.dumps(dict(rule=r["name"], persona=pi, k=k,
                                     readings=vals)) + "\n")
            fh.flush()
        return job

    n = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(one, j) for j in jobs]):
            fut.result(); n += 1
            if n % 30 == 0:
                print(f"  {n}/{len(jobs)}")
    fh.close()
    print("collection complete")


def analyze() -> None:
    rows = [json.loads(l) for l in open(RAW)]
    by = {}
    for r in rows:
        by.setdefault((r["k"], r["rule"]), {})[r["persona"]] = r["readings"]
    rules = sorted({k[1] for k in by})
    rng = np.random.default_rng(0)
    out = {"registration": "REGISTRATION_V10 E1b", "n_rules": len(rules),
           "conditions": {}}

    for k in (1, 3):
        parsed, across, within = 0, [], []
        pool = []                       # all readings, for the permutation null
        for rule in rules:
            per = by.get((k, rule), {})
            got = [v for v in per.values() if v]
            if len(got) < 3:
                continue
            parsed += 1
            singles = [g[0] for g in got]           # first reading per persona
            across.append(nondominated(singles))
            pool.extend([x for g in got for x in g])
            if k == 3:
                within.append(sum(nondominated(g) for g in got) >= 2)
        n = max(parsed, 1)
        rec = dict(parsed=parsed, parse_rate=round(parsed / len(rules), 3),
                   across_rate=round(float(np.mean(across)), 4) if across else None)
        if k == 3:
            rec["within_rate_majority"] = round(float(np.mean(within)), 4) if within else None
        # permutation null: re-pair thresholds and durations across the pool
        if pool:
            th = np.array([p[0] for p in pool]); du = np.array([p[1] for p in pool])
            null = []
            for _ in range(10000):
                t = rng.permutation(th); d = rng.permutation(du)
                idx = rng.integers(0, len(pool), size=(len(across), 3))
                null.append(np.mean([nondominated([(t[i], d[i]) for i in row])
                                     for row in idx]))
            null = np.array(null)
            rec["null_mean"] = round(float(null.mean()), 4)
            if rec["across_rate"] is not None:
                p = 2 * min((null >= rec["across_rate"]).mean(),
                            (null <= rec["across_rate"]).mean())
                rec["null_p_two_sided"] = round(float(min(p, 1.0)), 4)
        out["conditions"][f"k={k}"] = rec
        print(f"k={k}: parsed {parsed}/{len(rules)}  across-interpreter "
              f"non-dominance {rec['across_rate']}  null {rec.get('null_mean')}  "
              f"p={rec.get('null_p_two_sided')}"
              + (f"  within-interpreter {rec.get('within_rate_majority')}" if k == 3 else ""))
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if not a.analyze:
        collect()
    analyze()
