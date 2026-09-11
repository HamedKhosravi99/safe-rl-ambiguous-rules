"""V51 step 1: the V6-protocol LLM metric licenser, re-run identically on the
calibration and test grammar-reachable units (batched per repository,
target-blind visible fields + the repository's v5 documentation namespace,
<= 16 names per unit). Resumable; stops on the CLI's usage-limit signal.

Run: PYTHONPATH=. python3 corset_e2e/analysis/llm_license_v51.py
Writes results/e2e/v51_selections/selections.jsonl (one json per unit) and run_log.json
"""
from __future__ import annotations

import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.calibration.run_e2e_v4 import load_split, hid, vis  # noqa: E402
from corset_e2e.dsl.schema import IN_DSL, classify_target  # noqa: E402

R = os.path.join(ROOT, "results/e2e")
SEL_DIR = os.path.join(R, "v51_selections")
OUT = os.path.join(SEL_DIR, "selections.jsonl")
LOG = os.path.join(SEL_DIR, "run_log.json")
BATCH = 12
WORKERS = 3
KMAX = 16
SEED = 20260906
TEXT_CAP = 600
_LIMIT = re.compile(r"session limit|usage limit|rate limit|hit your .* limit|resets \d|API Error|overloaded", re.I)

PROMPT_HEAD = (
    "You are the metric-licensing stage of an alerting-rule interpreter. For each RULE below you see only "
    "target-blind fields: alert name, rule group, severity, source file and the human-written description. "
    "From the CATALOG of metric names of this repository, select up to 16 names most likely to appear in the "
    "rule's PromQL expression, the primary metric first. Use only names copied exactly from the CATALOG; never "
    "invent or modify a name. Output exactly one line per rule, in the order given, in the form\n"
    "rule_id|name1,name2,...\nwith no other text, no numbering and no commentary.\n\n")


def catalog(repo):
    p = os.path.join(R, "catalog_v5", f"{repo}.tokens")
    names = sorted({l.strip() for l in open(p) if l.strip()})
    return names


def unit_block(e):
    v = vis(e["rule_id"])
    text = " ".join((v.get("text") or "").split())[:TEXT_CAP]
    return (f"rule_id: {e['rule_id']}\nalert_name: {v.get('alert_name', '')}\ngroup: {v.get('group', '')}\n"
            f"severity: {v.get('severity', '')}\nfile: {v.get('file', '')}\ntext: {text}\n")


def ask(prompt, timeout=300.0):
    """One CLI call in its own process group; a hard kill of the whole group on timeout
    (subprocess.run's timeout leaves the CLI's children holding the pipe, which hung two
    calls for six hours in the first launch)."""
    try:
        p = subprocess.Popen(["claude", "-p", prompt], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    except FileNotFoundError:
        return None
    try:
        out, _err = p.communicate(timeout=timeout)
        return out
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            p.communicate(timeout=10)
        except Exception:
            pass
        return None


def parse(reply, rids, cat):
    got = {}
    if not reply:
        return got
    for line in reply.splitlines():
        line = line.strip().strip("`")
        if "|" not in line:
            continue
        rid, ms = line.split("|", 1); rid = rid.strip()
        if rid not in rids:
            continue
        names = []
        for m in ms.split(","):
            m = m.strip().strip("`").strip()
            if m and m in cat and m not in names:
                names.append(m)
        got[rid] = names[:KMAX]
    return got


def main():
    done = {}
    if os.path.exists(OUT):
        for line in open(OUT):
            d = json.loads(line); done[d["rid"]] = d
    units = []
    for split in ("cal", "test"):
        for e in load_split(split):
            if classify_target(hid(e["rule_id"])) == IN_DSL:
                units.append(dict(e, split=split))
    by_repo = {}
    for u in units:
        by_repo.setdefault(u["repo"], []).append(u)
    rng = random.Random(SEED)
    batches = []
    for repo in sorted(by_repo):
        us = sorted(by_repo[repo], key=lambda u: u["rule_id"]); rng.shuffle(us)
        for i in range(0, len(us), BATCH):
            batches.append((repo, us[i:i + BATCH]))
    todo = [(r, [u for u in b if u["rule_id"] not in done]) for r, b in batches]
    todo = [(r, b) for r, b in todo if b]
    print(f"units {len(units)} (cal {sum(u['split']=='cal' for u in units)}, test {sum(u['split']=='test' for u in units)}); batches {len(batches)}, to do {len(todo)}; already done {len(done)}", flush=True)
    cats = {repo: catalog(repo) for repo in by_repo}
    log = dict(cli=subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=30).stdout.strip(), batch=BATCH, workers=WORKERS,
               seed=SEED, started=time.strftime("%Y-%m-%d %H:%M:%S"), calls=[], limit_stop=False, dropped_names=0)
    stop = {"flag": False}

    def run_batch(repo, b, tag):
        cat = cats[repo]; catset = set(cat); rids = {u["rule_id"] for u in b}
        prompt = PROMPT_HEAD + f"CATALOG ({len(cat)} names of repository {repo}):\n" + "\n".join(cat) + "\n\nRULES:\n\n" + "\n".join(unit_block(u) for u in b)
        t0 = time.time(); reply = ask(prompt); timeouts = 0
        if reply is None:                      # timed out: one immediate retry of the same batch
            timeouts = 1; reply = ask(prompt)
            if reply is None:
                timeouts = 2
        dt = time.time() - t0
        limit = bool(reply is not None and _LIMIT.search(reply))
        got = parse(reply, rids, catset) if not limit else {}
        raw_counts = 0
        if reply and not limit:
            for line in reply.splitlines():
                if "|" in line:
                    rid, ms = line.split("|", 1)
                    if rid.strip() in rids:
                        raw_counts += len([m for m in ms.split(",") if m.strip()])
        return dict(repo=repo, tag=tag, n=len(b), rids=sorted(rids), got=got, seconds=round(dt, 1), reply_chars=(len(reply) if reply else 0),
                    limit=limit, raw_names=raw_counts, prompt_chars=len(prompt), timeouts=timeouts)

    def commit(res, retry_pool):
        kept = sum(len(v) for v in res["got"].values())
        log["dropped_names"] += max(0, res["raw_names"] - kept)
        log["calls"].append({k: res[k] for k in ("repo", "tag", "n", "seconds", "reply_chars", "limit", "prompt_chars", "timeouts")} | dict(answered=len(res["got"])))
        if res["limit"]:
            stop["flag"] = True; log["limit_stop"] = True; print(f"  USAGE LIMIT signalled on {res['repo']} {res['tag']}; stopping submission", flush=True)
            return
        with open(OUT, "a") as fh:
            for rid in res["rids"]:
                if rid in res["got"]:
                    d = dict(rid=rid, repo=res["repo"], names=res["got"][rid], tag=res["tag"]); done[rid] = d; fh.write(json.dumps(d) + "\n")
                else:
                    retry_pool.append(rid)
        print(f"  {res['repo']:30s} {res['tag']:8s} n={res['n']:2d} answered={len(res['got']):2d} {res['seconds']:6.1f}s prompt {res['prompt_chars']//1000}k chars timeouts {res['timeouts']}", flush=True)

    retry = []
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {}
        for i, (repo, b) in enumerate(todo):
            if stop["flag"]:
                break
            futs[ex.submit(run_batch, repo, b, f"b{i}")] = (repo, b)
            if len(futs) >= WORKERS:
                for f in as_completed(list(futs)):
                    commit(f.result(), retry); del futs[f]; break
        for f in as_completed(list(futs)):
            commit(f.result(), retry)
    # one retry pass for units missing from their reply
    if retry and not stop["flag"]:
        umap = {u["rule_id"]: u for u in units}; by = {}
        for rid in retry:
            by.setdefault(umap[rid]["repo"], []).append(umap[rid])
        rb = [(repo, us[i:i + BATCH]) for repo, us in by.items() for i in range(0, len(us), BATCH)]
        print(f"retry pass: {len(retry)} units in {len(rb)} batches", flush=True)
        leftover = []
        with ThreadPoolExecutor(WORKERS) as ex:
            for f in as_completed([ex.submit(run_batch, repo, b, f"r{i}") for i, (repo, b) in enumerate(rb)]):
                commit(f.result(), leftover)
        with open(OUT, "a") as fh:
            for rid in leftover:
                d = dict(rid=rid, repo=umap[rid]["repo"], names=[], tag="empty"); done[rid] = d; fh.write(json.dumps(d) + "\n")
        log["empty_after_retry"] = len(leftover)
    log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); log["n_done"] = len(done); log["n_units"] = len(units)
    json.dump(log, open(LOG, "w"), indent=1)
    print(f"done {len(done)}/{len(units)} units; limit_stop={log['limit_stop']}; dropped out-of-catalog names {log['dropped_names']}", flush=True)


if __name__ == "__main__":
    main()
