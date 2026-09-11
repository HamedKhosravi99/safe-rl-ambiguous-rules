"""REGISTRATION_V18: the exact decision-level pipeline on monitoring rules
of eight other organizations (pinned clones of the generation study).

Same parser, candidate generator, fixtures, dominance, compiler, screens,
face certificate and ladder as the kube-prometheus analysis; nothing
retuned.  See REGISTRATION_V18.md for the repository list and exclusions.

Run: SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.third_corpus
Writes results/e2e/third_corpus.json
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
import yaml

from .class_ladder import SUBSET_CAP, _grouped, _screen
from .control_suite import _eligible, compile_instance
from .dominance import pool_dominance
from .evaluate import build_prom_pool
from .exact_nonnested import analyse as exact_analyse
from .faithfulness_provenance import verdict_of
from .parse import (PromReading, PromTarget, _strip_template, dur_to_s,
                    parse_prom_expr, prom_threshold_bank)
from .policy_class_budget import free_values_multi

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = os.path.join(ROOT, "data", "rule_corpora")
OUT = os.path.join(ROOT, "results/e2e", "third_corpus.json")
REPOS = ["mimir", "loki", "cluster-monitoring-operator", "victoriametrics", "rook", "ceph", "thanos", "tidb"]
BUDGETS = (0.005, 0.05)
OP = 0.05
TOL = 1e-7


def rule_files(repo: str) -> List[str]:
    out = []
    for f in glob.glob(os.path.join(SRC, repo, "**", "*.y*ml"), recursive=True):
        try:
            txt = open(f, errors="ignore").read()
        except OSError:
            continue
        if "alert:" in txt and "expr:" in txt:
            out.append(f)
    return sorted(out)


def parse_repo(repo: str, commit: str):
    targets: List[PromTarget] = []
    skipped = Counter()
    seen = set()
    for f in rule_files(repo):
        try:
            docs = list(yaml.safe_load_all(open(f, errors="ignore")))
        except yaml.YAMLError:
            skipped["yaml_error_file"] += 1
            continue
        rel = os.path.relpath(f, os.path.join(SRC, repo))
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            groups = (doc.get("spec") or {}).get("groups") if isinstance(doc.get("spec"), dict) else None
            if groups is None:
                groups = doc.get("groups")
            if not isinstance(groups, list):
                continue
            for grp in groups:
                if not isinstance(grp, dict):
                    continue
                gname = str(grp.get("name", ""))
                for r in grp.get("rules") or []:
                    if not isinstance(r, dict) or "alert" not in r or "expr" not in r:
                        skipped["not_alert"] += 1
                        continue
                    parsed = parse_prom_expr(str(r["expr"]))
                    if parsed is None:
                        skipped["out_of_dsl_expr"] += 1
                        continue
                    ann = r.get("annotations") or {}
                    text = _strip_template(" ".join(str(ann.get(k, "")) for k in ("summary", "description", "message")))
                    if not text:
                        skipped["no_text"] += 1
                        continue
                    sev = str((r.get("labels") or {}).get("severity", ""))
                    reading = PromReading(metric=parsed["metric"], selectors=parsed["selectors"],
                                          comparator=parsed["comparator"], threshold=parsed["threshold"],
                                          rate_window_s=parsed["rate_window_s"], aggregation=parsed["aggregation"],
                                          agg_by=parsed["agg_by"], for_s=dur_to_s(r.get("for")) or 0.0, severity=sev)
                    key = (str(r["alert"]), reading.struct_key())
                    if key in seen:
                        skipped["duplicate"] += 1
                        continue
                    seen.add(key)
                    targets.append(PromTarget(name=str(r["alert"]), group=gname, reading=reading, text=text,
                                              repo=repo, commit=commit, file=rel, sub_source=gname,
                                              last_commit_ts=None,
                                              raw_expr=re.sub(r"\s+", " ", str(r["expr"]).strip())))
    return targets, dict(skipped)


def main():
    t0 = time.perf_counter()
    man = json.load(open(os.path.join(ROOT, "results/e2e", "catalog_src_manifest.json")))["repos"]
    targets, skipped = [], {}
    for repo in REPOS:
        ts, sk = parse_repo(repo, man[repo]["sha"])
        skipped[repo] = dict(sk, parsed=len(ts))
        targets += ts
        print(f"[{repo}] parsed {len(ts)} (skipped {sk})", flush=True)
    bank = prom_threshold_bank(targets)
    rows = []
    for ti, t in enumerate(targets):
        uid = f"{t.repo}/{t.name}#{ti}"
        try:
            pool = build_prom_pool(t, bank)
        except Exception as e:  # noqa: BLE001 - counted, never hidden
            rows.append(dict(uid=uid, repo=t.repo, pool_error=repr(e)[:120]))
            continue
        vecs = [list(c.vector) for c in pool.classes]
        dom = pool_dominance(vecs)
        ok, why, readings = _eligible(pool)
        row = dict(uid=uid, repo=t.repo, alert=t.name, K=len(vecs), n_fixtures=pool.n_fixtures,
                   non_dominated=bool(dom["genuinely_non_dominated"]),
                   faithful=verdict_of(t.raw_expr, float(t.reading.for_s))[0])
        if ok:
            m = compile_instance(readings)
            comp = {}
            for d in BUDGETS:
                a = exact_analyse(m, d)
                if not a.get("feasible", True):
                    comp[str(d)] = dict(feasible=False); continue
                feas = [s for s in a["singletons"] if s.get("feasible")]
                vs = [s for s in feas if abs(s["V"] - a["V_U"]) <= TOL]
                pol = any(s["worst_worst"] is not None and s["worst_worst"] <= d + TOL for s in vs)
                comp[str(d)] = dict(feasible=True, V_U=a["V_U"], fires=bool(a["fires_value"]), value_clear=bool(vs),
                                    face_clear=bool(pol), rpoa=a["rpoa"], n_value_sufficient=len(vs))
            row.update(status="compiled", n_states=m["nS"], compiled=comp)
        else:
            provably_clear = (why.startswith("|Max|=1 ") or why in ("singleton antichain", "singleton subfamily",
                                                                  "no threshold/duration crossing"))
            row.update(status="provably_clear_nested" if provably_clear else "unanalyzed_outside_grammar", reason=why)
        # free class and ladder at the operating budget
        vals = free_values_multi(vecs, list(BUDGETS))
        free = {}
        for d in BUDGETS:
            V_full, V_k = vals[d]
            free[str(d)] = dict(V_U=V_full, fires=bool(min(V_k) - V_full > TOL),
                                rpoa=((max(V_k) - V_full) / max(V_k)) if max(V_k) > 0 else 0.0)
        row["free"] = free
        arr = np.asarray(vecs, float)
        K = arr.shape[0]
        ladder = {}
        for k in (0, 1, 2):
            if k > K:
                continue
            subsets = list(itertools.combinations(range(K), k))
            trunc = len(subsets) > SUBSET_CAP
            fires = []
            for A in subsets[:SUBSET_CAP]:
                n_g, mm = _grouped(arr, A)
                fires.append(_screen(n_g, mm, K, OP)[2])
            ladder[str(k)] = dict(some=any(fires), every=all(fires) if fires else False, n=len(fires), truncated=trunc)
        row["ladder"] = ladder
        fr = None
        for k in (0, 1, 2):
            if str(k) in ladder and ladder[str(k)]["some"]:
                fr = k; break
        if fr is None:
            fr = "free" if free[str(OP)]["fires"] else None
        row["frontier"] = fr
        rows.append(row)
        if ti % 50 == 0:
            print(f"  {ti}/{len(targets)} pools ({time.perf_counter()-t0:.0f}s)", flush=True)

    good = [r for r in rows if "pool_error" not in r]

    def agg(sel):
        comp = [r for r in sel if r["status"] == "compiled"]
        nested = [r for r in sel if r["status"] == "provably_clear_nested"]
        unan = [r for r in sel if r["status"] == "unanalyzed_outside_grammar"]
        out = dict(n_pools=len(sel), non_dominance_rate=(sum(r["non_dominated"] for r in sel) / len(sel)) if sel else None,
                   faithful=sum(1 for r in sel if r["faithful"] == "FAITHFUL"),
                   lossy=sum(1 for r in sel if r["faithful"] == "LOSSY"),
                   compiled_eligible=len(comp), nested_clear=len(nested), unanalyzed=len(unan),
                   analyzable=len(comp) + len(nested))
        for d in BUDGETS:
            dd = str(d)
            fe = [r for r in comp if r["compiled"][dd].get("feasible")]
            out[f"fires_{dd}"] = sum(1 for r in fe if r["compiled"][dd]["fires"])
            out[f"value_clear_{dd}"] = sum(1 for r in fe if r["compiled"][dd]["value_clear"])
            out[f"face_clear_{dd}"] = sum(1 for r in fe if r["compiled"][dd]["face_clear"])
            out[f"free_fires_{dd}"] = sum(1 for r in sel if r["free"][dd]["fires"])
            out[f"infeasible_{dd}"] = len(comp) - len(fe)
        fa = [r for r in sel if r["faithful"] == "FAITHFUL" and r["status"] in ("compiled", "provably_clear_nested")]
        out["analyzable_faithful"] = len(fa)
        out["fires_0.05_faithful"] = sum(1 for r in fa if r["status"] == "compiled" and r["compiled"]["0.05"].get("fires"))
        lad = {}
        for k in ("0", "1", "2"):
            rs = [r for r in sel if k in r["ladder"]]
            lad[k] = dict(n=len(rs), some=sum(r["ladder"][k]["some"] for r in rs), every=sum(r["ladder"][k]["every"] for r in rs))
        out["ladder"] = lad
        out["frontier"] = dict(Counter(str(r["frontier"]) for r in sel))
        return out

    res = dict(registration="REGISTRATION_V18.md", repos=REPOS, commits={r: man[r]["sha"] for r in REPOS},
               parse=skipped, n_pool_errors=len(rows) - len(good), budgets=list(BUDGETS), operating=OP,
               pooled=agg(good), per_repo={r: agg([x for x in good if x["repo"] == r]) for r in REPOS},
               rows=rows, seconds=round(time.perf_counter() - t0, 1))
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(dict(parse=skipped, pooled=res["pooled"]), indent=1))


if __name__ == "__main__":
    main()
