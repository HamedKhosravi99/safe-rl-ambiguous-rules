"""ARTEMIS external validation, stage 1: units, expert sets, candidate pools,
semantic classes, matches, and the checker validation (REGISTRATION_V17).

Reads the ARTEMIS artifact (env ARTEMIS_DIR, a clone of
github.com/dmmendo/ARTEMIS) and writes results/e2e/artemis_units.json.
No retained set or coverage number is computed here; artemis_arrow.py
does that from the JSON, so the expensive semantic work is done once.

Run: PYTHONPATH=. python3 corset_e2e/external/artemis_load.py [--datasets FSM-AP,FSM-S,REG] [--workers 8]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))
from corset_e2e.external import ltl_equiv as L  # noqa: E402
import signal  # noqa: E402

PAIR_SECONDS = 30          # a single equivalence query may not run longer than this
UNIT_SECONDS = 1200        # nor a unit's whole semantic work
_TIMEOUTS = {"n": 0}


class _PairTimeout(Exception):
    pass


def _alarm(*_a):
    raise _PairTimeout()


def equiv(a: str, b: str):
    """L.equivalent with a wall-clock guard; a timed-out pair is undecided (None)."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(PAIR_SECONDS)
    try:
        return L.equivalent(a, b)
    except (_PairTimeout, RecursionError, MemoryError):
        _TIMEOUTS["n"] += 1
        return None
    finally:
        signal.alarm(0)

ART = os.environ.get(
    "ARTEMIS_DIR",
    os.path.join(ROOT, "external_data", "artemis"))
OUT = os.path.join(ROOT, "results/e2e", "artemis_units.json")

GROUP = {"Ventilator": "Ventilator", "RobotExplain": "Robotics",
         "FSM-AP": "LMCPS", "FSM-S": "LMCPS", "REG": "LMCPS"}
# (model, method, trials) -- registered pools
P1 = [("gemini-2.5-flash", "nl2structnl-reflect", 50)]
P2 = P1 + [(m, meth, 10) for m in ("gemini-2.5-flash", "gpt-4.1")
           for meth in ("nl2structnl", "nl2ltl", "nl2ltltemplate", "nl2spec", "NL2TL", "synthtl")] \
    + [("gemini-2.5-flash", meth, 10) for meth in ("NL2TL-FT", "deepstl")]
OPT_COLS = ["timing options", "timing bool exps", "scope 1 options",
            "scope 1 bool exps", "scope 2 options", "scope 2 bool exps"]
KEYS = ["decision1", "bool_exp1", "decision2", "bool_exp2", "decision3",
        "bool_exp3", "bool_exp4", "N_DURATION"]
MAX_N_DURATION = 1          # the artifact's own accuracy convention (compute_accuracy_metrics.ipynb)

_TEMPLATES = json.load(open(os.path.join(ART, "metadata", "fretish_structnl_to_ltl_dict.json")))


# ---------------------------------------------------------------------------
# expert sets (port of data_loader.get_all_outputs_for_df_row + get_ltl_from_output)
# ---------------------------------------------------------------------------

def _opt(cell):
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return [{}]
    return json.loads(cell)


def expand_expert_outputs(row, max_n: int = MAX_N_DURATION):
    data = [_opt(row[c]) for c in OPT_COLS]
    num_groups = len(data[0])
    outputs = []
    for g in range(num_groups):
        per_type = []
        for opts in data:
            grp = opts[g] if g < len(opts) else {}
            keys = list(grp.keys())
            lists = [grp[k] if isinstance(grp[k], list) else [grp[k]] for k in keys]
            per_type.append([dict(zip(keys, combo)) for combo in itertools.product(*lists)])
        for combo in itertools.product(*per_type):
            out = {}
            for part in combo:
                out.update(part)
            outputs.append(out)
    res = []
    for out in outputs:
        d3 = out.get("decision3") or ""
        if "N_DURATION" in d3 and out.get("N_DURATION") is None:
            for n in range(1, max_n + 1):
                o = dict(out); o["N_DURATION"] = n; res.append(o)
        else:
            res.append(out)
    for out in res:
        for k in KEYS:
            out.setdefault(k, None)
    return res


def ltl_from_output(out) -> str:
    d1, d2, d3 = out["decision1"], out["decision2"], out["decision3"]
    key = d1.replace("_ABSTRACT_VAR1_", d2.replace("_ABSTRACT_VAR2_", d3))
    t = _TEMPLATES[key]
    for k in ("bool_exp1", "bool_exp2", "bool_exp3", "bool_exp4"):
        v = out.get(k)
        if v is not None and k in t:
            t = t.replace(k, f"({v})")
    n = out.get("N_DURATION")
    if n is not None and "N_DURATION" in t:
        n = int(n)
        t = t.replace("N_DURATION+1", str(n + 1)).replace("N_DURATION-1", str(n - 1)).replace("N_DURATION", str(n))
    return t


def load_units(datasets):
    units = []
    for ds in datasets:
        df = pd.read_excel(os.path.join(ART, "benchmarks", "fret_specs", ds, "PlausibleSpecs.xlsx"), engine="openpyxl")
        for i in range(len(df)):
            row = df.iloc[i]
            outs = expand_expert_outputs(row)
            labels = [ltl_from_output(o) for o in outs]
            units.append(dict(uid=f"{ds}-{i}", dataset=ds, group=GROUP[ds], idx=i,
                              nl=str(row["NL"]), labels=labels))
    return units


# ---------------------------------------------------------------------------
# candidate pools
# ---------------------------------------------------------------------------

def result_path(ds, i, model, method, trials):
    return os.path.join(ART, "fretish_results", f"{ds}-{i}_model-{model}_trials-{trials}_{method}.json")


def load_trials(ds, i):
    """[(source, trial_idx, ltl_or_None, valid)] in registered file order."""
    trials = []
    missing = []
    for model, method, ntr in P2:
        p = result_path(ds, i, model, method, ntr)
        if not os.path.exists(p):
            missing.append(f"{model}/{method}/{ntr}")
            continue
        d = json.load(open(p))
        src = f"{model}/{method}"
        for j, o in enumerate(d):
            s = o.get("output_LTL") if isinstance(o, dict) else None
            if not isinstance(s, str) and isinstance(o, dict) and "decision1" in o:
                try:
                    s = ltl_from_output(o)
                except Exception:      # noqa: BLE001
                    s = None
            ok = isinstance(s, str) and "N_DURATION" not in s and L.is_valid_ltl(s)
            trials.append(dict(src=src, trial=j, ltl=s if ok else None, valid=bool(ok), in_p1=(model, method, ntr) in P1))
    return trials, missing


# ---------------------------------------------------------------------------
# per-unit semantic work
# ---------------------------------------------------------------------------

def _classes(formulas):
    """Merge a list of LTL strings into semantic classes; returns (class_of[i], reps, undecided_pairs)."""
    # syntactic pre-merge on the core normal form
    core_key = {}
    reps = []
    class_of = []
    undecided = 0
    for s in formulas:
        try:
            k = repr(L._core(s))
        except Exception:      # noqa: BLE001
            k = "PARSEFAIL::" + s
        if k in core_key:
            class_of.append(core_key[k]); continue
        # semantic merge against existing reps
        found = None
        for ci, rep in enumerate(reps):
            v = equiv(s, rep)
            if v is None:
                undecided += 1
                continue
            if v:
                found = ci; break
        if found is None:
            reps.append(s); found = len(reps) - 1
        core_key[k] = found
        class_of.append(found)
    return class_of, reps, undecided


def process_unit(u):
    t0 = time.perf_counter()
    _TIMEOUTS["n"] = 0
    trials, missing = load_trials(u["dataset"], u["idx"])
    # expert set: semantic classes of the labels
    lab_cls, lab_reps, lab_und = _classes(u["labels"])
    # candidate classes over the whole P2 pool (P1 is a subset, classes shared)
    valid = [t for t in trials if t["valid"]]
    cand_cls, cand_reps, cand_und = _classes([t["ltl"] for t in valid])
    for t, c in zip(valid, cand_cls):
        t["cls"] = c
    # matches: class rep vs each label class rep
    match = {}
    und_match = 0
    for ci, rep in enumerate(cand_reps):
        hits = []
        for li, lrep in enumerate(lab_reps):
            v = equiv(rep, lrep)
            if v is None:
                und_match += 1
            elif v:
                hits.append(li)
        match[ci] = hits
    return dict(uid=u["uid"], dataset=u["dataset"], group=u["group"], idx=u["idx"], nl=u["nl"],
                n_plausible_enumerated=len(u["labels"]), label_classes=len(lab_reps),
                label_class_of=lab_cls, label_reps=lab_reps,
                trials=[{k: v for k, v in t.items() if k != "ltl"} for t in trials],
                n_trials=len(trials), n_valid=len(valid), missing_files=missing,
                cand_reps=cand_reps, cand_match=match,
                undecided=dict(labels=lab_und, candidates=cand_und, matches=und_match, pair_timeouts=_TIMEOUTS["n"]),
                seconds=round(time.perf_counter() - t0, 2))


# ---------------------------------------------------------------------------
# checker validation against the archived Spot verdicts
# ---------------------------------------------------------------------------

def validate_unit(res):
    """Per registered file: my any-plausible verdict per valid trial vs the archived is_equal column."""
    out = []
    by_src = defaultdict(list)
    for t in res["trials"]:
        by_src[t["src"]].append(t)
    for model, method, ntr in P2:
        src = f"{model}/{method}"
        mp = result_path(res["dataset"], res["idx"], model, method, ntr)[:-5] + "_metrics.json"
        if not os.path.exists(mp) or src not in by_src:
            continue
        arch = json.load(open(mp))
        mine = [bool(res["cand_match"][str(t["cls"])] if isinstance(next(iter(res["cand_match"])), str)
                     else res["cand_match"][t["cls"]]) for t in by_src[src] if t["valid"]]
        if len(mine) != len(arch):
            out.append(dict(src=src, aligned=False, n_mine=len(mine), n_arch=len(arch)))
            continue
        agree = sum(1 for a, b in zip(mine, arch) if a == bool(b[0]))
        out.append(dict(src=src, aligned=True, n=len(arch), agree=agree,
                        mine_true=sum(mine), arch_true=sum(bool(b[0]) for b in arch)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Ventilator,RobotExplain,FSM-AP,FSM-S,REG")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    datasets = a.datasets.split(",")
    units = load_units(datasets)
    print(f"{len(units)} units; |P| distribution:",
          sorted(Counter(len(u["labels"]) for u in units).items())[:20], flush=True)
    commit = subprocess.run(["git", "-C", ART, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    results = []
    t0 = time.perf_counter()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")           # fresh interpreters: a crashed worker cannot deadlock the pool
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
        futs = {ex.submit(process_unit, u): u["uid"] for u in units}
        for k, f in enumerate(as_completed(futs)):
            try:
                r = f.result()
            except Exception as e:          # noqa: BLE001 - recorded, never hidden
                uid = futs[f]
                print(f"UNIT FAILED {uid}: {e!r}", flush=True)
                uu = next(x for x in units if x["uid"] == uid)
                r = dict(uid=uid, dataset=uu["dataset"], group=uu["group"], idx=uu["idx"], nl=uu["nl"],
                         n_plausible_enumerated=len(uu["labels"]), label_classes=0, label_class_of=[], label_reps=[],
                         trials=[], n_trials=0, n_valid=0, missing_files=[], cand_reps=[], cand_match={},
                         undecided=dict(labels=0, candidates=0, matches=0, pair_timeouts=0), seconds=0.0, failed=repr(e))
            results.append(r)
            if k % 10 == 0 or r["seconds"] > 30:
                print(f"[{k+1}/{len(units)}] {r['uid']} |P|={r['n_plausible_enumerated']}->{r['label_classes']} classes; "
                      f"valid {r['n_valid']}/{r['n_trials']} -> {len(r['cand_reps'])} classes; "
                      f"undecided {r['undecided']}; {r['seconds']}s (elapsed {time.perf_counter()-t0:.0f}s)", flush=True)
    results.sort(key=lambda r: (r["dataset"], r["idx"]))
    # validation
    val = []
    for r in results:
        for v in validate_unit(r):
            v["uid"] = r["uid"]; val.append(v)
    aligned = [v for v in val if v["aligned"]]
    n_pairs = sum(v["n"] for v in aligned)
    n_agree = sum(v["agree"] for v in aligned)
    summary = dict(n_files=len(val), n_aligned_files=len(aligned), n_trials_compared=n_pairs,
                   n_agree=n_agree, agreement=(n_agree / n_pairs if n_pairs else None),
                   unaligned_files=[v for v in val if not v["aligned"]][:50],
                   disagreeing_files=[v for v in aligned if v["agree"] != v["n"]][:200])
    print("VALIDATION:", json.dumps({k: v for k, v in summary.items() if not k.endswith("files")}))
    payload = dict(registration="REGISTRATION_V17.md; artifact github.com/dmmendo/ARTEMIS @ " + commit,
                   artifact_commit=commit, pools=dict(P1=P1, P2=P2), max_n_duration=MAX_N_DURATION,
                   datasets=datasets, checker_validation=summary, units=results)
    json.dump(payload, open(a.out, "w"))
    print("wrote", a.out, f"({os.path.getsize(a.out)/1e6:.1f} MB) in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
