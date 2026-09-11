"""REGISTRATION_V23: reward-free decide on the ARTEMIS retained sets.

For each unit and pool, reconstructs the calibrated retained set of the
frozen ARROW arm (choice-frequency score, archived q-hat), decides every
ordered entailment pair with the exact LTL checker of V17, and reports
w (the entailment antichain width), joint satisfiability of the strongest
readings, and whether the strongest reading is expert-plausible.

Run: PYTHONPATH=. python3 corset_e2e/external/artemis_decide.py [P2 P1]
Writes results/e2e/artemis_decide.json
"""
from __future__ import annotations

import json
import os
import signal
import statistics
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.external.ltl_equiv import satisfiable  # noqa: E402

R = os.path.join(ROOT, "results/e2e")
OUT = os.path.join(R, "artemis_decide.json")
PAIR_SECONDS = 30
CONJ_SECONDS = 60
BINS = [("1", 1, 1), ("2-5", 2, 5), ("6-10", 6, 10), (">10", 11, 10 ** 9)]


class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)


def guarded(fn, seconds):
    signal.alarm(seconds)
    try:
        return fn()
    except (_Timeout, RecursionError, MemoryError):
        return None
    finally:
        signal.alarm(0)


def entails(a, b):
    """True: a |= b; False: not; None: undecided."""
    s = guarded(lambda: satisfiable(f"({a}) & !({b})"), PAIR_SECONDS)
    return None if s is None else (s is False)


def bin_of(n):
    for name, lo, hi in BINS:
        if lo <= n <= hi:
            return name
    return ">10"


def retained_set(u, rec, which):
    tr = [t for t in u["trials"] if t["valid"] and (t["in_p1"] if which == "P1" else True)]
    freq = Counter(t["cls"] for t in tr)
    n = len(tr)
    U = sorted(c for c in freq if freq[c] / n >= rec["qhat"] - 1e-12)
    return U


def analyse_unit(u, rec, which):
    t0 = time.perf_counter()
    U = retained_set(u, rec, which)
    assert len(U) == rec["arms"]["arrow"]["size"], (u["uid"], which, len(U), rec["arms"]["arrow"]["size"])
    reps = [u["cand_reps"][c] for c in U]
    plaus = [bool(u["cand_match"].get(str(c))) for c in U]
    m = len(reps)
    sat = [guarded(lambda r=r: satisfiable(r), PAIR_SECONDS) for r in reps]
    idx = [i for i in range(m) if sat[i] is not False]
    n_unsat = m - len(idx)
    n_sat_undecided = sum(1 for i in idx if sat[i] is None)
    E = {}
    for i in idx:
        for j in idx:
            if i != j:
                E[(i, j)] = entails(reps[i], reps[j])
    undecided_pairs = sum(1 for v in E.values() if v is None)
    mutual = sum(1 for i in idx for j in idx if i < j and E[(i, j)] is True and E[(j, i)] is True)
    minimal = [i for i in idx if not any(E[(j, i)] is True and E[(i, j)] is not True for j in idx if j != i)]
    w = len(minimal)
    strongest = w == 1 and sat[minimal[0]] is True
    if strongest:
        joint = True
    elif w == 0:
        joint = False
    else:
        conj = "(" + ") & (".join(reps[i] for i in minimal) + ")"
        joint = guarded(lambda: satisfiable(conj), CONJ_SECONDS)
    # the readings a set-protecting controller must satisfy at once: the w strongest;
    # every other retained reading is entailed by one of them (given decided pairs)
    covered = all(any(E.get((j, i)) is True for j in minimal) for i in idx if i not in minimal)
    return dict(uid=u["uid"], group=rec["group"], n_plausible=rec["n_plausible"], bin=bin_of(rec["n_plausible"]),
                n_retained=m, n_unsat=n_unsat, n_sat_undecided=n_sat_undecided, undecided_pairs=undecided_pairs,
                n_pairs=len(E), mutual_entailment_pairs=mutual, w=w, strongest=bool(strongest),
                strongest_plausible=(plaus[minimal[0]] if strongest else None),
                n_minimal_plausible=sum(plaus[i] for i in minimal), any_plausible_retained=any(plaus),
                joint_satisfiable=joint, minimal_cover_decided=bool(covered),
                seconds=round(time.perf_counter() - t0, 2))


def aggregate(rows):
    n = len(rows)
    ws = [r["w"] for r in rows]
    js = Counter("undecided" if r["joint_satisfiable"] is None else ("yes" if r["joint_satisfiable"] else "no") for r in rows)
    w1 = [r for r in rows if r["strongest"]]
    out = dict(n_units=n, n_w1=len(w1), frac_w1=len(w1) / n,
               frac_w_le2=sum(w <= 2 for w in ws) / n, frac_w_le4=sum(w <= 4 for w in ws) / n,
               w_median=statistics.median(ws), w_max=max(ws), w_mean=statistics.mean(ws),
               retained_median=statistics.median(r["n_retained"] for r in rows),
               w_over_retained_median=statistics.median(r["w"] / r["n_retained"] for r in rows),
               joint=dict(js), frac_joint_sat=js["yes"] / n, frac_joint_unsat=js["no"] / n,
               strongest_plausible=sum(1 for r in w1 if r["strongest_plausible"]),
               frac_strongest_plausible=(sum(1 for r in w1 if r["strongest_plausible"]) / len(w1) if w1 else None),
               units_with_unsat_reading=sum(1 for r in rows if r["n_unsat"] > 0),
               units_with_undecided_pairs=sum(1 for r in rows if r["undecided_pairs"] > 0),
               undecided_pairs_total=sum(r["undecided_pairs"] for r in rows),
               pairs_total=sum(r["n_pairs"] for r in rows),
               mutual_entailment_pairs=sum(r["mutual_entailment_pairs"] for r in rows),
               cover_decided_all=all(r["minimal_cover_decided"] for r in rows))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pools", nargs="*", default=["P2", "P1"])
    ap.add_argument("--chunk", default=None, help="k/n: process units with index %% n == k, write a chunk file")
    ap.add_argument("--merge-chunks", type=int, default=0, help="n: merge n chunk files of the given pool into the archive")
    a = ap.parse_args()
    units = {u["uid"]: u for u in json.load(open(os.path.join(R, "artemis_units.json")))["units"]}
    ext = json.load(open(os.path.join(R, "artemis_external.json")))
    res = dict(registration="REGISTRATION_V23.md", artifact_commit=ext["artifact_commit"], delta_sem=ext["delta_sem"],
               pair_seconds=PAIR_SECONDS, conj_seconds=CONJ_SECONDS, pools={})
    if os.path.exists(OUT):
        try:
            res["pools"] = json.load(open(OUT)).get("pools", {})
        except Exception:
            pass

    def finish(which, rows, seconds):
        by_group = {g: aggregate([r for r in rows if r["group"] == g]) for g in sorted({r["group"] for r in rows})}
        by_bin = {b: aggregate([r for r in rows if r["bin"] == b]) for b, _, _ in BINS if any(r["bin"] == b for r in rows)}
        res["pools"][which] = dict(overall=aggregate(rows), by_group=by_group, by_bin=by_bin, per_unit=rows, seconds=round(seconds, 1))
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"[{which}] done in {seconds:.1f} s: {json.dumps(res['pools'][which]['overall'])}", flush=True)

    for which in a.pools:
        recs = ext["pools"][which]["per_unit"]
        if a.merge_chunks:
            rows = []
            for k in range(a.merge_chunks):
                rows += json.load(open(os.path.join(R, f"artemis_decide_{which}_chunk{k}.json")))["rows"]
            order = {r["uid"]: i for i, r in enumerate(recs)}
            rows.sort(key=lambda r: order[r["uid"]])
            assert [r["uid"] for r in rows] == [r["uid"] for r in recs], "chunks do not cover the pool exactly"
            finish(which, rows, sum(r["seconds"] for r in rows))
            continue
        t0 = time.perf_counter()
        rows = []
        if a.chunk:
            k, n = (int(x) for x in a.chunk.split("/"))
            recs = [r for i, r in enumerate(recs) if i % n == k]
        for i, rec in enumerate(recs):
            row = analyse_unit(units[rec["uid"]], rec, which)
            rows.append(row)
            if i % 10 == 0 or row["seconds"] > 20:
                print(f"[{which}] {i+1}/{len(recs)} {row['uid']} |U|={row['n_retained']} w={row['w']} "
                      f"joint={row['joint_satisfiable']} undecided={row['undecided_pairs']} {row['seconds']}s", flush=True)
        if a.chunk:
            k, n = (int(x) for x in a.chunk.split("/"))
            json.dump(dict(pool=which, chunk=k, n_chunks=n, rows=rows), open(os.path.join(R, f"artemis_decide_{which}_chunk{k}.json"), "w"), indent=1)
            print(f"[{which}] chunk {k}/{n} done: {len(rows)} units in {time.perf_counter() - t0:.1f} s", flush=True)
            continue
        finish(which, rows, time.perf_counter() - t0)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
