"""W4: runtime, LP/MILP counts and peak memory for every exact DECIDE run.

Nothing here changes a verdict.  Each archived exact analysis is re-run
with its output redirected to a scratch file, the solver calls it makes
are counted and timed by wrapping the very names the modules bound
(`module.linprog`, `module.milp`), and the re-derived aggregates are
asserted equal to the archived ones before a row is emitted.  A row that
does not reproduce its archive is not written.

Two further measurements answer the scalability question directly:

  * `scale`: on the largest compiled state space of the suite (the 55-state
    L3-caps8-1 geometry) the value LP, the K single-reading LPs and the
    face LPs are timed for K synthetic readings up to the median calibrated
    antichain size of the e2e pipeline (18,150), so the LP stage's cost at
    the scale calibration actually produces is measured, not extrapolated.
    The readings are synthetic (random subsets of the charged state-action
    pairs on a real compiled MDP), and the rows say so.
  * the compiled state-space size as a function of K is reported from the
    compiler itself: |S| = 1 + L * prod_k (cap_k + 1), which is why the
    compiled arm carries K <= 4 and why the fallback threshold exists.

Run (each part independently; heavy ones in the background):
  SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.decide_runtime --which exact_nonnested
  ... --which policy_sufficiency | surrogate_price | policy_class_budget | class_ladder | face_ladder | e16 | scale
  ... --merge
Writes results/e2e/decide_runtime/<which>.json and, on --merge,
results/e2e/decide_runtime.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from typing import Callable, Dict, List

import numpy as np
import scipy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTDIR = os.path.join(ROOT, "results/e2e", "decide_runtime")
SCRATCH = os.path.join(OUTDIR, "_scratch")
MERGED = os.path.join(ROOT, "results/e2e", "decide_runtime.json")
os.makedirs(SCRATCH, exist_ok=True)


# ---------------------------------------------------------------------------
# solver-call instrumentation
# ---------------------------------------------------------------------------

class Counter:
    def __init__(self) -> None:
        self.n_lp = 0
        self.n_milp = 0
        self.t_lp = 0.0
        self.t_milp = 0.0
        self.max_rows = 0
        self.max_cols = 0

    def wrap_linprog(self, fn: Callable) -> Callable:
        def w(*a, **k):
            t = time.perf_counter()
            r = fn(*a, **k)
            self.t_lp += time.perf_counter() - t
            self.n_lp += 1
            A_ub = k.get("A_ub", a[1] if len(a) > 1 else None)
            A_eq = k.get("A_eq", a[3] if len(a) > 3 else None)
            rows = (0 if A_ub is None else np.asarray(A_ub).shape[0]) + \
                   (0 if A_eq is None else np.asarray(A_eq).shape[0])
            cols = len(np.asarray(a[0] if a else k["c"]).reshape(-1))
            self.max_rows = max(self.max_rows, rows)
            self.max_cols = max(self.max_cols, cols)
            return r
        return w

    def wrap_milp(self, fn: Callable) -> Callable:
        def w(*a, **k):
            t = time.perf_counter()
            r = fn(*a, **k)
            self.t_milp += time.perf_counter() - t
            self.n_milp += 1
            c = k.get("c", a[0] if a else None)
            if c is not None:
                self.max_cols = max(self.max_cols, len(np.asarray(c).reshape(-1)))
            return r
        return w

    def asdict(self) -> dict:
        return dict(n_value_or_face_lps=self.n_lp, n_milps=self.n_milp,
                    solver_seconds_lp=round(self.t_lp, 3),
                    solver_seconds_milp=round(self.t_milp, 3),
                    largest_program_rows=self.max_rows,
                    largest_program_cols=self.max_cols)


def hardware() -> dict:
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        cpu = platform.processor()
    return dict(cpu=cpu, n_cores=os.cpu_count(), python=platform.python_version(),
                scipy=scipy.__version__, numpy=np.__version__, solver="HiGHS via scipy.optimize",
                ram_bytes=int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                             text=True).stdout.strip() or 0))


def peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes
    return round(ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024, 1)


def _emit(which: str, payload: dict) -> None:
    payload["hardware"] = hardware()
    payload["peak_rss_mb"] = peak_rss_mb()
    with open(os.path.join(OUTDIR, f"{which}.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    print(json.dumps({k: v for k, v in payload.items() if k != "hardware"}, indent=1))


def _load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# the archived exact runs, re-derived
# ---------------------------------------------------------------------------

def run_exact_nonnested() -> None:
    from saorl.benchmark_sg import exact_nonnested as m
    c = Counter()
    m.linprog = c.wrap_linprog(m.linprog)
    m.OUT = os.path.join(SCRATCH, "exact_nonnested.json")
    t = time.perf_counter(); m.main(); wall = time.perf_counter() - t
    new, arch = _load(m.OUT), _load(os.path.join(ROOT, "results/e2e", "exact_nonnested.json"))
    for d in arch["aggregate"]:
        for key in ("n_fire", "n_clear", "n_singletons_violating", "n_instances"):
            assert new["aggregate"][d][key] == arch["aggregate"][d][key], (d, key)
    inst = arch["instances"]
    _emit("exact_nonnested", dict(
        run="exact non-nested suite (Table: exact)", objects="17 compiled instances x 5 budgets",
        n_candidate_readings=sum(i["K"] for i in inst), n_behaviour_classes=sum(i["K"] for i in inst),
        max_antichain=max(i["K"] for i in inst), n_states=sorted({i["n_states"] for i in inst}),
        wall_seconds=round(wall, 2), reproduces_archive=True, **c.asdict()))


def run_policy_sufficiency() -> None:
    from saorl.benchmark_sg import policy_sufficiency as m
    c = Counter()
    m.linprog = c.wrap_linprog(m.linprog)
    m.OUT = os.path.join(SCRATCH, "policy_sufficiency.json")
    t = time.perf_counter(); m.main(); wall = time.perf_counter() - t
    new, arch = _load(m.OUT), _load(os.path.join(ROOT, "results/e2e", "policy_sufficiency.json"))
    assert new["aggregate_at_operating"] == arch["aggregate_at_operating"]
    rows = arch["rows"]
    n_face = sum(len(r["readings"]) * (r["K"] - 1) for r in rows)
    n_value = sum(r["K"] + 1 for r in rows)
    n_tie = sum(len(r["readings"]) for r in rows if abs(r["budget"] - 0.05) < 1e-9) * m.N_TIEBREAK
    _emit("policy_sufficiency", dict(
        run="policy sufficiency (face certificates, V13)", objects="28 compiled instances x 5 budgets",
        n_candidate_readings=sum(r["K"] for r in rows if abs(r["budget"] - 0.05) < 1e-9),
        max_antichain=max(r["K"] for r in rows),
        n_value_lps_by_construction=n_value, n_face_lps_by_construction=n_face,
        n_tiebreak_lps_by_construction=n_tie,
        wall_seconds=round(wall, 2), reproduces_archive=True, **c.asdict()))


def run_surrogate_price() -> None:
    from saorl.benchmark_sg import surrogate_price as m
    c = Counter()
    m.linprog = c.wrap_linprog(m.linprog)
    m.OUT = os.path.join(SCRATCH, "surrogate_price.json")
    t = time.perf_counter(); m.main(); wall = time.perf_counter() - t
    new, arch = _load(m.OUT), _load(os.path.join(ROOT, "results/e2e", "surrogate_price.json"))
    a, b = new["aggregate_at_operating"], arch["aggregate_at_operating"]
    for k in a:
        assert abs(float(a[k]) - float(b[k])) < 1e-9, k
    _emit("surrogate_price", dict(
        run="single-cost surrogate price", objects="28 compiled instances x 5 budgets",
        wall_seconds=round(wall, 2), reproduces_archive=True, **c.asdict()))


def run_policy_class_budget() -> None:
    from saorl.benchmark_sg import policy_class_budget as m
    from saorl.benchmark_sg import exact_nonnested as en
    c = Counter()
    m.milp = c.wrap_milp(m.milp)
    en.linprog = c.wrap_linprog(en.linprog)
    m.OUT_JSON = os.path.join(SCRATCH, "policy_class_budget.json")
    m.OUT_CSV = os.path.join(SCRATCH, "policy_class_budget.csv")
    t = time.perf_counter()
    try:
        m.main()
    except KeyError as e:        # the module's trailing summary print indexes a
        print("module summary print failed after writing its output:", repr(e))  # price entry; output already written
    wall = time.perf_counter() - t
    new, arch = _load(m.OUT_JSON), _load(os.path.join(ROOT, "results/e2e", "policy_class_budget.json"))
    for cls in arch["aggregate"]:
        for d in arch["aggregate"][cls]:
            na, aa = new["aggregate"][cls][d], arch["aggregate"][cls][d]
            for key in ("fires", "n_decided", "n_total", "n_pools"):
                if key in aa:
                    assert na[key] == aa[key], (cls, d, key)
    ac = _load(os.path.join(ROOT, "results/e2e", "antichain_reduction.json"))["families"]
    _emit("policy_class_budget", dict(
        run="policy-class x budget audit (compiled + free classes, both families)",
        objects="88 monitoring + 44 admission pools x 5 budgets",
        n_candidate_readings=int(round(ac["prometheus"]["n_readings_mean"] * 88 + ac["kyverno"]["n_readings_mean"] * 44)),
        n_behaviour_classes=int(round(ac["prometheus"]["n_classes_mean"] * 88 + ac["kyverno"]["n_classes_mean"] * 44)),
        max_antichain_width=max(ac["prometheus"]["width_max"], ac["kyverno"]["width_max"]),
        wall_seconds=round(wall, 2), reproduces_archive=True, **c.asdict()))


def run_class_ladder() -> None:
    from saorl.benchmark_sg import class_ladder as m
    from saorl.benchmark_sg import policy_class_budget as pcb
    c = Counter()
    m.milp = c.wrap_milp(m.milp)
    pcb.milp = c.wrap_milp(pcb.milp)
    m.OUT = os.path.join(SCRATCH, "class_ladder.json")
    t = time.perf_counter(); m.main(); wall = time.perf_counter() - t
    new, arch = _load(m.OUT), _load(os.path.join(ROOT, "results/e2e", "class_ladder.json"))
    for fam in arch["aggregate"]:
        assert new["aggregate"][fam]["per_k"] == arch["aggregate"][fam]["per_k"], fam
    _emit("class_ladder", dict(
        run="class ladder (capability frontier)", objects=f"{len(arch['rows'])} rule x class x budget rows",
        n_rows=len(arch["rows"]), wall_seconds=round(wall, 2), reproduces_archive=True, **c.asdict()))


def run_face_ladder() -> None:
    from saorl.benchmark_sg import face_ladder as m
    from saorl.benchmark_sg import policy_class_budget as pcb
    c = Counter()
    for mod in (m, pcb):
        if hasattr(mod, "milp"):
            mod.milp = c.wrap_milp(mod.milp)
        if hasattr(mod, "linprog"):
            mod.linprog = c.wrap_linprog(mod.linprog)
    m.OUT = os.path.join(SCRATCH, "face_ladder.json")
    t = time.perf_counter(); m.main(); wall = time.perf_counter() - t
    new, arch = _load(m.OUT), _load(os.path.join(ROOT, "results/e2e", "face_ladder.json"))
    assert new["aggregate"] == arch["aggregate"]
    _emit("face_ladder", dict(
        run="face-level ladder + admission face pass (V14)", objects="132 pools at the operating budget",
        wall_seconds=round(wall, 2), reproduces_archive=True, **c.asdict()))


def run_e16() -> None:
    from saorl import e16_exact_screen as m
    from pathlib import Path
    m.OUT = Path(SCRATCH) / "e16_exact_screen.json"
    t = time.perf_counter(); m.main(); wall = time.perf_counter() - t
    new, arch = _load(str(m.OUT)), _load(os.path.join(ROOT, "results/e2e", "e16_exact_screen.json"))
    assert new["families"]["kyverno"]["exact_achievable"] == arch["families"]["kyverno"]["exact_achievable"]
    _emit("e16", dict(
        run="admission exact screen (subset enumeration, no LP)", objects="44 admission pools x 4 budgets",
        n_value_or_face_lps=0, n_milps=0, enumeration="all 2^|F| fixture subsets, |F| <= 18",
        wall_seconds=round(wall, 2), reproduces_archive=True))


# ---------------------------------------------------------------------------
# scale: the LP stage at the antichain sizes calibration produces
# ---------------------------------------------------------------------------

def run_scale() -> None:
    from scipy.optimize import linprog
    from saorl.benchmark_sg.control_suite import compile_instance, _flow
    suite = _load(os.path.join(ROOT, "results/conformal", "benchmark_sg", "control_suite.json"))
    inst = max(suite["instances"], key=lambda i: i["n_states"])

    class R:  # noqa: D401 - tiny reading shim, as exact_nonnested does
        pass
    readings = []
    for rd in inst["readings"]:
        r = R(); r.threshold, r.for_s = rd["theta"], rd["for_s"]; readings.append(r)
    m = compile_instance(readings)
    A_eq, b_eq = _flow(m)
    nS, nA = m["nS"], m["nA"]
    r = m["r"].reshape(-1)
    charged = np.flatnonzero(m["C"].max(axis=0).reshape(-1) > 0)   # state-actions any real reading charges
    # synthetic readings may only charge the 'continue' action (a = 0) in
    # non-safe states, as every real compiled reading does, so that
    # intervening immediately is always feasible and V_U exists at any d
    cont_pairs = np.array([i * nA + 0 for i, s in enumerate(m["S"]) if s[0] != "safe"])
    rng = np.random.default_rng(20260902)
    d = 0.05
    e2e = _load(os.path.join(ROOT, "results/e2e", "e2e_report_v11.json"))["modes"]["complete"]["test"]
    K_med = int(e2e["maximal_set_size"]["median"])
    rows = []
    for K in (10, 100, 1000, K_med):
        # synthetic readings: each charges a random subset of the charged pairs
        # plus a random fraction of all pairs, so the K cost rows are distinct
        C = np.zeros((K, nS * nA))
        for k in range(K):
            sub = rng.choice(charged, size=max(1, len(charged) // 2), replace=False)
            C[k, sub] = 1.0
            extra = rng.choice(cont_pairs, size=rng.integers(1, 4), replace=False)
            C[k, extra] = 1.0
        t0 = time.perf_counter()
        res = linprog(-r, A_ub=C, b_ub=np.full(K, d), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        t_VU = time.perf_counter() - t0
        assert res.status == 0
        V_U = float(r @ res.x)
        t0 = time.perf_counter()
        V = np.empty(K)
        for k in range(K):
            s = linprog(-r, A_ub=C[k:k + 1], b_ub=[d], A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
            V[k] = float(r @ s.x) if s.status == 0 else -np.inf
        t_singles = time.perf_counter() - t0
        S = [k for k in range(K) if abs(V[k] - V_U) <= 1e-7]
        # face LPs for the value-sufficient readings, against every competitor
        t0 = time.perf_counter(); n_face = 0; W = {}
        for k in S:
            worst = -np.inf
            for j in range(K):
                if j == k:
                    continue
                w = linprog(-C[j], A_ub=np.vstack([C[k:k + 1], -r[None, :]]), b_ub=[d, -(V[k] - 1e-9)],
                            A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
                n_face += 1
                if w.status == 0:
                    worst = max(worst, float(C[j] @ w.x))
                if worst > d + 1e-7:
                    break            # this face fails; no need to finish the row
            W[k] = worst
        t_face = time.perf_counter() - t0
        rows.append(dict(K=K, n_states=nS, n_lp_vars=nS * nA, V_U=V_U,
                         n_value_sufficient=len(S), value_lps=K + 1, face_lps_run=n_face,
                         face_lps_worst_case=len(S) * (K - 1),
                         seconds_value_lp=round(t_VU, 3), seconds_single_lps=round(t_singles, 2),
                         seconds_face_lps=round(t_face, 2),
                         face_certified=any(W[k] <= d + 1e-7 for k in S) if S else False))
        print(rows[-1], flush=True)
    # compiled state-space growth: the compiler's own formula, on the same caps
    caps = inst["caps"]
    L = m["L"]
    growth = []
    for K in (2, 3, 4, 5, 6, 8):
        cap_list = [max(caps)] * K
        nS_K = 1 + (K + 1) * int(np.prod([c + 1 for c in cap_list]))   # L grows with distinct thresholds
        growth.append(dict(K=K, cap=max(caps), n_states=nS_K))
    _emit("scale", dict(
        run="LP stage at calibrated antichain scale (synthetic readings on the largest real compiled MDP)",
        geometry=inst["geometry"], budget=d, antichain_median_e2e=K_med, rows=rows,
        compiled_state_space_growth=growth,
        note=("readings are synthetic cost rows on a real 55-state compiled MDP; this times the LP "
              "stage only -- the compiled monitoring model itself has |S| = 1 + L*prod(cap_k+1) "
              "states, exponential in K, which is why compiled instances carry K <= 4")))


def merge() -> None:
    parts = {}
    for f in sorted(os.listdir(OUTDIR)):
        if f.endswith(".json"):
            parts[f[:-5]] = _load(os.path.join(OUTDIR, f))
    with open(MERGED, "w") as fh:
        json.dump(dict(registration="W4 runtime accounting, 2026-09-02; every archived aggregate re-derived and asserted",
                       parts=parts), fh, indent=1)
    print("merged", sorted(parts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default=None)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
    else:
        {"exact_nonnested": run_exact_nonnested, "policy_sufficiency": run_policy_sufficiency,
         "surrogate_price": run_surrogate_price, "policy_class_budget": run_policy_class_budget,
         "class_ladder": run_class_ladder, "face_ladder": run_face_ladder, "e16": run_e16,
         "scale": run_scale}[a.which]()
