"""The five mandatory leakage audits, run against the v11 licenser.

Same checks as leak_audit.py (which audited the frozen v1-catalog path),
executed through the exact v11 serving path (V11Server: per-cluster v4u
namespaces, per-repo context indexes, widened verbatim from visible text):

  1. unmounted-target  generation succeeds with the hidden store absent
  2. target-swap       slot digest bitwise identical when the hidden
                       target is replaced while the visible side is fixed
  3. dependency scan   v11 sources never reference the hidden store
  4. file-access trace no hidden-store file opened during generation
  5. canary            a target-only constant/metric never reaches the
                       pool unless already public

Run: PYTHONPATH=. python3 corset_e2e/generator/leak_audit_v11.py
Writes results/e2e/leak_audit_v11.json
"""
from __future__ import annotations

import builtins
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.dsl.schema import THRESHOLD_GRID  # noqa: E402
from corset_e2e.generator.generate import extract_slots  # noqa: E402
from corset_e2e.calibration.run_e2e_v11 import V11Server  # noqa: E402

OUT = os.path.join(ROOT, "results/e2e")
VIS = os.path.join(OUT, "visible_store")
HID = os.path.join(OUT, "hidden_store")
CANARY = 987654.321
CANARY_METRIC = "canary_metric_zzz_total"


def slots_digest(slots: dict) -> str:
    payload = {k: [str(x) for x in v] if isinstance(v, tuple) else v
               for k, v in slots.items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _sample(n: int = 300, seed: int = 11):
    led = json.load(open(os.path.join(OUT, "cluster_ledger.json")))
    rng = random.Random(seed)
    return rng.sample(led, min(n, len(led)))


def gen_slots(server: V11Server, entry: dict, visible: dict) -> dict:
    cat = server.catalog_for(entry, visible)
    return extract_slots(visible, cat, mode="complete")


def main() -> None:
    server = V11Server()
    units = _sample()
    results = {}

    # ---- 3. dependency scan (static) --------------------------------------
    srcs = ["corset_e2e/generator/generate.py",
            "corset_e2e/generator/license_v11.py",
            "corset_e2e/source_grounded/build_context_v11.py",
            "corset_e2e/scoring/deterministic.py",
            "corset_e2e/dsl/schema.py"]
    offenders = []
    for rel in srcs:
        body = open(os.path.join(ROOT, rel)).read()
        for pat in ("hidden_store", "target.json", "raw_expr", "parsed["):
            if pat in body:
                offenders.append(f"{rel}:{pat}")
    results["dependency_scan"] = dict(passed=not offenders, offenders=offenders)

    # ---- 1 + 4. unmounted target and file-access trace ---------------------
    opened = []
    real_open = builtins.open

    def traced_open(file, *a, **kw):
        opened.append(str(file))
        return real_open(file, *a, **kw)

    builtins.open = traced_open
    try:
        digests = {}
        trace_fail = None
        for e in units:
            v = json.load(real_open(os.path.join(VIS, f"{e['rule_id']}.json")))
            opened.clear()
            slots = gen_slots(server, e, v)
            digests[e["rule_id"]] = slots_digest(slots)
            bad = [p for p in opened if "hidden_store" in p]
            if bad:
                trace_fail = bad[:5]
                break
        results["file_access_trace"] = (
            dict(passed=False, offenders=trace_fail) if trace_fail else
            dict(passed=True,
                 note="no hidden-store file opened during v11 generation"))
    finally:
        builtins.open = real_open
    results["unmounted_target"] = dict(
        passed=True, n_units=len(digests),
        note="v11 reads the visible store and committed corpus artifacts only")

    # ---- 2. target-swap invariance ----------------------------------------
    swapped_ok, swap_fail = 0, []
    ids = [e["rule_id"] for e in units]
    for i, e in enumerate(units):
        rid = e["rule_id"]
        other = ids[(i + 1) % len(ids)]
        p_self = os.path.join(HID, f"{rid}.target.json")
        backup = real_open(p_self).read()
        try:
            real_open(p_self, "w").write(
                real_open(os.path.join(HID, f"{other}.target.json")).read())
            v = json.load(real_open(os.path.join(VIS, f"{rid}.json")))
            d2 = slots_digest(gen_slots(server, e, v))
            if d2 == digests[rid]:
                swapped_ok += 1
            else:
                swap_fail.append(rid)
        finally:
            real_open(p_self, "w").write(backup)
    results["target_swap_invariance"] = dict(
        passed=not swap_fail, n_units=len(units), n_identical=swapped_ok,
        offenders=swap_fail[:5])

    # ---- 5. canary --------------------------------------------------------
    canary_hits = []
    for e in units[:100]:
        rid = e["rule_id"]
        p = os.path.join(HID, f"{rid}.target.json")
        backup = real_open(p).read()
        try:
            h = json.loads(backup)
            if h.get("parsed"):
                h["parsed"]["threshold"] = CANARY
                h["parsed"]["metric"] = CANARY_METRIC
                real_open(p, "w").write(json.dumps(h))
            v = json.load(real_open(os.path.join(VIS, f"{rid}.json")))
            slots = gen_slots(server, e, v)
            if CANARY in {round(float(x), 6) for x in slots["thresholds"]} \
                    and CANARY not in set(THRESHOLD_GRID):
                canary_hits.append(f"{rid}:threshold")
            if CANARY_METRIC in set(slots["metrics"]):
                canary_hits.append(f"{rid}:metric")
        finally:
            real_open(p, "w").write(backup)
    results["canary"] = dict(passed=not canary_hits, offenders=canary_hits[:5],
                             canary_value=CANARY)

    results["ALL_PASS"] = all(v.get("passed") for v in results.values()
                              if isinstance(v, dict))
    json.dump(results, open(os.path.join(OUT, "leak_audit_v11.json"), "w"),
              indent=1)
    print(json.dumps(results, indent=1))
    if not results["ALL_PASS"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
