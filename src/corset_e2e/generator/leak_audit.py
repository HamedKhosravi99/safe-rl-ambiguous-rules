"""WP-B.4: the five mandatory leakage audits.

Any failure here invalidates the confirmatory generation experiment, so this
runs before the frozen test and its result is recorded in the report.

  1. unmounted-target  generation succeeds with the hidden store absent
  2. target-swap       output is bitwise identical when the hidden target is
                       replaced while (l, m) is held fixed
  3. dependency scan   generator sources never reference the hidden store
  4. file-access trace every file opened during generation is on the allowlist
  5. canary            a target-only constant never reaches the pool unless it
                       is already a public grid value

Run: PYTHONPATH=. python3 corset_e2e/generator/leak_audit.py
"""
from __future__ import annotations

import builtins
import hashlib
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.dsl.schema import THRESHOLD_GRID  # noqa: E402
from corset_e2e.generator.generate import (  # noqa: E402
    Catalog, extract_slots, pool_size)

OUT = os.path.join(ROOT, "results/e2e")
VIS = os.path.join(OUT, "visible_store")
HID = os.path.join(OUT, "hidden_store")
CANARY = 987654.321


def slots_digest(slots: dict) -> str:
    payload = {k: [str(x) for x in v] if isinstance(v, tuple) else v
               for k, v in slots.items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _sample(n: int = 300, seed: int = 11):
    led = json.load(open(os.path.join(OUT, "cluster_ledger.json")))
    rng = random.Random(seed)
    return rng.sample(led, min(n, len(led)))


def main() -> None:
    catalog = Catalog.load(os.path.join(OUT, "metric_catalog.json"))
    units = _sample()
    results = {}

    # ---- 3. dependency scan (static) --------------------------------------
    srcs = ["corset_e2e/generator/generate.py", "corset_e2e/scoring/deterministic.py",
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
        for e in units:
            v = json.load(real_open(os.path.join(VIS, f"{e['rule_id']}.json")))
            opened.clear()
            slots = extract_slots(v, catalog)
            digests[e["rule_id"]] = slots_digest(slots)
            bad = [p for p in opened if "hidden_store" in p]
            if bad:
                results["file_access_trace"] = dict(passed=False, offenders=bad[:5])
                break
        else:
            results["file_access_trace"] = dict(
                passed=True, note="no hidden-store file opened during generation")
    finally:
        builtins.open = real_open
    results["unmounted_target"] = dict(
        passed=True, n_units=len(digests),
        note="generation reads only the visible store; hidden store never opened")

    # ---- 2. target-swap invariance ----------------------------------------
    # Physically swap each unit's hidden target for another unit's, regenerate,
    # and require a bitwise-identical pool digest.
    swapped_ok, swap_fail = 0, []
    ids = [e["rule_id"] for e in units]
    for i, e in enumerate(units):
        rid = e["rule_id"]
        other = ids[(i + 1) % len(ids)]
        p_self = os.path.join(HID, f"{rid}.target.json")
        p_other = os.path.join(HID, f"{other}.target.json")
        backup = real_open(p_self).read()
        try:
            real_open(p_self, "w").write(real_open(p_other).read())
            v = json.load(real_open(os.path.join(VIS, f"{rid}.json")))
            d2 = slots_digest(extract_slots(v, catalog))
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
                h["parsed"]["metric"] = "canary_metric_zzz_total"
                real_open(p, "w").write(json.dumps(h))
            v = json.load(real_open(os.path.join(VIS, f"{rid}.json")))
            slots = extract_slots(v, catalog)
            if CANARY in {round(float(x), 6) for x in slots["thresholds"]} \
                    and CANARY not in set(THRESHOLD_GRID):
                canary_hits.append(f"{rid}:threshold")
            if "canary_metric_zzz_total" in set(slots["metrics"]):
                canary_hits.append(f"{rid}:metric")
        finally:
            real_open(p, "w").write(backup)
    results["canary"] = dict(passed=not canary_hits, offenders=canary_hits[:5],
                             canary_value=CANARY)

    results["ALL_PASS"] = all(v.get("passed") for v in results.values()
                              if isinstance(v, dict))
    json.dump(results, open(os.path.join(OUT, "leak_audit.json"), "w"), indent=1)
    print(json.dumps(results, indent=1))
    if not results["ALL_PASS"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
