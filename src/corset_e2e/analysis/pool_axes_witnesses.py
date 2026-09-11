"""V11B endpoints 4-5: behavioral witnesses and per-axis ambiguity shares.

Registered in REGISTRATION_V11B.md before computation. Population: the
source-grounded Prometheus pools (the paper's monitoring corpus), built
by the archived pool constructor. Two facts are surfaced:

WITNESSES. Pool candidates are merged into behavior classes by their
fingerprint over the fixture universe, so every pair of distinct retained
classes differs on at least one fixture BY CONSTRUCTION -- the witness is
that fixture. This script makes the statement checkable (asserts witness
coverage is exactly 1.0 over all class pairs), counts the syntactic
paraphrases the merge removed (candidates vs classes -- the semantic
deduplication statistic), and emits one decoded witness per pool for the
paper's exhibit.

AXIS SHARES. Among pools with at least two behavior classes: the
fraction whose classes differ on each grammar axis (metric, comparator,
threshold, window, for-duration, aggregation). Admission pools are out of
scope here -- their readings differ on validate predicates, not these
axes -- and the paper says so.

Run: PYTHONPATH=. SAORL_CONFORMAL=1 python3 corset_e2e/analysis/pool_axes_witnesses.py
Writes results/e2e/pool_axes_witnesses.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))
os.environ.setdefault("SAORL_CONFORMAL", "1")

from saorl.benchmark_sg.parse import parse_prometheus, prom_threshold_bank  # noqa: E402
from saorl.benchmark_sg.evaluate import build_prom_pool  # noqa: E402

OUT = os.path.join(ROOT, "results/e2e")
AXES = ("metric", "comparator", "threshold", "window", "for_s",
        "aggregation")


def axis_value(rep, axis):
    for name in ((axis,) if axis != "window" else ("window", "range_s",
                                                   "rate_window_s")):
        if hasattr(rep, name):
            return getattr(rep, name)
    return None


def main() -> None:
    pt, _ = parse_prometheus()
    bank = prom_threshold_bank(pt)
    pools = []
    for ti, t in enumerate(pt):
        try:
            pool = build_prom_pool(t, bank)
        except Exception:
            continue
        pools.append((f'{getattr(t, "name", "?")}#{ti}', pool))

    n_multi = 0
    tot_cands = tot_classes = tot_pairs = tot_witnessed = 0
    axis_counts: Counter = Counter()
    exhibits = []
    for uid, pool in pools:
        classes = pool.classes
        n_c = len(classes)
        n_raw = sum(len(getattr(c, "members", [])) or 1 for c in classes)
        tot_cands += max(n_raw, n_c)
        tot_classes += n_c
        if n_c < 2:
            continue
        n_multi += 1
        reps = [c.rep for c in classes]
        vecs = [tuple(c.vector) for c in classes]
        first_witness = None
        for i in range(n_c):
            for j in range(i + 1, n_c):
                tot_pairs += 1
                diff = [x for x, (a, b) in enumerate(zip(vecs[i], vecs[j]))
                        if a != b]
                assert diff, (uid, i, j,
                              "two distinct behavior classes share a "
                              "fingerprint; the merge invariant is broken")
                tot_witnessed += 1
                if first_witness is None:
                    first_witness = dict(
                        pair=(i, j), fixture=diff[0],
                        fires=(vecs[i][diff[0]], vecs[j][diff[0]]))
        for axis in AXES:
            vals = {repr(axis_value(r, axis)) for r in reps}
            if len(vals) > 1:
                axis_counts[axis] += 1
        if len(exhibits) < 3 and first_witness is not None:
            i, j = first_witness["pair"]
            exhibits.append(dict(
                rule=uid, fixture_index=first_witness["fixture"],
                fires=first_witness["fires"],
                reading_a={a: repr(axis_value(reps[i], a)) for a in AXES},
                reading_b={a: repr(axis_value(reps[j], a)) for a in AXES}))

    assert tot_witnessed == tot_pairs, "witness coverage below 1.0"
    payload = dict(
        registration="V11B endpoints 4-5 (see REGISTRATION_V11B.md)",
        n_pools=len(pools), n_pools_multi=n_multi,
        candidates_total=tot_cands, classes_total=tot_classes,
        pairs_total=tot_pairs, witness_coverage=1.0,
        axis_share={a: axis_counts.get(a, 0) / max(n_multi, 1)
                    for a in AXES},
        axis_counts=dict(axis_counts),
        exhibits=exhibits)
    json.dump(payload, open(os.path.join(OUT, "pool_axes_witnesses.json"),
                            "w"), indent=1)
    print(json.dumps({k: v for k, v in payload.items() if k != "exhibits"},
                     indent=1))
    print("exhibit:", json.dumps(exhibits[:1], indent=1))


if __name__ == "__main__":
    main()
