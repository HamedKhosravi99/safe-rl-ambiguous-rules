"""Dev-time asset build: harvest the public metric catalog.

Deliberately NOT part of `corset_e2e.generator`. Metric names cannot be
enumerated from a grammar, so the catalog is a frozen public asset harvested
once from the DEVELOPMENT split. Keeping this out of the generator package
means the generator has no code path to the hidden store at all, which the
dependency scan in leak_audit.py verifies statically.
"""
from __future__ import annotations

import json
import os
from typing import Sequence


def build_catalog(dev_ids: Sequence[str], hidden_dir: str, out_path: str) -> dict:
    """Harvest the public metric catalog from the DEVELOPMENT split only."""
    metrics = set()
    for rid in dev_ids:
        h = json.load(open(os.path.join(hidden_dir, f"{rid}.target.json")))
        if h.get("parser_status") == "OK" and h.get("parsed"):
            metrics.add(h["parsed"]["metric"])
    payload = dict(source="development split only", n_metrics=len(metrics),
                   metrics=sorted(metrics))
    json.dump(payload, open(out_path, "w"), indent=1)
    return payload
