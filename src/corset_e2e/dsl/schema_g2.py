"""G2: a bounded expression-tree grammar, declared by a dev-derived spec.

A reading is no longer a six-tuple but a parsed expression, so nothing has
to be projected away and the gold can be the rule itself. What keeps the
space bounded and auditable is that G2 is not "PromQL": it is the frozen
grammar plus an explicitly enumerated capability set and explicitly
enumerated scalar axes, all chosen on the development split
(analysis/derive_g2_from_dev.py) and read from results/e2e/G2_SPEC.json.

Membership is decided by the same featurizer used to design the grammar,
evaluated against the ENLARGED grids: an expression is in G2 when every
capability it requires is declared and every scalar it names is on an
axis. That makes reach auditable -- for any excluded rule the answer to
"why" is the specific capability or value that is missing.

The theory is unaffected by the representation change. Theorem 1 and the
face theorem need only that each reading induce a cost function evaluable
on the unit's fixtures; an AST does that exactly as a six-tuple did.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SPEC_PATH = os.path.join(ROOT, "results/e2e", "G2_SPEC.json")

IN_G2 = "IN_G2"
OUT_G2_CAPABILITY = "OUT_OF_G2_CAPABILITY"
OUT_G2_UNPARSED = "OUT_OF_G2_UNPARSED"


@lru_cache(maxsize=1)
def spec() -> dict:
    with open(SPEC_PATH) as fh:
        s = json.load(fh)
    from corset_e2e.dsl.schema import FOR_S, RATE_WINDOWS_S, THRESHOLD_GRID
    s["_thresholds"] = ({round(float(x), 6) for x in THRESHOLD_GRID}
                        | {round(float(x), 6) for x in s["threshold_extra"]})
    s["_windows"] = ({float(w) for w in RATE_WINDOWS_S if w is not None}
                     | {float(w) for w in s["window_extra"]})
    s["_fors"] = ({float(x) for x in FOR_S}
                  | {float(x) for x in s["for_extra"]})
    s["_caps"] = frozenset(s["capabilities"])
    return s


def required(hidden: dict):
    """Capabilities the rule needs, measured against G2's enlarged axes."""
    from corset_e2e.analysis.grammar_taxonomy import features_for
    s = spec()
    return features_for(hidden, thresholds=s["_thresholds"],
                        windows=s["_windows"], fors=s["_fors"])


def classify_target_g2(hidden: dict) -> str:
    feats, _how = required(hidden)
    if feats is None:
        return OUT_G2_UNPARSED
    return IN_G2 if feats <= spec()["_caps"] else OUT_G2_CAPABILITY


def missing_capabilities(hidden: dict) -> frozenset:
    feats, _how = required(hidden)
    if feats is None:
        return frozenset({"<unparsed>"})
    return frozenset(feats) - spec()["_caps"]
