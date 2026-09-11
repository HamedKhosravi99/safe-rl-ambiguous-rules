"""Run the REAL language-plausibility ensemble once and serialize it.

This is the offline step behind SA-ORL's non-circularity claim. For every
candidate of every rule, a 7-member expert prompt ensemble (saorl.judge.
LLMEnsembleJudge, backed by the local `claude` CLI) scores how faithfully the
formal predicate reads the natural-language rule. The scores are written to
`saorl/plausibility_cache.json`; the headline construction then reads that cache
(CachedEnsembleJudge) and needs no model access, so U_alpha is reproducible.

The point is to check whether a real ensemble reproduces the hand-set structure
the kill test relied on -- in particular, whether the conservative anomaly
reading stays plausible enough to be retained. Run (slow; makes LLM calls):

    python3 -m saorl.build_plaus_cache
"""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

from .dsl import Atom, Candidate, Persist
from .judge import ENSEMBLE_PERSONAS, LLMEnsembleJudge

FORBID = frozenset({"continue"})
CACHE_PATH = Path(__file__).with_name("plausibility_cache.json")

SEVERE = "Do not continue operating the asset when its degradation is severe."
WEAROUT = (
    "Do not continue operating once the asset has entered its final, critical "
    "wear-out phase very close to failure."
)

# (rule_text, [(candidate, gloss)]) for each rule set we score.
RULE_SETS: Dict[str, dict] = {
    "ambiguous": dict(
        rule_text=SEVERE,
        items=[
            (Candidate("psi1: RUL_hat<20", Atom("rul_hat", "le", 20), FORBID),
             "the estimated remaining useful life rul_hat is at most 20 cycles"),
            (Candidate("psi2: Q05<10", Atom("q05", "le", 10), FORBID),
             "the conservative lower-bound estimate q05 is at most 10 cycles"),
            (Candidate("psi3: anom>0.8", Atom("anom", "ge", 0.8), FORBID),
             "the anomaly score anom is at least 0.8"),
            (Candidate("psi4: Persist3(RUL<25)",
                       Persist(3, Atom("rul_hat", "le", 25)), FORBID),
             "rul_hat has stayed at or below 25 cycles for 3 consecutive cycles"),
        ],
    ),
    "sharp": dict(
        rule_text=WEAROUT,
        items=[
            (Candidate("phi1: RUL_hat<15", Atom("rul_hat", "le", 15), FORBID),
             "rul_hat is at most 15 cycles"),
            (Candidate("phi2: anom>0.3", Atom("anom", "ge", 0.3), FORBID),
             "the anomaly score anom is at least 0.3"),
            (Candidate("phi3: RUL_hat<85", Atom("rul_hat", "le", 85), FORBID),
             "rul_hat is at most 85 cycles"),
            (Candidate("phi4: RUL_hat<3", Atom("rul_hat", "le", 3), FORBID),
             "rul_hat is at most 3 cycles"),
        ],
    ),
    "conservative": dict(
        rule_text=SEVERE,
        items=[
            (Candidate("liberal: RUL<12", Atom("rul_hat", "le", 12), FORBID),
             "rul_hat is at most 12 cycles"),
            (Candidate("mid: RUL<22", Atom("rul_hat", "le", 22), FORBID),
             "rul_hat is at most 22 cycles"),
            (Candidate("consv: anom>0.6", Atom("anom", "ge", 0.6), FORBID),
             "the anomaly score anom is at least 0.6"),
            (Candidate("vconsv: RUL<40", Atom("rul_hat", "le", 40), FORBID),
             "rul_hat is at most 40 cycles"),
        ],
    ),
}


def _model_id() -> str:
    try:
        out = subprocess.run(
            ["claude", "--version"], capture_output=True, text=True, timeout=30
        )
        return out.stdout.strip() or "claude-cli"
    except Exception:
        return "claude-cli"


def build(max_workers: int = 8) -> dict:
    """Score every (set, candidate, persona) in parallel; return the cache dict."""
    judges = {
        name: LLMEnsembleJudge(rule_text=spec["rule_text"],
                               glosses={c.name: g for c, g in spec["items"]})
        for name, spec in RULE_SETS.items()
    }
    # flat task list: one LLM call per (set, candidate, persona index)
    tasks = []
    for sname, spec in RULE_SETS.items():
        for cand, _ in spec["items"]:
            for pi, persona in enumerate(ENSEMBLE_PERSONAS):
                tasks.append((sname, cand, pi, persona))

    results: Dict[tuple, float] = {}

    def run(task):
        sname, cand, pi, persona = task
        v = judges[sname]._ask(judges[sname]._prompt(persona, cand))
        return (sname, cand.name, pi), v

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for key, v in ex.map(run, tasks):
            results[key] = v
            done += 1
            print(f"  [{done}/{len(tasks)}] {key[0]}/{key[1]} m{key[2]} -> {v}")

    cache = {"model": _model_id(), "personas": list(ENSEMBLE_PERSONAS), "sets": {}}
    for sname, spec in RULE_SETS.items():
        entry = {"rule_text": spec["rule_text"], "scores": {}}
        for cand, _ in spec["items"]:
            vals = [results[(sname, cand.name, pi)]
                    for pi in range(len(ENSEMBLE_PERSONAS))]
            entry["scores"][cand.name] = [v for v in vals if v is not None]
        cache["sets"][sname] = entry
    return cache


def main():
    print(f"Scoring {sum(len(s['items']) for s in RULE_SETS.values())} candidates "
          f"x {len(ENSEMBLE_PERSONAS)} personas with the real `claude` ensemble...")
    cache = build()
    CACHE_PATH.write_text(json.dumps(cache, indent=2))
    print(f"\nWrote {CACHE_PATH}")
    for sname, entry in cache["sets"].items():
        print(f"\n  {sname}: \"{entry['rule_text']}\"")
        for name, vals in entry["scores"].items():
            mean = sum(vals) / len(vals) if vals else float("nan")
            print(f"    {name:24s} mean={mean:.3f}  n={len(vals)}  {vals}")


if __name__ == "__main__":
    main()
