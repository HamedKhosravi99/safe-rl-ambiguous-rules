"""Run the REAL language-plausibility ensemble for Domain 2 (temporal) and cache it.

Domain-2 analogue of saorl.build_plaus_cache. For every candidate of every
temporal rule set (saorl.gridworld.RULE_SETS_T) a 7-member expert prompt ensemble
(saorl.judge.LLMEnsembleJudge, local `claude` CLI) scores how faithfully the
formal temporal predicate reads the natural-language crossing-gate rule. Scores
are serialized to `saorl/plausibility_cache_temporal.json`; the headline
construction then reads that cache (CachedEnsembleJudge) with no model access, so
the temporal U_alpha is reproducible.

The non-circularity question here is specifically *temporal*: does a real
language ensemble agree that "any warning" (alarm OR advisory) is faithful to the
broad warn-rule, yet too broad for the sharp alarm-only rule -- the asymmetry the
temporal kill test relies on? Run (slow; makes LLM calls):

    python3 -m saorl.build_plaus_cache_temporal
"""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

from .gridworld import RULE_SETS_T
from .judge import ENSEMBLE_PERSONAS, LLMEnsembleJudge

CACHE_PATH = Path(__file__).with_name("plausibility_cache_temporal.json")

# Temporal-domain feature glossary: a single warning edge + the past-time window
# operator. The danger DURATION after a warning is latent and somewhat variable,
# so "how many steps to hold" is a genuinely temporal, genuinely ambiguous choice.
TEMPORAL_GLOSSARY = (
    "Setting (a patrol corridor with recurring crossing-style hazards, observed "
    "each step):\n"
    "- warning: a one-step EDGE signal that fires only on the step a hazard "
    "begins. The crossing is dangerous to advance through for a SHORT, somewhat "
    "variable number of steps right after a warning, after which it becomes safe "
    "again. There is no explicit all-clear signal; you must judge how long to "
    "hold.\n"
    "- Within_W(warning): true for the W steps immediately following a warning "
    "edge.\n"
    "- advancing is the risky action; waiting is the safe action."
)

DOMAIN_NOUN = "crossing-gate safety"
ACTION_GLOSS = "'advance'"


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
        name: LLMEnsembleJudge(
            rule_text=spec["rule_text"],
            glosses={c.name: g for c, g in spec["items"]},
            glossary=TEMPORAL_GLOSSARY,
            domain_noun=DOMAIN_NOUN,
            action_gloss=ACTION_GLOSS,
        )
        for name, spec in RULE_SETS_T.items()
    }
    tasks = []
    for sname, spec in RULE_SETS_T.items():
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
    for sname, spec in RULE_SETS_T.items():
        entry = {"rule_text": spec["rule_text"], "scores": {}}
        for cand, _ in spec["items"]:
            vals = [results[(sname, cand.name, pi)]
                    for pi in range(len(ENSEMBLE_PERSONAS))]
            entry["scores"][cand.name] = [v for v in vals if v is not None]
        cache["sets"][sname] = entry
    return cache


def main():
    n = sum(len(s["items"]) for s in RULE_SETS_T.values())
    print(f"Scoring {n} temporal candidates x {len(ENSEMBLE_PERSONAS)} personas "
          f"with the real `claude` ensemble...")
    cache = build()
    CACHE_PATH.write_text(json.dumps(cache, indent=2))
    print(f"\nWrote {CACHE_PATH}")
    for sname, entry in cache["sets"].items():
        print(f"\n  {sname}: \"{entry['rule_text']}\"")
        for name, vals in entry["scores"].items():
            mean = sum(vals) / len(vals) if vals else float("nan")
            print(f"    {name:28s} mean={mean:.3f}  n={len(vals)}  {vals}")


if __name__ == "__main__":
    main()
