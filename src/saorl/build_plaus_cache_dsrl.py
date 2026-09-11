"""Score the DSRL/OSRL episodic-cost rule with the frozen 7-persona ensemble.

The DSRL benchmark experiments (pace/dsrl_sweep.py) previously ADOPTED the
canonical OSRL thresholds {10,20,40} as the retained set by convention.  Under
CORSET the set must come from the same construction as every other rule: the
natural-language cost rule below is scored by the identical 7-persona Claude
ensemble (saorl.judge.ENSEMBLE_PERSONAS, one independent CLI call per
(candidate, persona)), cached here, and thresholded at the corpus-calibrated
conformal q-hat.  No data-calibration gates exist for DSRL (the benchmark has
no labeled incident/normal split), so the composite score reduces to the
language term alone -- validity is unaffected (Theorem 1 holds for any frozen
score); only efficiency depends on it.

Run once:  PYTHONPATH=. python3 -m saorl.build_plaus_cache_dsrl
Output:    saorl/plausibility_cache_dsrl.json
"""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

from .dsl import Atom, Candidate
from .judge import ENSEMBLE_PERSONAS, LLMEnsembleJudge

RULE_TEXT = (
    "Keep the robot's accumulated safety cost low over an episode -- brief, "
    "incidental contact with the constraint region is tolerable, but do not "
    "run up a substantial total violation."
)

GLOSSARY = (
    "Feature glossary (one Bullet-Safety-Gym control episode):\n"
    "- cost: every timestep spent in the constraint region (collision, "
    "boundary or velocity violation) adds 1 to the episode's accumulated "
    "safety cost; episodes run a few hundred steps.\n"
    "- An unconstrained, reward-greedy policy typically accrues an episodic "
    "cost of roughly 60-200 on these tasks; a policy that never touches the "
    "constraint region accrues 0.\n"
    "- A cost-budget reading with budget b treats the rule as violated once "
    "the episode's accumulated cost exceeds b."
)

ITEMS: List[Tuple[Candidate, str]] = [
    (Candidate("psi_b10: cost<=10", Atom("cost", "ge", 10), frozenset({"operate"})),
     "the episode's accumulated safety cost must stay at or below 10 "
     "(strict: near-zero contact with the constraint region)"),
    (Candidate("psi_b20: cost<=20", Atom("cost", "ge", 20), frozenset({"operate"})),
     "the episode's accumulated safety cost must stay at or below 20 "
     "(moderate: brief contact tolerated, sustained contact not)"),
    (Candidate("psi_b40: cost<=40", Atom("cost", "ge", 40), frozenset({"operate"})),
     "the episode's accumulated safety cost must stay at or below 40 "
     "(lenient: a sizable fraction of the episode may touch the region)"),
]


def _model_id() -> str:
    try:
        out = subprocess.run(["claude", "--version"], capture_output=True,
                             text=True, timeout=30)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def build(max_workers: int = 8) -> dict:
    judge = LLMEnsembleJudge(
        rule_text=RULE_TEXT,
        glosses={c.name: g for c, g in ITEMS},
        glossary=GLOSSARY,
        domain_noun="robot-safety",
        action_gloss="'keep operating in the constraint region'",
    )
    jobs = [(cand, persona) for cand, _ in ITEMS for persona in ENSEMBLE_PERSONAS]

    def one(job):
        cand, persona = job
        return cand.name, persona, judge._ask(judge._prompt(persona, cand))

    scores: Dict[str, Dict[str, float]] = {c.name: {} for c, _ in ITEMS}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for name, persona, v in ex.map(one, jobs):
            if v is not None:
                scores[name][persona] = v

    ordered = {
        name: [scores[name][p] for p in ENSEMBLE_PERSONAS if p in scores[name]]
        for name in scores
    }
    return {
        "model": _model_id(),
        "personas": list(ENSEMBLE_PERSONAS),
        "sets": {
            "dsrl_cost": {"rule_text": RULE_TEXT, "scores": ordered},
        },
    }


def main() -> None:
    cache = build()
    out = Path(__file__).with_name("plausibility_cache_dsrl.json")
    json.dump(cache, open(out, "w"), indent=1)
    print(f"model: {cache['model']}")
    for name, vals in cache["sets"]["dsrl_cost"]["scores"].items():
        mean = sum(vals) / len(vals) if vals else float("nan")
        print(f"  {name:22s} n={len(vals)} mean={mean:.3f} {vals}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
