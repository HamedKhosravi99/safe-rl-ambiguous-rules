"""Language-plausibility judges for SA-ORL candidates (human-free replacement
for the original 3-annotator audit).

Plausibility answers one question the data terms cannot: does the predicate psi
actually *mean* what the natural-language rule says? A reading can fire on every
incident (high p_inc) and still be a wrong interpretation -- e.g. "anomaly > 0.3"
for "avoid operating when degradation is severe" is faithful-looking but far too
weak. Only a language-grounded judge catches that.

Design goals (per project direction -- no surveys, fully reproducible):
  * The judge is an *ensemble* scored at scale, not 3 recruited humans.
  * Its output is frozen/cached so U_alpha is reproducible from the cache alone,
    with no network call at construction time.
  * A no-LLM `DataDrivenJudge` ablation is provided so the whole pipeline can run
    with zero external dependencies; it is deliberately weaker (it cannot detect
    a faithful-looking but semantically wrong reading) and exists to quantify how
    much the language signal actually buys.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .dataset import SemanticDataset, p_incident, p_normal
from .dsl import Candidate


class PlausibilityJudge:
    """Maps a candidate to an ensemble of plausibility scores in [0,1]."""

    def scores(self, cand: Candidate) -> Tuple[float, ...]:  # pragma: no cover
        raise NotImplementedError


@dataclass
class CachedEnsembleJudge(PlausibilityJudge):
    """Plausibility from a frozen ensemble of language judgments.

    `cache` maps candidate name -> tuple of member scores (e.g. an LLM ensemble
    of K prompts/temperatures run offline and serialized). Because it is read
    from the cache, U_alpha is fully reproducible and needs no network access.
    """
    cache: Dict[str, Sequence[float]]

    def scores(self, cand: Candidate) -> Tuple[float, ...]:
        return tuple(float(x) for x in self.cache.get(cand.name, ()))


@dataclass
class DataDrivenJudge(PlausibilityJudge):
    """No-LLM ablation: plausibility from incident/normal separation alone.

    Member scores are seeded jitter around clip(p_inc - p_norm). Reproducible and
    dependency-free, but blind to language: a predicate that fires on incidents
    yet does not match the rule's meaning still scores high here. Use it to
    measure the value added by the language ensemble, not as the headline judge.
    """
    dsem: SemanticDataset
    n_members: int = 7
    noise: float = 0.05
    seed: int = 0

    def scores(self, cand: Candidate) -> Tuple[float, ...]:
        import numpy as np

        inc = self.dsem.incident_points()
        nor = self.dsem.normal_points()
        a = self.dsem.activation_set(cand)
        base = max(0.0, min(1.0, p_incident(a, inc) - p_normal(a, nor)))
        rng = np.random.default_rng((self.seed * 1_000_003 + hash(cand.name)) % (2**31))
        vals = np.clip(base + rng.normal(0.0, self.noise, self.n_members), 0.0, 1.0)
        return tuple(float(v) for v in vals)


# --- real LLM ensemble (the headline, run offline then cached) ----------------

# Seven distinct expert framings. Diversity here is what makes this an *ensemble*
# rather than a single opinion: each member reads the same rule + predicate but
# from a different professional stance, so genuine disagreement (not sampling
# temperature) drives the spread. These are frozen so the run is reproducible
# given the same model.
ENSEMBLE_PERSONAS: Tuple[str, ...] = (
    "a reliability engineer who reads the rule strictly and literally",
    "a safety auditor who errs toward catching unsafe operation early",
    "a plain-language reader with no engineering background",
    "a skeptical linguist who penalizes readings that are broader or "
    "narrower than the words actually say",
    "a prognostics ML practitioner who knows these features well",
    "a cost-conscious operations manager who dislikes needless early action",
    "a careful neutral annotator weighing faithfulness on the merits",
)

FEATURE_GLOSSARY = (
    "Feature glossary (one asset, observed each operating cycle):\n"
    "- rul_hat: point estimate of Remaining Useful Life in cycles until "
    "failure; LOWER means MORE degraded. Healthy assets sit at rul_hat >= 80; "
    "an asset is in its incident/wear-out band around rul_hat <= 20; failure "
    "is at 0.\n"
    "- q05: a conservative 5th-percentile lower-bound estimate of RUL "
    "(sits a few cycles below rul_hat).\n"
    "- anom: an anomaly/health score in [0,1]; HIGHER means MORE degraded. "
    "It is ~0.2 on a healthy asset and ~0.8 when rul is around 20.\n"
    "- Persist_m(condition): the condition has held continuously for the last "
    "m cycles."
)


def _parse_score(text: str) -> Optional[float]:
    """Pull the first number in [0,1] out of a model reply; None if none found."""
    for tok in re.findall(r"[-+]?\d*\.?\d+", text):
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v
    return None


@dataclass
class LLMEnsembleJudge(PlausibilityJudge):
    """Real language-plausibility ensemble backed by the local `claude` CLI.

    For each candidate we issue one independent call per persona in
    ENSEMBLE_PERSONAS, asking how faithfully the formal predicate captures the
    natural-language rule `rule_text`. Each reply is a single number in [0,1];
    the tuple of replies is the ensemble. This is meant to be run ONCE offline
    and serialized via `build_cache`; the headline construction then reads the
    cache (CachedEnsembleJudge) with no network/model access.

    `describe` maps a candidate to a human-readable gloss of its predicate. If a
    name is absent from `glosses`, the predicate's own repr is used.
    """
    rule_text: str
    glosses: Dict[str, str] = field(default_factory=dict)
    cli: str = "claude"
    model: Optional[str] = None
    timeout_s: float = 120.0
    glossary: str = FEATURE_GLOSSARY          # domain feature glossary
    domain_noun: str = "maintenance"           # how the rule is described
    action_gloss: str = "'continue operating'"  # the forbidden action

    def describe(self, cand: Candidate) -> str:
        return self.glosses.get(cand.name, repr(cand.predicate))

    def _prompt(self, persona: str, cand: Candidate) -> str:
        return (
            f"You are {persona}.\n\n"
            f"{self.glossary}\n\n"
            f"Natural-language {self.domain_noun} rule:\n  \"{self.rule_text}\"\n\n"
            f"A proposed formal reading of that rule forbids the {self.action_gloss} "
            f"action exactly when this condition holds:\n"
            f"  {self.describe(cand)}\n\n"
            f"On a scale from 0 to 1, how faithfully does this condition capture "
            f"what the rule actually means? 1.0 = an exact, faithful reading; "
            f"0.5 = defensible but imperfect; 0.0 = clearly the wrong reading "
            f"(far too broad, too narrow, or unrelated).\n"
            f"Reply with ONLY a single number between 0 and 1, nothing else."
        )

    def _ask_raw(self, prompt: str) -> Optional[str]:
        """Raw model reply text (None on timeout / missing CLI). Used when the
        caller needs the full reply -- e.g. generated code -- not a parsed score."""
        cmd = [self.cli, "-p", prompt]
        if self.model:
            cmd += ["--model", self.model]
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout_s
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        return out.stdout

    def _ask(self, prompt: str) -> Optional[float]:
        raw = self._ask_raw(prompt)
        return _parse_score(raw) if raw is not None else None

    def scores(self, cand: Candidate) -> Tuple[float, ...]:
        vals = []
        for persona in ENSEMBLE_PERSONAS:
            v = self._ask(self._prompt(persona, cand))
            if v is not None:
                vals.append(v)
        return tuple(vals)


def load_llm_cache(set_name: str, path: Optional[str] = None) -> CachedEnsembleJudge:
    """Build a CachedEnsembleJudge from the serialized real-LLM ensemble.

    `set_name` selects a rule set ("ambiguous" / "sharp" / "conservative") in the
    JSON produced by saorl.build_plaus_cache. Raises if the cache is missing so a
    silent fall-back to hand-set scores can never masquerade as the real run.
    """
    p = Path(path) if path else Path(__file__).with_name("plausibility_cache.json")
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found -- run `python3 -m saorl.build_plaus_cache` first"
        )
    blob = json.loads(p.read_text())
    scores = blob["sets"][set_name]["scores"]
    return CachedEnsembleJudge({k: tuple(v) for k, v in scores.items()})


def attach_plausibility(
    cands: List[Candidate], judge: PlausibilityJudge
) -> List[Candidate]:
    """Return copies of `cands` with plausibility set from `judge`."""
    return [
        Candidate(c.name, c.predicate, c.forbidden_actions, judge.scores(c))
        for c in cands
    ]
