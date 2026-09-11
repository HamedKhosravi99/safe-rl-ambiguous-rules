"""Phase 1 SCORE: score the frozen reading pool against the Slack posting-rate
policy text with the frozen 7-persona ensemble (claude CLI, exactly as
saorl.judge.LLMEnsembleJudge), then apply the corpus conformal threshold
q-hat = 0.5 to obtain the retained set U (paper section 22, phase 1).

  * ensemble = the 7 frozen ENSEMBLE_PERSONAS from saorl.judge;
  * one claude call per (persona, candidate); each reply is a single number in
    [0,1] measuring how faithfully the capped reading captures the rule;
  * composite score s(psi) = mean of the 7 persona scores (no data-calibration
    gate: this demo rule has no incident/normal semantic split, so -- exactly as
    the DSRL fresh-rule case in saorl.conformal -- the score reduces to the
    language term; Theorem 1 is score-agnostic, only efficiency is affected);
  * q-hat = 0.5 reused from the corpus (results/paper_extra/conformal/conformal_report
    .json); the demo rule is a fresh deployment unit exchangeable with the corpus
    rows, so its corpus-calibrated threshold applies (paper section 22, phase 1);
  * retained U = {psi : s(psi) >= q-hat}.

Cache: writes demo/ensemble_scores.json (per-persona scores) so U is
reproducible from the cache with no further model calls, and
demo/conformal_set_report.json (the retained-set report + gate).

GATE (paper section 22 phase 1): U must be non-singleton (|U| >= 2).
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from saorl.judge import ENSEMBLE_PERSONAS, _parse_score  # noqa: E402

import readings as R  # noqa: E402

ENSEMBLE_SCORES = HERE / "ensemble_scores.json"
CONFORMAL_REPORT = HERE / "conformal_set_report.json"
CORPUS_CONFORMAL = REPO_ROOT / "results/paper_extra" / "conformal" / "conformal_report.json"

GLOSSARY = (
    "Setting: an automated app posting messages to a chat channel through a Web "
    "API. The documented hard limit is 1 message per second per channel, and the "
    "policy separately allows short bursts above that limit for short periods. "
    "The metered action is posting one message (one API call). Within one short "
    "burst window the app may attempt several posts; a reading caps how many "
    "messages the burst may contain before the app must throttle back to the 1 "
    "message/second limit. A burst cap of k = 2 is essentially the hard 1/second "
    "limit with no meaningful burst; larger k honors a larger burst allowance."
)


def _prompt(persona: str, cand_name: str) -> str:
    return (
        f"You are {persona}.\n\n"
        f"{GLOSSARY}\n\n"
        f"Natural-language API posting-rate rule:\n  \"{R.RULE_TEXT}\"\n\n"
        f"A proposed formal reading of that rule forbids the 'post a message' "
        f"action exactly when this condition holds:\n"
        f"  {R.gloss(cand_name)}\n\n"
        f"On a scale from 0 to 1, how faithfully does this condition capture "
        f"what the rule actually means? 1.0 = an exact, faithful reading; "
        f"0.5 = defensible but imperfect; 0.0 = clearly the wrong reading "
        f"(far too permissive, far too strict, or unrelated).\n"
        f"Reply with ONLY a single number between 0 and 1, nothing else."
    )


def _ask(persona: str, cand_name: str, cli: str = "claude",
         timeout_s: float = 120.0) -> Optional[float]:
    cmd = [cli, "-p", _prompt(persona, cand_name)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout_s, stdin=subprocess.DEVNULL)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return _parse_score(out.stdout)


def score_pool(max_workers: int = 8) -> Dict[str, List[float]]:
    """One claude call per (persona, candidate); returns name -> [7 scores]."""
    pool = R.build_pool()
    names = [c.name for c in pool]
    jobs: List[Tuple[str, int, str]] = [
        (nm, j, persona) for nm in names
        for j, persona in enumerate(ENSEMBLE_PERSONAS)
    ]
    results: Dict[str, List[Optional[float]]] = {nm: [None] * len(ENSEMBLE_PERSONAS)
                                                 for nm in names}
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut = {ex.submit(_ask, persona, nm): (nm, j)
               for (nm, j, persona) in jobs}
        for f in cf.as_completed(fut):
            nm, j = fut[f]
            results[nm][j] = f.result()
    # drop any None (failed call) so a missing score never masquerades as 0
    clean: Dict[str, List[float]] = {}
    for nm, vals in results.items():
        clean[nm] = [float(v) for v in vals if v is not None]
    return clean


def load_qhat() -> float:
    """Reuse the corpus conformal threshold q-hat = 0.5 (paper section 22)."""
    rep = json.loads(CORPUS_CONFORMAL.read_text())
    # the canonical corpus threshold is the full-corpus q-hat (the modal LOO
    # value is identical); the demo reuses it as a fresh exchangeable rule.
    qh = float(rep["qhat_full_corpus"])
    assert abs(qh - 0.5) < 1e-9, f"unexpected corpus q-hat {qh}"
    return 0.5


def build_report(scores: Dict[str, List[float]], qhat: float) -> dict:
    pool = R.build_pool()
    means = {c.name: (float(np.mean(scores[c.name])) if scores.get(c.name) else 0.0)
             for c in pool}
    retained = sorted([nm for nm, m in means.items() if m >= qhat],
                      key=lambda n: (n != R.BOTTOM_NAME, n))
    # order retained tight->loose by cap k (bottom last)
    def _k(nm: str) -> float:
        return R.NEVER_FIRES if nm == R.BOTTOM_NAME else int(nm.split(">=")[1])
    retained = sorted(retained, key=_k)
    most_plausible = max(retained, key=lambda nm: means[nm]) if retained else None
    return {
        "policy": "real-a02-slack-posting-rate",
        "rule_text": R.RULE_TEXT,
        "qhat": qhat,
        "qhat_source": "corpus results/paper_extra/conformal/conformal_report.json (reused; demo rule is a fresh exchangeable deployment unit)",
        "n_personas": len(ENSEMBLE_PERSONAS),
        "personas": list(ENSEMBLE_PERSONAS),
        "per_candidate": {
            c.name: {
                "gloss": R.gloss(c.name),
                "scores": scores.get(c.name, []),
                "mean": round(means[c.name], 6),
                "retained": means[c.name] >= qhat,
            } for c in pool
        },
        "retained_set": retained,
        "retained_means": {nm: round(means[nm], 6) for nm in retained},
        "most_plausible": most_plausible,
        "anchored_reading": R._cap_name(R.ANCHOR_K),
        "n_retained": len(retained),
        "non_singleton": len(retained) >= 2,
    }


def main() -> dict:
    print("== Phase 1 SCORE: 7-persona ensemble over the reading pool ==")
    if ENSEMBLE_SCORES.exists():
        scores = json.loads(ENSEMBLE_SCORES.read_text())["scores"]
        print(f"  (loaded cached ensemble scores from {ENSEMBLE_SCORES.name})")
    else:
        scores = score_pool()
        ENSEMBLE_SCORES.write_text(json.dumps(
            {"policy": "real-a02-slack-posting-rate",
             "personas": list(ENSEMBLE_PERSONAS),
             "scores": scores}, indent=2))
        print(f"  wrote {ENSEMBLE_SCORES.name}")
    qhat = load_qhat()
    report = build_report(scores, qhat)
    CONFORMAL_REPORT.write_text(json.dumps(report, indent=2))

    print(f"  q-hat = {qhat} (reused from corpus)")
    for nm, d in report["per_candidate"].items():
        flag = "RETAINED" if d["retained"] else "dropped"
        print(f"    {nm:16s} mean={d['mean']:.3f}  {flag}")
    print(f"  retained set U = {report['retained_set']}")
    print(f"  |U| = {report['n_retained']}  non-singleton={report['non_singleton']}")
    print(f"  most-plausible = {report['most_plausible']}   "
          f"anchored(strict) = {report['anchored_reading']}")
    if not report["non_singleton"]:
        print("  ** GATE FAILED: singleton retained set -- widen (k,W) grid or "
              "switch to the DockerHub backup policy (real-a07). **")
    else:
        print("  ** GATE PASSED: non-singleton retained set. **")
    print(f"  wrote {CONFORMAL_REPORT.name}")
    return report


if __name__ == "__main__":
    main()
