"""Score the run-cap (burst) readings with the FROZEN 7-persona ensemble.

POST-HOC (2026-09-01), closing the Keep gap of replay_crossing.py: the
run-length candidates entered the crossed pool stipulated as defensible;
this scores them through the SAME frozen scorer the original pool went
through -- same personas, same prompt template, same CLI, same parse, same
composite (persona mean), same corpus threshold q-hat = 0.5.

Fidelity mechanics: score_readings._prompt is reused AS-IS by monkeypatching
readings.gloss to also resolve the three new candidate names, so the prompt
surrounding the gloss is byte-identical to the frozen one.  The frozen
cache (ensemble_scores.json) and the frozen report are NOT touched; scores
land in ensemble_scores_runcaps.json and results/e2e/live_crossing_keep.json.

One honest drift is unavoidable and recorded: the original pool was scored
in July through the then-current `claude` CLI; these calls go through the
CLI installed today.  The frozen protocol pins personas/prompt/parse/
threshold, not a model snapshot -- score_readings.py itself just invokes
`claude`.  The CLI version is recorded in the output.

Run:  python3 demo/score_runcaps.py
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import readings as R                       # noqa: E402  (frozen pool module)
import score_readings as S                 # noqa: E402  (frozen scorer)
from saorl.judge import ENSEMBLE_PERSONAS  # noqa: E402

OUT_SCORES = HERE / "ensemble_scores_runcaps.json"
OUT_KEEP = REPO_ROOT / "results/e2e" / "live_crossing_keep.json"

RUN_CAPS = (2, 3, 4)


def _run_name(b: int) -> str:
    return f"psi_run>={b}"


def _run_gloss(b: int) -> str:
    return (f"allow bursts of at most {b} messages in a row -- i.e. forbid "
            f"posting exactly when the app has already posted in each of the "
            f"last {b} consecutive posting opportunities, so a burst must "
            f"pause after {b} back-to-back messages (equivalently, at most "
            f"{b} messages in any {b + 1} consecutive opportunities); the "
            f"total number of messages in the window is otherwise unlimited")


def main() -> None:
    # resolve the new names through the frozen prompt template verbatim
    glosses = {_run_name(b): _run_gloss(b) for b in RUN_CAPS}
    orig_gloss = R.gloss
    R.gloss = lambda nm: glosses[nm] if nm in glosses else orig_gloss(nm)

    names = [_run_name(b) for b in RUN_CAPS]
    jobs = [(nm, j, persona) for nm in names
            for j, persona in enumerate(ENSEMBLE_PERSONAS)]
    results: Dict[str, List[Optional[float]]] = {
        nm: [None] * len(ENSEMBLE_PERSONAS) for nm in names}
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        fut = {ex.submit(S._ask, persona, nm): (nm, j)
               for (nm, j, persona) in jobs}
        for f in cf.as_completed(fut):
            nm, j = fut[f]
            results[nm][j] = f.result()
    R.gloss = orig_gloss

    n_failed = sum(1 for vals in results.values() for v in vals if v is None)
    assert n_failed == 0, f"{n_failed} persona calls failed; rerun, do not " \
        "average over a partial panel"

    qhat = S.load_qhat()
    cli_version = subprocess.run(["claude", "--version"], capture_output=True,
                                 text=True).stdout.strip()
    per = {}
    for nm in names:
        vals = [float(v) for v in results[nm]]
        per[nm] = dict(gloss=glosses[nm], scores=vals,
                       mean=round(float(np.mean(vals)), 6),
                       retained=bool(np.mean(vals) >= qhat))
    retained = [nm for nm in names if per[nm]["retained"]]

    OUT_SCORES.write_text(json.dumps(dict(
        registration=("post-hoc run-cap scoring through the frozen ensemble; "
                      "frozen cache untouched"),
        date=datetime.date.today().isoformat(), cli_version=cli_version,
        qhat=qhat, personas=list(ENSEMBLE_PERSONAS), per_candidate=per,
    ), indent=1), encoding="utf8")

    # keep-report for the crossed pool: original retained set + the new arm
    frozen = json.loads((HERE / "conformal_set_report.json").read_text())
    OUT_KEEP.write_text(json.dumps(dict(
        registration=("Keep closure for the crossed live pool: run caps "
                      "scored 2026-09-01 through the frozen 7-persona "
                      "ensemble, prompt template byte-identical (gloss "
                      "resolver extended at runtime), corpus qhat reused; "
                      "CLI drift disclosed"),
        cli_version=cli_version, qhat=qhat,
        frozen_retained=frozen["retained_set"],
        runcap_per_candidate=per, runcap_retained=retained,
        n_runcap_retained=len(retained),
    ), indent=1), encoding="utf8")

    print(f"wrote {OUT_SCORES.name} and {OUT_KEEP}")
    for nm in names:
        print(f"  {nm}: scores={per[nm]['scores']} mean={per[nm]['mean']}"
              f" retained={per[nm]['retained']}")
    print("runcap retained:", retained, f" (qhat={qhat})")


if __name__ == "__main__":
    main()
