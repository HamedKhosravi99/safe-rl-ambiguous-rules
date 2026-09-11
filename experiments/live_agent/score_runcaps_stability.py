"""Stability of the run-cap Keep closure: replicate the frozen panel N times.

The registered closure (score_runcaps.py) ran the frozen 7-persona panel
ONCE, as the frozen protocol defines the score; run<=3 cleared the corpus
threshold by 0.007.  This script measures the sampling noise of that
protocol: N_PANELS fresh replicate panels through the byte-identical
machinery.  The registered verdict is NOT re-averaged or replaced -- the
protocol froze one-panel scoring, and swapping the aggregation after
seeing a result is the score-optimization failure this project has already
paid for once.  Replicates are reported as dispersion around the
registered run.

Two decision-relevant bits, kept separate:
  (a) GUARD SAFETY needs only run<=2 to stay rejected: the deployed
      guard's schedule violates run<=2 alone, and a cap-3 schedule can
      contain no 4-run, so retention of run<=3 / run<=4 cannot condemn it.
  (b) NON-NESTEDNESS of the calibrated pool needs run<=3 retained; this
      is the knife-edge quantity.

Run:  python3 demo/score_runcaps_stability.py
Writes demo/ensemble_scores_runcaps_stability.json and
       results/e2e/live_crossing_keep_stability.json
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

import readings as R                       # noqa: E402
import score_readings as S                 # noqa: E402
import score_runcaps as SR                 # noqa: E402  (frozen glosses reused)
from saorl.judge import ENSEMBLE_PERSONAS  # noqa: E402

OUT_RAW = HERE / "ensemble_scores_runcaps_stability.json"
OUT = REPO_ROOT / "results/e2e" / "live_crossing_keep_stability.json"

N_PANELS = 5


def one_panel(names: List[str]) -> Dict[str, List[float]]:
    jobs = [(nm, j, persona) for nm in names
            for j, persona in enumerate(ENSEMBLE_PERSONAS)]
    res: Dict[str, List[Optional[float]]] = {
        nm: [None] * len(ENSEMBLE_PERSONAS) for nm in names}
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        fut = {ex.submit(S._ask, persona, nm): (nm, j)
               for (nm, j, persona) in jobs}
        for f in cf.as_completed(fut):
            nm, j = fut[f]
            res[nm][j] = f.result()
    bad = sum(1 for v in res.values() for x in v if x is None)
    assert bad == 0, f"{bad} persona calls failed; a partial panel is not " \
        "a replicate"
    return {nm: [float(x) for x in v] for nm, v in res.items()}


def main() -> None:
    glosses = {SR._run_name(b): SR._run_gloss(b) for b in SR.RUN_CAPS}
    orig = R.gloss
    R.gloss = lambda nm: glosses[nm] if nm in glosses else orig(nm)
    names = list(glosses)
    panels = []
    for i in range(N_PANELS):
        panels.append(one_panel(names))
        print(f"panel {i + 1}/{N_PANELS} done")
    R.gloss = orig

    qhat = S.load_qhat()
    reg = json.loads((HERE / "ensemble_scores_runcaps.json").read_text())
    reg_means = {nm: reg["per_candidate"][nm]["mean"] for nm in names}

    summary = {}
    for nm in names:
        means = [float(np.mean(p[nm])) for p in panels]
        summary[nm] = dict(
            registered_mean=reg_means[nm],
            replicate_means=[round(m, 6) for m in means],
            lo=round(min(means), 6), hi=round(max(means), 6),
            retained_in=sum(1 for m in means if m >= qhat),
            n_panels=N_PANELS)

    r2_rejected_all = summary["psi_run>=2"]["retained_in"] == 0
    r3_in = summary["psi_run>=3"]["retained_in"]

    OUT_RAW.write_text(json.dumps(dict(
        date=datetime.date.today().isoformat(),
        cli_version=subprocess.run(["claude", "--version"],
                                   capture_output=True,
                                   text=True).stdout.strip(),
        n_panels=N_PANELS, panels=panels), indent=1), encoding="utf8")
    OUT.write_text(json.dumps(dict(
        registration=("replicate panels around the registered run-cap "
                      "closure; the registered verdict is the first panel's "
                      "and is not re-averaged"),
        qhat=qhat, n_panels=N_PANELS, per_candidate=summary,
        guard_safety_stable=bool(r2_rejected_all),
        r3_retained_in=int(r3_in)), indent=1), encoding="utf8")

    print(f"wrote {OUT}")
    for nm in names:
        s = summary[nm]
        print(f"  {nm}: registered {s['registered_mean']}, replicates "
              f"{s['replicate_means']} -> retained in {s['retained_in']}"
              f"/{N_PANELS}")
    print("guard-safety stable (r2 rejected in all panels):", r2_rejected_all)
    print(f"r3 retained in {r3_in}/{N_PANELS} replicate panels")


if __name__ == "__main__":
    main()
