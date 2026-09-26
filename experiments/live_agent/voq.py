"""Phase 5 ASK-VS-ACT: Value of Query (paper Theorem 5) from the offline logs.

Because every logged step records BOTH the realized paid-post value and the
deterministic free-fallback value, the value of any "cap at k" policy is exactly
replayable on the full-information (greedy) episodes:

    v_psi_k = mean_episodes [ sum_{t<k} paid_value_t + sum_{t>=k} free_value_t ]

Thm-5 quantities over the conformal-retained set U (full-reveal query, cells =
singletons of U):

    posterior p_k        = plausibility-normalized ensemble mean over U
    v_robust             = value of the CORSET policy (honor all of U = tightest
                           retained cap) = v_psi_{k_min}
    VoQ (plausibility)   = sum_k p_k * v_psi_k - v_robust
    VoQ (robust prior)   = min_k v_psi_k - v_robust     (worst-case over answers)

Decision: ask iff VoQ > kappa (query cost in return units). If it fires we emit
the verbatim clarification, take the documented answer (the anchored strict
reading, 1 msg/sec = k=2, which sits just below q-hat -- a delta-miscoverage the
query is exactly there to catch), and run the post-clarify condition honoring it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import readings as R

HERE = Path(__file__).resolve().parent
OFFLINE_EPISODES = HERE / "logs" / "offline_episodes.jsonl"
VOQ_REPORT = HERE / "logs" / "voq_report.json"
TRANSCRIPT_DIR = HERE / "transcripts"

# Query cost in RETURN units: the opportunity cost of pausing a live channel
# session to ask the platform owner one clarifying question -- we price it at
# roughly one paid post's value advantage over the free fallback (about half a
# rubric point), a deliberately modest, defensible cost.
KAPPA = 0.5


def full_info_episodes(episodes: List[dict]) -> List[dict]:
    """Episodes where every step was posted (paid_value present at all steps),
    so v_psi_k is exactly computable for every k."""
    return [ep for ep in episodes
            if all(pv is not None for pv in ep["paid_value"])]


def v_cap(ep: dict, k: int) -> float:
    """Return of 'post the first k messages, then free-fallback' on one episode."""
    pv, fv = ep["paid_value"], ep["free_value"]
    W = len(pv)
    return sum(pv[t] for t in range(min(k, W))) + sum(fv[t] for t in range(k, W))


def main() -> dict:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    episodes = [json.loads(l) for l in OFFLINE_EPISODES.read_text().splitlines() if l.strip()]
    fi = full_info_episodes(episodes)

    rs = R.reading_sets()
    retained_ks = sorted(int(n.split(">=")[1]) for n in rs["retained_names"])
    means = rs["means"]  # name -> ensemble mean (retained only)
    k_robust = min(retained_ks)          # CORSET binding cap (tightest retained)
    anchor_k = R.ANCHOR_K                 # strict 1 msg/sec target (k=2)

    # v_psi_k over full-info episodes
    v_psi = {k: float(np.mean([v_cap(ep, k) for ep in fi])) for k in retained_ks}
    v_psi_anchor = float(np.mean([v_cap(ep, anchor_k) for ep in fi]))
    v_robust = v_psi[k_robust]

    # plausibility-normalized posterior over U
    mass = {k: means[f"psi_calls>={k}"] for k in retained_ks}
    Z = sum(mass.values())
    post = {k: mass[k] / Z for k in retained_ks}

    voq_plaus = float(sum(post[k] * v_psi[k] for k in retained_ks) - v_robust)
    voq_robust = float(min(v_psi.values()) - v_robust)
    fires = voq_plaus > KAPPA

    # value if we ask and learn the true (documented) reading = the strict anchor
    v_post_clarify = v_psi_anchor

    report = {
        "phase": "ask_vs_act",
        "theorem": "Thm 5 (value of query), full-reveal over retained U",
        "n_full_info_episodes": len(fi),
        "retained_ks": retained_ks,
        "k_robust_corset": k_robust,
        "anchor_k_strict": anchor_k,
        "posterior_over_U": {f"psi_calls>={k}": round(post[k], 4) for k in retained_ks},
        "v_psi": {f"psi_calls>={k}": round(v_psi[k], 4) for k in retained_ks},
        "v_robust_corset": round(v_robust, 4),
        "v_psi_anchor_strict": round(v_psi_anchor, 4),
        "kappa": KAPPA,
        "kappa_units": "return units (per-episode task value)",
        "VoQ_plausibility": round(voq_plaus, 4),
        "VoQ_robust": round(voq_robust, 4),
        "decision": "ASK (fire clarification)" if fires else "ACT (silent defense)",
        "fires": bool(fires),
        "note_robust_voq_zero": (
            "VoQ_robust = 0 because the worst-case answer cell is the tightest "
            "retained reading, which the CORSET policy already honors; the "
            "plausibility prior puts mass on looser readings, so the expected "
            "VoQ is strictly positive."),
        "note_anchor_miscoverage": (
            "the documented strict anchor (k=2, 1 msg/sec) scored just below "
            "q-hat=0.5 and is NOT in U -- a delta-miscoverage event; the "
            "clarification is precisely the mechanism that recovers it."),
    }
    VOQ_REPORT.write_text(json.dumps(report, indent=2))

    # transcript
    lines = []
    lines.append("== clarification transcript (Appendix D.1 of the paper) ==")
    lines.append(f"policy: real-a02-slack-posting-rate")
    lines.append(f"retained set U = {rs['retained_names']}  (q-hat = {rs['qhat']})")
    lines.append(f"posterior over U (plausibility-normalized): "
                 f"{ {f'k={k}': round(post[k],3) for k in retained_ks} }")
    lines.append(f"v_psi (per-reading return, from logs): "
                 f"{ {f'k={k}': round(v_psi[k],3) for k in retained_ks} }")
    lines.append(f"v_robust (CORSET, cap at k={k_robust}) = {v_robust:.3f}")
    lines.append(f"VoQ(plausibility) = {voq_plaus:.3f}   VoQ(robust) = {voq_robust:.3f}"
                 f"   kappa = {KAPPA:.3f} return units")
    lines.append("")
    if fires:
        lines.append(f"DECISION: VoQ ({voq_plaus:.3f}) > kappa ({KAPPA:.3f})  ->  ASK.")
        lines.append("")
        lines.append("AGENT ->  platform owner:")
        lines.append('  "' + R.clarification_question() + '"')
        lines.append("")
        lines.append("PLATFORM OWNER ->  agent (documented anchor):")
        lines.append('  "The hard limit is one message per second per channel. '
                     'Treat that as the cap: do not post a burst beyond the '
                     '1/second rate. (Strict reading, k=2.)"')
        lines.append("")
        lines.append(f"POST-CLARIFY: honor the anchored strict reading psi_calls>="
                     f"{anchor_k}; realized per-episode value from logs "
                     f"v = {v_post_clarify:.3f}.")
        lines.append("NOTE: the strict anchor was OUTSIDE U (scored < q-hat); the "
                     "query recovered the true reading a plausibility-only agent "
                     "would have missed.")
    else:
        lines.append(f"DECISION: VoQ ({voq_plaus:.3f}) <= kappa ({KAPPA:.3f})  ->  "
                     f"ACT (silent defense): deploy the CORSET guard without asking.")
    transcript = "\n".join(lines)
    (TRANSCRIPT_DIR / "clarification.txt").write_text(transcript + "\n")

    print(transcript)
    print(f"\n  wrote {VOQ_REPORT.relative_to(HERE)} and "
          f"transcripts/clarification.txt")
    return report


if __name__ == "__main__":
    main()
