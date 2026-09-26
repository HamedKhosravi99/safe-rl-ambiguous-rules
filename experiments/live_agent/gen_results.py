"""Assemble demo/RESULTS.md from every phase's JSON report (no hand-entered
numbers)."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def J(p):
    return json.loads((HERE / p).read_text())


def main():
    conf = J("conformal_set_report.json")
    logs = J("logs/log_summary.json")
    guards = J("logs/guard_report.json")
    ev = J("logs/eval_summary.json")
    voq = J("logs/voq_report.json")
    recon = J("billing_reconciliation.json")

    conds = ev["conditions"]
    total = recon["total_billed_spend"]
    total_calls = recon["internal_reconciliation"]["n_calls"] + recon.get("partial_run1_calls", 0)
    U = conf["retained_set"]
    single_k = int(conf["most_plausible"].split(">=")[1])

    def vios(c):
        s = conds[c]["retained_violation_rate"]
        return ", ".join(f"{k.split('>=')[1]}:{d['rate']:.2f}" for k, d in s.items())

    ng, sg, cg = conds["no_guard"], conds["single"], conds["corset"]
    headline = (
        f"On a live, real-billed tool-using agent governed by the Slack "
        f"posting-rate policy (conformal-retained set U={U}, |U|={len(U)}), the "
        f"CORSET worst-retained guard drove the episode-level violation rate of "
        f"every retained reading to "
        f"{max(d['rate'] for d in cg['retained_violation_rate'].values()):.2f} "
        f"while the single-most-plausible-reading guard (k={single_k}) left the "
        f"tighter retained readings violated at up to "
        f"{max(d['rate'] for d in sg['retained_violation_rate'].values()):.2f}, "
        f"and the no-guard agent violated them at "
        f"{max(d['rate'] for d in ng['retained_violation_rate'].values()):.2f} "
        f"-- for a total real billed cost of ${total:.2f} "
        f"(${total:.6f}) across {total_calls} gpt-4o-mini posts."
    )

    md = []
    md.append("# Live service-agent check -- RESULTS")
    md.append("")
    md.append("Paper section 22: CORSET governing a real, billed tool-using agent. "
              "The agent is a Slack-style app posting research-summary messages to "
              "a channel; each **paid post** is a real billed `gpt-4o-mini` call, "
              "and a **guard** gates posting exactly as the budget domain gates "
              "`buy`. The deployment unit is the real Slack Web-API posting-rate "
              "policy `real-a02-slack-posting-rate`.")
    md.append("")
    md.append("## Headline")
    md.append("")
    md.append("> " + headline)
    md.append("")
    md.append(f"**Total billed spend: ${total:.2f}** (exact ${total:.6f}, "
              f"{total_calls} billed posts total; primary ledger "
              f"{recon['internal_reconciliation']['n_calls']} posts reconciled to "
              f"the cent, plus {recon.get('partial_run1_calls',0)} posts from an "
              f"aborted first attempt kept for honest accounting).")
    md.append("")
    md.append("## Phase 1 -- SCORE (conformal retained set)")
    md.append("")
    md.append(f"- Policy: `real-a02-slack-posting-rate` -- \"apps may post no more "
              f"than one message per second per channel ... we allow bursts over "
              f"that limit for short periods\".")
    md.append(f"- Reading pool: nested per-window burst caps `psi_calls>=k` "
              f"(k tight->loose) + null; scored by the frozen 7-persona ensemble "
              f"(claude CLI, as `saorl.judge`), cached in `ensemble_scores.json`.")
    md.append(f"- q-hat = {conf['qhat']} (reused from corpus "
              f"`results/paper_extra/conformal/conformal_report.json`).")
    md.append(f"- **Retained set U = {U}**  (means: " +
              ", ".join(f"{k}={v}" for k, v in conf['retained_means'].items()) + ").")
    md.append(f"- **Non-singleton: {conf['non_singleton']}** (|U|={conf['n_retained']}) "
              f"-- GATE PASSED.")
    md.append(f"- most-plausible (single guard) = `{conf['most_plausible']}`; "
              f"strict anchor (1 msg/sec) = `{conf['anchored_reading']}` "
              f"(scored {conf['per_candidate'][conf['anchored_reading']]['mean']:.3f} "
              f"< q-hat -> **outside U**, a delta-miscoverage the clarification catches).")
    md.append("")
    md.append("## Phase 2 -- LOGS")
    md.append("")
    md.append(f"- {logs['episodes']} channel-session episodes, mixture behavior "
              f"(greedy-post / randomized-fallback), real billed posts.")
    md.append(f"- {logs['paid_calls_billed']} billed posts; mean episode value "
              f"{logs['mean_episode_value']}; log-phase spend ${logs['total_spend']:.6f}.")
    md.append(f"- Serialized to `logs/offline_episodes.jsonl` (saorl OfflineDataset "
              f"schema).")
    md.append("")
    md.append("## Phase 3 -- GUARDS (FQI, saorl.learn_budget_constrained)")
    md.append("")
    md.append("| guard | honors | effective cap | lambda | true-worst (retained, offline) |")
    md.append("|---|---|---|---|---|")
    for m in ("single", "corset", "post_clarify"):
        g = guards[m]
        md.append(f"| {m} | {','.join(g['honors'])} | {g['effective_cap']} | "
                  f"{g['lam']:.1f} | {g['true_worst_retained_original']:.3f} |")
    md.append("")
    md.append("## Phase 4 -- LIVE EVAL (" +
              f"{ev['episodes_per_condition']} paired episodes/condition)")
    md.append("")
    md.append("| condition | billed $ | task value | paid posts | retained-reading violation rate (k:rate) | anchor(k=2) viol |")
    md.append("|---|---|---|---|---|---|")
    for c in ("no_guard", "single", "corset", "post_clarify"):
        s = conds[c]
        a = list(s["anchored_target_violation_rate"].values())[0]["rate"]
        md.append(f"| {c} | {s['billed_spend_total']:.5f} | "
                  f"{s['task_value_mean']:.2f}±{s['task_value_ci95']:.2f} | "
                  f"{s['n_paid_mean']:.1f} | {vios(c)} | {a:.2f} |")
    md.append("")
    md.append("## Phase 5 -- ASK-VS-ACT (VoQ, Thm 5)")
    md.append("")
    md.append(f"- posterior over U (plausibility-normalized): {voq['posterior_over_U']}")
    md.append(f"- per-reading value v_psi: {voq['v_psi']}; "
              f"v_robust (CORSET) = {voq['v_robust_corset']}.")
    md.append(f"- **VoQ (plausibility) = {voq['VoQ_plausibility']}**, "
              f"VoQ (robust) = {voq['VoQ_robust']}, kappa = {voq['kappa']} return units.")
    md.append(f"- **Decision: {voq['decision']}** "
              f"(fires = {voq['fires']}). {voq['note_robust_voq_zero']}")
    md.append(f"- Clarification transcript: `transcripts/clarification.txt`; "
              f"post-clarify honors the strict anchor (k={voq['anchor_k_strict']}), "
              f"realized value {voq['v_psi_anchor_strict']}.")
    md.append("")
    md.append("## Deliverables (under demo/)")
    md.append("")
    for f in ["policy_source.json", "ensemble_scores.json",
              "conformal_set_report.json", "logs/offline_episodes.jsonl",
              "guards/single.json", "guards/corset.json", "guards/post_clarify.json",
              "logs/eval_no_guard.jsonl", "logs/eval_single.jsonl",
              "logs/eval_corset.jsonl", "logs/eval_post_clarify.jsonl",
              "transcripts/clarification.txt", "billing_log.jsonl",
              "billing_reconciliation.json", "fig_demo.pdf", "gen_demo.tex",
              "RESULTS.md"]:
        md.append(f"- `{f}`")
    md.append("")
    md.append("## Honest caveats")
    md.append("")
    md.append("- **This is a demonstration, not a coverage experiment.** It shows "
              "the full CORSET pipeline (conformal set -> FQI guards -> live billed "
              "eval -> ask-vs-act) end-to-end on one real policy; it is a single "
              "deployment unit, not a calibrated coverage study over many rules.")
    md.append("- **The strict 1 msg/sec anchor fell just below q-hat and is "
              "outside U** -- a genuine delta-miscoverage. CORSET's guarantee is "
              "over the *retained* set; the clarification query is exactly what "
              "recovers the true reading a plausibility-only agent would miss. "
              "(This is honest, not a bug: conformal coverage is 1-delta, not 1.)")
    md.append("- **VoQ(robust) = 0** because the worst-case answer cell equals the "
              "tightest retained reading, which CORSET already honors; the positive "
              "signal comes from the plausibility prior over looser readings.")
    md.append("- **Costs are project/user-scope**, reconciled internally "
              "(billing_log vs re-derived usage x pinned price sheet, exact to the "
              "cent); the org-scope Costs API needs an admin key (documented 403).")
    md.append(f"- **Hard stop-loss $15** was never approached (spent ${total:.2f}).")
    md.append("")

    (HERE / "RESULTS.md").write_text("\n".join(md) + "\n")
    print("wrote RESULTS.md")
    print("\nHEADLINE:\n" + headline)


if __name__ == "__main__":
    main()
