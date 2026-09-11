"""Phase 6 FIGURE: demo/fig_demo.pdf (paper section 22, phase 6).

Two panels, all numbers from the live-eval episode logs (logs/eval_*.jsonl) and
the demo-wide billing records:

  (a) cumulative PAID POSTS (the quota axis) over the W=12 posting steps of a
      channel window, mean +- 95% band per condition; the retained caps and the
      strict anchor drawn as horizontal lines; the clarification event marked;
  (b) cumulative BILLED DOLLARS over the same steps, mean per condition (real $).

No hand-entered numbers: everything is recomputed from the JSONL logs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import readings as R  # noqa: E402

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE / "logs"
FIG = HERE / "fig_demo.pdf"

CONDS = ["no_guard", "single", "corset", "post_clarify"]
LABEL = {"no_guard": "no-guard (passthrough)", "single": "single (k=5)",
         "corset": "CORSET worst-retained (k=3)", "post_clarify": "post-clarify (k=2)"}
COLOR = {"no_guard": "#7f7f7f", "single": "#1f77b4",
         "corset": "#d62728", "post_clarify": "#2ca02c"}


def load(cond):
    return [json.loads(l) for l in (EVAL_DIR / f"eval_{cond}.jsonl").read_text().splitlines() if l.strip()]


def cum_curves(eps):
    """Per-episode cumulative paid-posts and cumulative $ over steps -> (mean,ci)."""
    W = len(eps[0]["actions"])
    posts = np.zeros((len(eps), W))
    dollars = np.zeros((len(eps), W))
    for i, ep in enumerate(eps):
        c = 0.0
        d = 0.0
        for t in range(W):
            if ep["actions"][t] == "buy":
                c += 1
            d += ep["costs"][t]
            posts[i, t] = c
            dollars[i, t] = d
    def mc(a):
        m = a.mean(0)
        ci = 1.96 * a.std(0, ddof=1) / np.sqrt(a.shape[0])
        return m, ci
    return mc(posts), mc(dollars), W


def main():
    rs = R.reading_sets()
    retained_ks = sorted(int(n.split(">=")[1]) for n in rs["retained_names"])
    anchor_k = R.ANCHOR_K
    single_k = int(rs["single_name"].split(">=")[1])

    data = {c: load(c) for c in CONDS}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # panel (a): cumulative paid posts (quota axis)
    for c in CONDS:
        (pm, pci), _, W = cum_curves(data[c])
        x = np.arange(1, W + 1)
        ax1.plot(x, pm, color=COLOR[c], label=LABEL[c], lw=2)
        ax1.fill_between(x, pm - pci, pm + pci, color=COLOR[c], alpha=0.15)
    for k in retained_ks:
        ls = "--" if k == single_k else ":"
        lw = 1.6 if k == single_k else 1.0
        ax1.axhline(k, color="#555555", ls=ls, lw=lw, alpha=0.7)
        ax1.text(W + 0.05, k, f" $\\psi_{{k={k}}}$" + (" (single)" if k == single_k else ""),
                 va="center", fontsize=7, color="#333333")
    ax1.axhline(anchor_k, color="#2ca02c", ls="-.", lw=1.6, alpha=0.9)
    ax1.text(W + 0.05, anchor_k, f" anchor k={anchor_k}\n (1 msg/sec)",
             va="center", fontsize=7, color="#2ca02c")
    ax1.annotate("clarification\nfired (VoQ>$\\kappa$)", xy=(W, anchor_k),
                 xytext=(W - 5.5, anchor_k + 3.2), fontsize=7.5, color="#2ca02c",
                 arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2))
    ax1.set_xlabel("posting step within channel window")
    ax1.set_ylabel("cumulative paid posts (quota axis)")
    ax1.set_title("(a) posting quota: retained caps bind the guards")
    ax1.set_xlim(1, W + 2.6)
    ax1.legend(fontsize=7.5, loc="upper left")
    ax1.grid(alpha=0.2)

    # panel (b): cumulative billed dollars
    for c in CONDS:
        _, (dm, dci), W = cum_curves(data[c])
        x = np.arange(1, W + 1)
        ax2.plot(x, dm * 1e3, color=COLOR[c], label=LABEL[c], lw=2)
        ax2.fill_between(x, (dm - dci) * 1e3, (dm + dci) * 1e3,
                         color=COLOR[c], alpha=0.15)
    ax2.set_xlabel("posting step within channel window")
    ax2.set_ylabel("cumulative billed cost (milli-USD, $10^{-3}$ USD)")
    ax2.set_title("(b) real billed spend per condition")
    ax2.set_xlim(1, W)
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.grid(alpha=0.2)

    fig.suptitle("E8: CORSET governing a live billed agent "
                 "(Slack posting-rate policy)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG)
    print(f"wrote {FIG.relative_to(HERE)}")


if __name__ == "__main__":
    main()
